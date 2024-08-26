import os
import time
import json
import random
import argparse
import datetime
import tqdm, pdb
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.onnx
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
import torch.nn as nn
import torch.nn.functional as F

from config import get_config
from models import build_model
from data import build_loader
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.logger import create_logger
from utils.utils import  NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor
from utils.utils import load_checkpoint_ema, load_pretrained_ema, save_checkpoint_ema

from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count

from timm.utils import ModelEma as ModelEma

from torchvision import transforms
from PIL import Image

from models.csms6s import CrossScan, CrossMerge

# from models.vmamba import selective_scan_ref, CrossScan_onnx, mamba_init, selective_scan_easy

from models.vmamba import selective_scan_ref, mamba_init, selective_scan_easy

import onnxruntime
import onnx
import math

# from onnxsim import simplify
# from onnx import shape_inference
import onnxoptimizer

from collections import Counter

from torch.autograd import Function
from torch.onnx import OperatorExportTypes


class CustomSelectiveScan(Function):
    @staticmethod
    def forward(ctx, xs, dts, As, Bs, Cs, Ds, delta_bias, chunksize):
        output = selective_scan_easy(xs, dts, As, Bs, Cs, Ds, delta_bias, chunksize=chunksize)
        
        return output
    
    @staticmethod
    def symbolic(g: torch.Graph, xs, dts, As, Bs, Cs, Ds, delta_bias, chunksize):
        return g.op("SelectiveScan", xs, dts, As, Bs, Cs, Ds, delta_bias, chunksize_i=chunksize)


def CrossScan_onnx(x: torch.Tensor):
    # B, C, H, W = x.shape
    B, C, H, W = map(int, x.shape)

    ### original code
    # xs = x.new_empty((B, 4, C, H * W))
    # xs[:, 0] = x.flatten(2, 3)
    # xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
    # xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

    ### modify by lihaokun 20240709
    xs_0 = x.flatten(2, 3)
    xs_1 = x.transpose(dim0=2, dim1=3).flatten(2, 3)
    xs_01 = torch.stack((xs_0, xs_1), dim=1)
    xs_23 = torch.flip(xs_01, dims=[-1])
    xs = torch.cat((xs_01, xs_23), dim=1)

    return xs

class sub_model(nn.Module):
    def __init__(self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,):
        super().__init__()

        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        
        k_group = 4
        
        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            mamba_init.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(4)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = mamba_init.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        self.Ds = mamba_init.D_init(d_inner, copies=k_group, merge=True) # (K * D)

        print("finish init sub_model")

    def forward(self, x=None):

        print("forward begin")

        # 定义开始和结束事件
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias

        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True

        B, D, H, W = map(int, x.shape)
        D, N = map(int, A_logs.shape)
        K, D, R = map(int, dt_projs_weight.shape)
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, chunksize=32):
            # ### original code
            # return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex)

            # selective_scan_dict = torch.load('selective_scan_dict.pt')

            # u = selective_scan_dict["u"]
            # delta = selective_scan_dict["delta"]
            # A = selective_scan_dict["A"]
            # B = selective_scan_dict["B"]
            # C = selective_scan_dict["C"]
            # D = selective_scan_dict["D"]
            # delta_bias = selective_scan_dict["delta_bias"]
            
            # print("finish load selective_scan_dict")
        
            # for item in [u, delta, A, B, C, D, delta_bias]:
            #     print("shape:{} mean:{} min:{} max:{} var:{}".format(item.size(), item.mean(), item.min(), item.max(), torch.var(item)))

            ### modify by lihaokun
            # result_original = selective_scan_ref(u, delta, A, B, C, D, delta_bias, delta_softplus, True)

            result =  selective_scan_easy(u, delta, A, B, C, D, delta_bias, delta_softplus, False, chunksize)

            # difference = result_original-result
            # relative_error = difference/result_original
            # print("difference:{}".format(difference))
            # print("difference mean:{}  max:{}  min:{}".format(difference.abs().mean(), difference.max(), difference.min()))
            # print("relative_error mean:{}  max:{}  min:{}".format(relative_error.abs().mean(), relative_error.max(), relative_error.min()))
        
            return result



        # 记录开始时间
        start_event.record()

        xs = CrossScan_onnx(x)
        # xs = CrossScan.apply(x)

        # 记录结束时间
        end_event.record()

        # 等待所有流上的所有先前GPU任务完成
        torch.cuda.synchronize()

        # 计算耗时
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"CrossScan_onnx time: {elapsed_time_ms} ms")

        # 记录开始时间
        start_event.record()

        x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # 记录结束时间
        end_event.record()

        # 等待所有流上的所有先前GPU任务完成
        torch.cuda.synchronize()

        # 计算耗时
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"After CrossScan_onnx, before selective_scan time: {elapsed_time_ms} ms")

        # 记录开始时间
        start_event.record()

        # selective_scan 里有一个for循环，循环次数与维度L正相关。随着L增大，onnx的图就会迅速增大。
        # 维度L为16*16时，Netron可视化submodel的onnx需要大概五分钟的渲染时间。
        if not args.custom_operator:
                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, args.chunksize
                ).view(B, K, -1, H, W)

        # modify by lihaokun in 20240826 to add onnx node for selective_scan
        else:
            ys: torch.Tensor = CustomSelectiveScan.apply(
                xs, dts, As, Bs, Cs, Ds, delta_bias, args.chunksize).view(B, K, -1, H, W)

        # 记录结束时间
        end_event.record()

        # 等待所有流上的所有先前GPU任务完成
        torch.cuda.synchronize()

        # 计算耗时
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Selective_scan time: {elapsed_time_ms} ms")

        # 记录开始时间
        start_event.record()

        y: torch.Tensor = CrossMerge.apply(ys)
    
        # 记录结束时间
        end_event.record()

        # 等待所有流上的所有先前GPU任务完成
        torch.cuda.synchronize()

        # 计算耗时
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"CrossMerge time: {elapsed_time_ms} ms")

        # 记录开始时间
        start_event.record()

        y = y.view(B, -1, H, W) 

        # 记录结束时间
        end_event.record()

        # 等待所有流上的所有先前GPU任务完成
        torch.cuda.synchronize()

        # 计算耗时
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"After CrossMerge time: {elapsed_time_ms} ms")
        
        
        return y


def parse_option():
    parser = argparse.ArgumentParser('test sub model convet to onnx', add_help=False)
    parser.add_argument('--cfg', type=str, default="/home/test/code/VMamba-main/classification/configs/vssm/vmambav2_tiny_224.yaml", metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int,default=1, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default="/mnt/hdd/Datasets/imagenet/", help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],help='no: no cache, '
                             'full: cache all data, ''part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained', default= "/home/test/code/VMamba-main/vssm_tiny_0230_ckpt_epoch_262.pth",
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='/tmp', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default=time.strftime("%Y%m%d%H%M%S", time.localtime()), help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='eval on imagenet')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    parser.add_argument('--optim', type=str, help='overwrite optimizer if provided, can be adamw/sgd.')

    parser.add_argument('--memory_limit_rate', type=float, default=-1, help='limitation of gpu memory use')

    parser.add_argument('--convert_to_onnx', action='store_true',
                        help='whether to convert to onnx model')
    
    parser.add_argument('--onnx_path', type=str, default='submodel_lhk.onnx',
                        help='save as xxx.onnx')
    
    parser.add_argument('--simplify_onnx_path', type=str, default='')

    parser.add_argument('--token_H_W', type=int,default=56, help="token number for H and W")

    parser.add_argument('--chunksize', type=int,default=32, help="chunksize")

    parser.add_argument('--custom_operator', action='store_true',
                        help='whether to use custom_operator for onnx')


    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, args):
    # build submodel
    model = sub_model().cuda()
    image_tensor = torch.randint(-5, 5, [args.batch_size, 192, args.token_H_W, args.token_H_W]).float().cuda() / 50 # [B, C, H, W]

    # # 模型转换为onnx
    # torch.onnx.export(model,               # 运行的模型
    #         image_tensor.cuda(),                  # 模型输入（通常是dummy input）
    #         args.onnx_path,       # 输出ONNX文件的路径
    #         export_params=True, # 是否导出模型参数
    #         opset_version=12,   # ONNX版本
    #         verbose = True, 
    #         do_constant_folding=True,  # 是否执行常量折叠优化
    #         input_names=['input'],   # 输入张量的名称
    #         output_names=['output']) # 输出张量的名称

    # 模型转换为onnx
    torch.onnx.export(model,               # 运行的模型
            image_tensor.cuda(),                  # 模型输入（通常是dummy input）
            args.onnx_path,       # 输出ONNX文件的路径
            export_params=True, # 是否导出模型参数
            opset_version=12,   # ONNX版本
            verbose = True, 
            do_constant_folding=True,  # 是否执行常量折叠优化
            input_names=['input'],   # 输入张量的名称
            output_names=['output'], # 输出张量的名称
            dynamic_axes={'input': {0: 'batch_size'},  # 变量批次大小
                        'output': {0: 'batch_size'}},
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH)
        
    # 加载ONNX模型
    onnx_model = onnx.load(args.onnx_path)
    
    # 验证模型
    ### 自定义onnx算子无法通过验证，所以跳过
    if args.custom_operator:
        print("ignore check_model because the onnx checker not supports custom operator")
    else:
        onnx.checker.check_model(onnx_model)


    graph = onnx_model.graph
    # 计数算子类型
    operator_counter = Counter(node.op_type for node in graph.node)
    op_sum = 0
    for op_type, count in operator_counter.items():
        op_sum+= count
        print(f"{op_type}: {count}")
    print(f"before optimize op_sum: {op_sum}")
    
    if not args.custom_operator:
        # 创建运行时会话并测试模型
        sess = onnxruntime.InferenceSession(args.onnx_path)

        input_feed = {}
        output_fetch = []

        input_name = sess.get_inputs()[0].name
        input_feed[input_name] = image_tensor.cpu().numpy()  # 替换为你的输入数据
        
        # 运行模型并获取输出
        outputs = sess.run(output_fetch, input_feed)
        output = torch.tensor(outputs[0]).cpu()

        print("sum of onnx output:{}".format(torch.sum(output)))
        print("min of onnx output:{}".format(torch.min(output)))
        print("max of onnx output:{}".format(torch.max(output)))

        output_pytorch = model(image_tensor.cuda()).cpu()
        print("sum of pytorch output:{}".format(torch.sum(output_pytorch)))
        print("min of pytorch output:{}".format(torch.min(output_pytorch)))
        print("max of pytorch output:{}".format(torch.max(output_pytorch)))

        mean_difference = torch.mean((output_pytorch-output).abs())
        print("mean_difference of pytorch output and onnx output:{}".format(mean_difference))

    
    if args.simplify_onnx_path != "":
        # # 简化onnx模型,简化后的模型在reshape节点会报错，弃用
        # model_simp, check = simplify(onnx_model)
        # model_simp_inferred = shape_inference.infer_shapes(model_simp)

        optimized_model = onnxoptimizer.optimize(onnx_model)

        # 保存简化后的模型
        onnx.save(optimized_model, args.simplify_onnx_path)
        print("Model simplification done and saved as {}".format(args.simplify_onnx_path))

        # 加载ONNX模型
        onnx_model = onnx.load(args.simplify_onnx_path)

        graph = onnx_model.graph
        # 计数算子类型
        operator_counter = Counter(node.op_type for node in graph.node)
        op_sum = 0
        for op_type, count in operator_counter.items():
            op_sum+= count
            print(f"{op_type}: {count}")
        print(f"op_sum: {op_sum}")
        
        if not args.custom_operator:
            # 验证模型
            onnx.checker.check_model(onnx_model)

            # 创建运行时会话并测试模型
            sess = onnxruntime.InferenceSession(args.simplify_onnx_path)

            input_feed = {}
            output_fetch = []

            input_name = sess.get_inputs()[0].name
            input_feed[input_name] = image_tensor.cpu().numpy()  # 替换为你的输入数据
            
            # 运行模型并获取输出
            outputs = sess.run(output_fetch, input_feed)
            output = torch.tensor(outputs[0]).cpu()

            print("sum of simplify onnx output:{}".format(torch.sum(output)))
            print("min of simplify onnx output:{}".format(torch.min(output)))
            print("max of simplify onnx output:{}".format(torch.max(output)))

if __name__ == '__main__':
    args, config = parse_option()

    # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    #     rank = int(os.environ["RANK"])
    #     world_size = int(os.environ['WORLD_SIZE'])
    #     print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    # else:
    #     rank = -1
    #     world_size = -1
    # torch.cuda.set_device(rank)
    # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # dist.barrier()

    # seed = config.SEED + dist.get_rank()

    seed = 2024
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if True: 
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    main(config, args)
