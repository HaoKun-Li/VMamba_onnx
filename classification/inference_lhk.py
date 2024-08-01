# --------------------------------------------------------
# Modified by $@#Anonymous#@$
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

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

import time

from collections import Counter


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小到224x224
    transforms.ToTensor(),          # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

if torch.multiprocessing.get_start_method() != "spawn":
    print(f"||{torch.multiprocessing.get_start_method()}||", end="")
    torch.multiprocessing.set_start_method("spawn", force=True)


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default="/home/test/code/VMamba-main/classification/configs/vssm/vmambav2_tiny_224.yaml", metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int,default=128, help="batch size for single GPU")
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

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')

    parser.add_argument('--memory_limit_rate', type=float, default=-1, help='limitation of gpu memory use')

    parser.add_argument('--convert_to_onnx', action='store_true',
                        help='whether to convert to onnx model')
    
    parser.add_argument('--onnx_path', type=str, default='model_lhk.onnx',
                        help='save as xxx.onnx')
    
    parser.add_argument('--eval_onnx', action='store_true',
                        help='eval onnx on imagenet')

    parser.add_argument('--inference_time_pytorch', action='store_true',
                        help='eval inference time of pytorch vmamba')
                    
    parser.add_argument('--inference_time_onnx', action='store_true',
                        help='eval inference time of onnx vmamba')
    
    parser.add_argument('--inference_time_simply_onnx', action='store_true',
                        help='eval inference time of onnx vmamba')
    
    parser.add_argument('--simplify_onnx', action='store_true',
                        help='whether to simplify_onnx onnx model')
    
    parser.add_argument('--simplify_onnx_path', type=str, default='')
                        


    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config



@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg



@torch.no_grad()
def validate_onnx(config, data_loader, onnx_model_path):
    import onnxruntime
    import onnx
    
    # 加载ONNX模型
    onnx_model = onnx.load(onnx_model_path)
    
    # 验证模型
    onnx.checker.check_model(onnx_model)
    
    # 创建运行时会话并测试模型
    sess = onnxruntime.InferenceSession(onnx_model_path)

    input_feed = {}
    output_fetch = []
        
    criterion = torch.nn.CrossEntropyLoss()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.numpy()
        target = target.cpu()

        # print("shape of input:{}".format(images.shape))
        
        input_name = sess.get_inputs()[0].name
        input_feed[input_name] = images  # 替换为你的输入数据
        
        # 运行模型并获取输出
        outputs = sess.run(output_fetch, input_feed)
        output = torch.tensor(outputs[0]).cpu()

        # print("shape of output:{}".format(output.shape))

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # acc1 = reduce_tensor(acc1)
        # acc5 = reduce_tensor(acc5)
        # loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg




def main(config, args):
    input_names = ["input_1"]
    output_names = ["output_1"]

    if args.eval or args.convert_to_onnx or args.inference_time_pytorch:
        model = build_model(config)
        model.cuda()
        model.eval()

        # test on imagenet test set
        load_pretrained_ema(config, model, logger, None)
        print("finish load model")

    if args.inference_time_pytorch:
        image_tensor = transform(Image.open("/mnt/hdd/Datasets/imagenet/train/n01440764/n01440764_18.JPEG")).unsqueeze(0).cuda()

        print(image_tensor.size())
        image_tensor = image_tensor.repeat(args.batch_size, 1, 1, 1)
        print(image_tensor.size())

        # # 将 image_tensor 转换为 NumPy array
        # image_tensor_npz = image_tensor.cpu().numpy()

        # # 保存为 npz 文件
        # np.savez('input_tensor_data.npz', data=image_tensor_npz)
        # print("save image_tensor as npz")

        print("inference_time_pytorch batch_size:{}".format(args.batch_size))
        repetitions = 100
        total_time = 0

        # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):

        # warmup
        for warm_iter in range(10):
            # print("warmup processing {}".format(warm_iter))
            output = model(image_tensor)

            # assert False

            # # 将 output 转换为 NumPy array
            # output_npz = output.cpu().detach().numpy()

            # # 保存为 npz 文件
            # np.savez('output_tensor_data.npz', data=output_npz)
            # print("save image_tensor as npz")

        for rep in range(repetitions):
            # print("rep processing {}".format(rep))
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            output = model(image_tensor)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            total_time += curr_time

        Throughput = (repetitions*args.batch_size)/total_time
        print('Final Throughput:{} image/second'.format(Throughput))
        print('Time for one image:{} second/image'.format(total_time/(repetitions*args.batch_size)))




    # evaluate on imagenet
    if args.eval:
        dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

    # 模型转换为onnx
    if args.convert_to_onnx:
        image_tensor = transform(Image.open("/mnt/hdd/Datasets/imagenet/train/n01440764/n01440764_18.JPEG")).unsqueeze(0)
        image_tensor = image_tensor.repeat(args.batch_size, 1, 1, 1)
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
                            'output': {0: 'batch_size'}})
    
    if args.eval_onnx:
        dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
        acc1, acc5, loss = validate_onnx(config, data_loader_val, args.onnx_path)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

    if args.simplify_onnx_path != "" and args.simplify_onnx:
        import onnxruntime
        import onnx
        import onnxoptimizer

        # 加载ONNX模型
        onnx_model = onnx.load(args.onnx_path)

        graph = onnx_model.graph
        # 计数算子类型
        operator_counter = Counter(node.op_type for node in graph.node)
        op_sum = 0
        for op_type, count in operator_counter.items():
            op_sum+= count
            print(f"{op_type}: {count}")
        print(f"before optimize op_sum: {op_sum}")
        
        # 进行优化
        optimized_model = onnxoptimizer.optimize(onnx_model)

        graph = optimized_model.graph
        # 计数算子类型
        operator_counter = Counter(node.op_type for node in graph.node)
        op_sum = 0
        for op_type, count in operator_counter.items():
            op_sum+= count
            print(f"{op_type}: {count}")
        print(f"after optimize op_sum: {op_sum}")

        # 保存简化后的模型
        onnx.save(optimized_model, args.simplify_onnx_path)
        print("Model simplification done and saved as {}".format(args.simplify_onnx_path))
    

    if args.inference_time_onnx:
        import onnxruntime
        import onnx
        
        # 加载ONNX模型
        onnx_model = onnx.load(args.onnx_path)
        
        # 验证模型
        onnx.checker.check_model(onnx_model)
        
        # 创建运行时会话并测试模型
        sess = onnxruntime.InferenceSession(args.onnx_path)

        input_feed = {}
        output_fetch = []

        image_tensor = transform(Image.open("/mnt/hdd/Datasets/imagenet/train/n01440764/n01440764_18.JPEG")).unsqueeze(0).cpu()

        image_tensor = image_tensor.repeat(args.batch_size, 1, 1, 1).numpy()

        input_name = sess.get_inputs()[0].name
        input_feed[input_name] = image_tensor  # 替换为你的输入数据

        print("batch_size: {}".format(args.batch_size))
        repetitions = 100
        total_time = 0
        
        # warmup
        for warm_iter in range(10):
            outputs = sess.run(output_fetch, input_feed)
            # output = torch.tensor(outputs[0]).cpu()

        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            outputs = sess.run(output_fetch, input_feed)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            total_time += curr_time

        Throughput = (repetitions*args.batch_size)/total_time
        print('Final Throughput of onnx:{} image/second'.format(Throughput))
        print('Time for one image of onnx:{} second/image'.format(total_time/(repetitions*args.batch_size)))

        # 运行模型并获取输出
        outputs = sess.run(output_fetch, input_feed)
        output = torch.tensor(outputs[0]).cpu()

    
    if args.inference_time_simply_onnx:
        import onnxruntime
        import onnx
        
        # 加载ONNX模型
        onnx_model = onnx.load(args.simplify_onnx_path)
        
        # 验证模型
        onnx.checker.check_model(onnx_model)
        
        # 创建运行时会话并测试模型
        sess = onnxruntime.InferenceSession(args.simplify_onnx_path)

        input_feed = {}
        output_fetch = []

        image_tensor = transform(Image.open("/mnt/hdd/Datasets/imagenet/train/n01440764/n01440764_18.JPEG")).unsqueeze(0).cpu()

        image_tensor = image_tensor.repeat(args.batch_size, 1, 1, 1).numpy()

        input_name = sess.get_inputs()[0].name
        input_feed[input_name] = image_tensor  # 替换为你的输入数据

        print("batch_size: {}".format(args.batch_size))
        repetitions = 100
        total_time = 0
        
        # warmup
        for warm_iter in range(10):
            outputs = sess.run(output_fetch, input_feed)
            # output = torch.tensor(outputs[0]).cpu()

        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            outputs = sess.run(output_fetch, input_feed)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            total_time += curr_time

        Throughput = (repetitions*args.batch_size)/total_time
        print('Final Throughput of simply_onnx:{} image/second'.format(Throughput))
        print('Time for one image of simply_onnx:{} second/image'.format(total_time/(repetitions*args.batch_size)))

        # 运行模型并获取输出
        outputs = sess.run(output_fetch, input_feed)
        output = torch.tensor(outputs[0]).cpu()



#     image_tensor = transform(Image.open("/mnt/hdd/Datasets/imagenet/train/n01440764/n01440764_18.JPEG")).unsqueeze(0)
#     # 简单推理
# # 方式1
#     images = image_tensor.cuda(non_blocking=True)

#     print("finish load images")

#     # compute output
#     with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
#         output = model(images)

#         print("finish inference type 1")

# # 方式2
#     res = model(image_tensor.cuda())
#     print("finish inference type 1")

#     pdb.set_trace()
#     print(output)
#     print(sum(output[0]))
#     print(res)
#     print(sum(res[0]))


# 模型转换从onnx

    # torch.onnx.export(model,               # 运行的模型
    #               image_tensor.cuda(),                  # 模型输入（通常是dummy input）
    #               "model.onnx",       # 输出ONNX文件的路径
    #               export_params=True, # 是否导出模型参数
    #               opset_version=10,   # ONNX版本
    #               do_constant_folding=True,  # 是否执行常量折叠优化
    #               input_names=['input'],   # 输入张量的名称
    #               output_names=['output'], # 输出张量的名称
    #               dynamic_axes={'input': {0: 'batch_size'},  # 变量批次大小
    #                             'output': {0: 'batch_size'}})




if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if True: 
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    # to make sure all the config.OUTPUT are the same
    config.defrost()
    if dist.get_rank() == 0:
        obj = [config.OUTPUT]
        # obj = [str(random.randint(0, 100))] # for test
    else:
        obj = [None]
    dist.broadcast_object_list(obj)
    dist.barrier()
    config.OUTPUT = obj[0]
    print(config.OUTPUT, flush=True)
    config.freeze()
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    if args.memory_limit_rate > 0 and args.memory_limit_rate < 1:
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_rate)
        usable_memory = torch.cuda.get_device_properties(0).total_memory * args.memory_limit_rate / 1e6
        print(f"===========> GPU memory is limited to {usable_memory}MB", flush=True)

    main(config, args)
