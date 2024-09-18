# 测试Pytorch模型在imagenet数据集上的准确率
# INFO number of params: 30705832
# INFO number of GFLOPs: 4.857742079999999
# Acc@1 82.482 Acc@5 95.996
# use SelectiveScanOflex, CrossScanTriton, CrossMergeTriton
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg "/home/test/code/VMamba-main/classification/configs/vssm/vmambav2_tiny_224.yaml" --batch-size 128 --data-path "/mnt/hdd/Datasets/imagenet/" --output /tmp --pretrained "/home/test/code/VMamba-main/vssm_tiny_0230_ckpt_epoch_262.pth"


# 单元测试
# 完整的onnx模型过于庞大，模型转换过程缓慢且无法使用Netron可视化，难以debug。
# 因此，构造一个小的模型，包含使用pytorch版的selective_scan_ref, CrossScan, CrossMerge，并转换为onnx模型，并使用onnxoptimizer简化。然后，输入相同随机数据，比对pytorch模型和onnx模型的输出误差。
# 调整核心算子的代码后，可正常导出onnx模型，且对比发现输出误差小于1e-5。
CUDA_VISIBLE_DEVICES=1 python convert_sub_model_to_onnx.py --batch_size 4 --simplify_onnx_path submodel_lhk_simplify_8_8.onnx --token_H_W 8

CUDA_VISIBLE_DEVICES=1 python convert_sub_model_to_onnx.py --batch_size 4 --simplify_onnx_path submodel_lhk_simplify_16_16.onnx --token_H_W 16

CUDA_VISIBLE_DEVICES=1 python convert_sub_model_to_onnx.py --batch_size 4 --simplify_onnx_path submodel_lhk_simplify_32_32.onnx --token_H_W 32



# 通过单元测试后，修改vmamba模型代码，然后导出onnx模型。
# 把Pytorch版的vmamba模型转换为onnx模型
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29502 inference_lhk.py --batch-size 1 --convert_to_onnx --onnx_path "model_lhk_0611.onnx" 2>&1 | tee convert_log20240611.txt

# 无einsum，shape截断版
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29502 inference_lhk.py --batch-size 1 --convert_to_onnx --onnx_path "model_lhk_no_einsum_int_shape_0704.onnx" 2>&1 | tee convert_log20240704.txt

# 无einsum，shape截断版，替换inplace操作后，再次导出onnx模型
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29507 inference_lhk.py --batch-size 1 --convert_to_onnx --onnx_path "model_lhk_noEinsum_intShape_noInplace_0709.onnx" 2>&1 | tee convert_log20240709.txt

# chunk并行版，无einsum，shape截断，替换inplace操作，导出onnx模型
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29507 inference_lhk.py --batch-size 1 --convert_to_onnx --onnx_path "model_lhk_chunk_noEinsum_intShape_noInplace_0709.onnx" 2>&1 | tee convert_log20240710.txt

# chunk并行版，无einsum，shape截断，替换inplace操作，避免分母为0，导出onnx模型，使用onnxoptimizer简化vmamba onnx模型
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29507 inference_lhk.py --batch-size 1 --convert_to_onnx --onnx_path "model_lhk_chunk_noEinsum_intShape_noInplace_0712.onnx" --simplify_onnx_path "model_lhk_simpliy_chunk_noEinsum_intShape_noInplace_0712.onnx" --simplify_onnx 2>&1 | tee convert_log20240712.txt

# batchsize为16，chunk并行版，无einsum，shape截断，替换inplace操作，避免分母为0，导出onnx模型，使用onnxoptimizer简化vmamba onnx模型
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29508 inference_lhk.py --batch-size 16 --convert_to_onnx --onnx_path "model_lhk_chunk_noEinsum_intShape_noInplace_batchsize16_0715.onnx" --simplify_onnx_path "model_lhk_simpliy_chunk_noEinsum_intShape_noInplace_batchsize16_0715.onnx" --simplify_onnx 2>&1 | tee convert_log20240715.txt

# batchsize为32，chunk并行版，无einsum，shape截断，替换inplace操作，避免分母为0，导出onnx模型，使用onnxoptimizer简化vmamba onnx模型
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29508 inference_lhk.py --batch-size 32 --convert_to_onnx --onnx_path "model_lhk_chunk_noEinsum_intShape_noInplace_batchsize32_0722.onnx" --simplify_onnx_path "model_lhk_simpliy_chunk_noEinsum_intShape_noInplace_batchsize32_0722.onnx" --simplify_onnx 2>&1 | tee convert_log20240722.txt

# batchsize为1，chunk并行版，无einsum，shape截断，替换inplace操作，避免分母为0，使用分块计算的cunsum算子,导出onnx模型，使用onnxoptimizer简化vmamba onnx模型
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29508 inference_lhk.py --batch-size 1 --convert_to_onnx --onnx_path "model_lhk_chunk_noEinsum_intShape_noInplace_chunkcumsum_batchsize1_0723.onnx" --simplify_onnx_path "model_lhk_simpliy_chunk_noEinsum_intShape_noInplace_chunkcumsum_batchsize1_0723.onnx" --simplify_onnx 

# batchsize为16，chunk并行版，无einsum，shape截断，替换inplace操作，避免分母为0，使用分块计算的cunsum算子,导出onnx模型，使用onnxoptimizer简化vmamba onnx模型
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29508 inference_lhk.py --batch-size 16 --convert_to_onnx --onnx_path "model_lhk_chunk_noEinsum_intShape_noInplace_chunkcumsum_batchsize16_0725.onnx" --simplify_onnx_path "model_lhk_simpliy_chunk_noEinsum_intShape_noInplace_chunkcumsum_batchsize16_0725.onnx" --simplify_onnx 


# batchsize为1，chunk并行版，无einsum，shape截断，替换inplace操作，避免分母为0，使用分块计算的cunsum算子,导出onnx模型，使用onnxoptimizer简化vmamba onnx模型，替换某一conv1d为matmul
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29508 inference_lhk.py --batch-size 1 --convert_to_onnx --onnx_path "model_lhk_chunk_noEinsum_intShape_noInplace_chunkcumsum_batchsize1_replaceConv1d_0827_v2.onnx" --simplify_onnx_path "model_lhk_simpliy_chunk_noEinsum_intShape_noInplace_chunkcumsum_batchsize1_replaceConv1d_0827_v2.onnx" --simplify_onnx


# batchsize为1，chunk并行版，无einsum，shape截断，替换inplace操作，避免分母为0，使用分块计算的cunsum算子,导出onnx模型，使用onnxoptimizer简化vmamba onnx模型，替换某一conv1d为matmul，使用自定义的onnx算子代替SelectiveScan
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29508 inference_lhk.py --batch-size 1 --convert_to_onnx --onnx_path "model_lhk_chunk_noEinsum_intShape_noInplace_chunkcumsum_batchsize1_custom_operator_0827.onnx" --simplify_onnx_path "model_lhk_simpliy_chunk_noEinsum_intShape_noInplace_chunkcumsum_batchsize1_custom_operator_0827.onnx" --simplify_onnx --custom_operator


# batchsize为1，chunk并行版，无einsum，shape截断，替换inplace操作，避免分母为0，使用分块计算的cunsum算子,导出onnx模型，使用onnxoptimizer简化vmamba onnx模型，替换某一conv1d为matmul，使用自定义的onnx算子代替SelectiveScan，修复reshape形状获取失败的问题
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29508 inference_lhk.py --batch-size 1 --convert_to_onnx --onnx_path "model_lhk_chunk_noEinsum_intShape_noInplace_chunkcumsum_batchsize1_custom_operator_0909.onnx" --simplify_onnx_path "model_lhk_simpliy_chunk_noEinsum_intShape_noInplace_chunkcumsum_batchsize1_custom_operator_0909.onnx" --simplify_onnx --custom_operator




#### 使用onnxoptimizer简化vmamba onnx模型
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 inference_lhk.py --onnx_path "model_lhk_0611.onnx" --simplify_onnx_path "model_lhk_simpliy_0611.onnx" --simplify_onnx

# model_lhk_no_einsum_int_shape_0704.onnx可用，但model_lhk_no_einsum_int_shape_0704_simplify.onnx不可用
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 inference_lhk.py --onnx_path "model_lhk_no_einsum_int_shape_0704.onnx" --simplify_onnx_path "model_lhk_no_einsum_int_shape_0704_simplify.onnx" --simplify_onnx 

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 inference_lhk.py --onnx_path "submodel_lhk_simplify_8_8.onnx" --simplify_onnx_path "submodel_lhk_simplify_v2_8_8.onnx" --simplify_onnx

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 inference_lhk.py --onnx_path "submodel_lhk_simplify_8_8.onnx" --simplify_onnx_path "submodel_lhk_simplify_v2_8_8.onnx" --simplify_onnx

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 inference_lhk.py --onnx_path "submodel_lhk_simplify_16_16.onnx" --simplify_onnx_path "submodel_lhk_simplify_v2_16_16.onnx" --simplify_onnx

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 inference_lhk.py --onnx_path "submodel_lhk_simplify_32_32.onnx" --simplify_onnx_path "submodel_lhk_simplify_v2_32_32.onnx" --simplify_onnx


# 测试替换算子后的pytorch模型在imagenet测试集的准确率
# Acc@1 82.488 Acc@5 95.994
# 使用pytorch版的selective_scan_ref, CrossScan, CrossMerge
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 inference_lhk.py --eval


python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 inference_lhk.py --eval --pretrained /home/test/code/VMamba-main/vssm1_tiny_0230s_ckpt_epoch_264.pth


# 测试onnx模型在imagenet数据集上的准确率
# 测试集共50000张数据，跑了20000数据时，Acc@1约为86%，属于正常情况
# 完整跑完需要55小时，中间终端断开了所以没有跑完
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29502 inference_lhk.py --batch-size 1 --onnx_path "model_lhk_0611.onnx" --eval_onnx 2>&1 | tee eval_onnx_log20240611.txt

# Acc@1 82.490 Acc@5 96.006
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29503 inference_lhk.py --batch-size 1 --onnx_path "model_lhk_no_einsum_int_shape_0704.onnx" --eval_onnx 2>&1 | tee eval_onnx_log20240708.txt

# Nan
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29503 inference_lhk.py --batch-size 1 --onnx_path "model_lhk_chunk_noEinsum_intShape_noInplace_0709.onnx" --eval_onnx 2>&1 | tee eval_onnx_log20240710.txt

# Acc@1 82.400 Acc@5 95.908
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29503 inference_lhk.py --batch-size 1 --onnx_path "model_lhk_chunk_noEinsum_intShape_noInplace_0712.onnx" --eval_onnx 2>&1 | tee eval_onnx_log20240715.txt

# Acc@1 82.400 Acc@5 95.908
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29504 inference_lhk.py --batch-size 1 --onnx_path "model_lhk_simpliy_chunk_noEinsum_intShape_noInplace_0712.onnx" --eval_onnx 2>&1 | tee eval_onnx_log20240715_2.txt

# Acc@1 82.400 Acc@5 95.908
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29504 inference_lhk.py --batch-size 1 --onnx_path "model_lhk_simpliy_chunk_noEinsum_intShape_noInplace_0712.onnx" --eval_onnx 2>&1 | tee eval_onnx_log20240715_2.txt

# Acc@1 82.400 Acc@5 95.908
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29504 inference_lhk.py --batch-size 16 --onnx_path "model_lhk_simpliy_chunk_noEinsum_intShape_noInplace_batchsize16_0715.onnx" --eval_onnx 2>&1 | tee eval_onnx_log20240716.txt

# Acc@1 82.400 Acc@5 95.908
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29504 inference_lhk.py --batch-size 16 --onnx_path "model_lhk_simpliy_chunk_noEinsum_intShape_noInplace_chunkcumsum_batchsize16_0725.onnx" --eval_onnx 


########## 测试pytorch模型、onnx模型、简化后的onnx模型的推理速度(FP32)
# 模型	                       主要推理设备	                        批量大小	   Image/Second     Second/Image
# 原PyTorch模型  	         Nvidia-A100-40G	                        1 	        78.4899	       0.0127
# 原PyTorch模型  	         Nvidia-A100-40G	                        16 	        911.2843       0.0011
# 原PyTorch模型  	         Nvidia-A100-40G	                        32 	        976.6180       0.0010
# 原PyTorch模型  	         Nvidia-A100-40G	                        64 	        1016.9207       0.0010
# 替换特殊算子为PyTorch算子	  Nvidia-A100-40G	                         1	         0.7058	        1.4168
# onnx模型	                Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz	1	        0.2667	       3.7491
# onnxoptimizer简化onnx模型	 Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz	1	        0.2678	       3.7343
# Bmodel模型	              比特大陆BM1684X	                         1	           --	          --
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 inference_lhk.py --batch-size 1 --onnx_path "model_lhk_0611.onnx" --simplify_onnx_path "model_lhk_simpliy_0611.onnx" --inference_time_pytorch --inference_time_onnx --inference_time_simply_onnx



########## 测试pytorch模型、onnx模型、简化后的onnx模型的推理速度
# 模型	                       主要推理设备	                             批量大小	   Image/Second     Second/Image
# 纯PyTorch算子+no_einsum	   Nvidia-A100-40G	                            1	         0.6082	        1.6441
# onnx模型+no_einsum+截断shape Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz	1	         1.0589	        0.9444
# onnxoptimizer简化onnx模型	   Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz	1	          error	         error
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29502 inference_lhk.py --onnx_path "model_lhk_no_einsum_int_shape_0704.onnx" --simplify_onnx_path "model_lhk_no_einsum_int_shape_0704_simplify.onnx" --inference_time_pytorch --inference_time_onnx --inference_time_simply_onnx

# Final Throughput of onnx:0.7588933935852942 image/second
# Time for one image of onnx:1.3177081372070307 second/image
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29505 inference_lhk.py --onnx_path "model_lhk_0705.onnx" --inference_time_onnx

# Final Throughput:5.5966128877964 image/second
# Time for one image:0.17867950134277344 second/image
# Final Throughput of onnx:2.13033090514856 image/second
# Time for one image of onnx:0.4694106430053712 second/image
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29505 inference_lhk.py --onnx_path "model_lhk_chunk_noEinsum_intShape_noInplace_0709.onnx" --inference_time_pytorch --inference_time_onnx



########## 测试pytorch模型、onnx模型、简化后的onnx模型的推理速度
# 模型	                       主要推理设备	                        批量大小	   Image/Second     Second/Image
# 原PyTorch模型  	         Nvidia-A100-40G	                        1 	        78.4899	       0.0127
# 原PyTorch模型  	         Nvidia-A100-40G	                        16 	        911.2843       0.0011
# 原PyTorch模型  	         Nvidia-A100-40G	                        32 	        976.6180       0.0010
# 原PyTorch模型  	         Nvidia-A100-40G	                        64 	        1016.9207       0.0010
# chunk版PyTorch模型  	     Nvidia-A100-40G	                         1	         6.6773	       0.1498
# onnx模型	                Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz	1	         1.3399        0.7463
# onnxoptimizer简化onnx模型	 Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz	1	         1.3250         0.7547
# Bmodel模型	              比特大陆BM1684X	                         1	           --	          --
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 inference_lhk.py --batch-size 1 --onnx_path "model_lhk_chunk_noEinsum_intShape_noInplace_0712.onnx" --simplify_onnx_path "model_lhk_simpliy_chunk_noEinsum_intShape_noInplace_0712.onnx" --inference_time_pytorch --inference_time_onnx --inference_time_simply_onnx


########## 测试pytorch模型、onnx模型、简化后的onnx模型的推理速度
# 模型	                       主要推理设备	                        批量大小	   Image/Second     Second/Image
# onnx模型	                Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz  16	       2.2994           0.4349
# onnxoptimizer简化onnx模型	 Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz	16	       2.3013           0.4345
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 inference_lhk.py --batch-size 16 --onnx_path "model_lhk_chunk_noEinsum_intShape_noInplace_batchsize16_0715.onnx" --simplify_onnx_path "model_lhk_simpliy_chunk_noEinsum_intShape_noInplace_batchsize16_0715.onnx" --inference_time_onnx --inference_time_simply_onnx


########## 测试pytorch模型、onnx模型、简化后的onnx模型的推理速度
# 模型	                       主要推理设备	                        批量大小	   Image/Second     Second/Image
# onnx模型	                Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz  32	       2.3286            0.4294
# onnxoptimizer简化onnx模型	 Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz	32	        2.3207           0.4309
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 inference_lhk.py --batch-size 32 --onnx_path "model_lhk_chunk_noEinsum_intShape_noInplace_batchsize32_0722.onnx" --simplify_onnx_path "model_lhk_simpliy_chunk_noEinsum_intShape_noInplace_batchsize32_0722.onnx" --inference_time_onnx --inference_time_simply_onnx


########## 测试pytorch模型、onnx模型、简化后的onnx模型的推理速度
# 模型	                       主要推理设备	                        批量大小	   Image/Second     Second/Image
# PyTorch模型                 Nvidia-A100-40G	                      1             3.8032           0.2629
# onnx模型	                Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz  1	        2.1616	        0.4626
# onnxoptimizer简化onnx模型	 Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz	1	         2.1677         0.4613
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 inference_lhk.py --batch-size 1 --onnx_path "model_lhk_chunk_noEinsum_intShape_noInplace_chunkcumsum_batchsize1_0723.onnx" --simplify_onnx_path "model_lhk_simpliy_chunk_noEinsum_intShape_noInplace_chunkcumsum_batchsize1_0723.onnx" --inference_time_onnx --inference_time_simply_onnx

########## 测试pytorch模型、onnx模型、简化后的onnx模型的推理速度
# 模型	                       主要推理设备	                        批量大小	   Image/Second     Second/Image
# PyTorch模型                 Nvidia-A100-40G	                      16             60.9305       0.01641
# onnx模型	                Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz  16	        2.7462	        0.3641
# onnxoptimizer简化onnx模型	 Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz	16	         2.7084         0.3692
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 inference_lhk.py --batch-size 16 --onnx_path "model_lhk_chunk_noEinsum_intShape_noInplace_chunkcumsum_batchsize16_0725.onnx" --simplify_onnx_path "model_lhk_simpliy_chunk_noEinsum_intShape_noInplace_chunkcumsum_batchsize16_0725.onnx" --inference_time_onnx --inference_time_simply_onnx



# 温馨提示：
# 完整的vmamba onnx模型过于庞大，模型转换过程缓慢且无法使用Netron可视化，难以debug。
# 代码文件convert_sub_model_to_onnx.py，构造了一个小的模型submodel_lhk_simplify.onnx，其包含了pytorch版的核心算子selective_scan_ref, CrossScan, CrossMerge，并使用了onnxoptimizer简化。输入相同随机数据，比对了pytorch模型和onnx模型的输出误差。
# 建议比特大陆方可先尝试转换submodel_lhk_simplify.onnx，调通后再转换vmamaba的onnx。




# test submodule with chunk
CUDA_VISIBLE_DEVICES=1 python convert_sub_model_to_onnx.py --batch_size 4 --simplify_onnx_path submodel_lhk_chunk_simplify_56_56.onnx --token_H_W 56 --chunksize 24


加载真实数据的测试结果：
chunksize 4    float32
difference mean:0.00027034664526581764  max:0.0615386962890625  min:0.0

chunksize 4    float64
difference mean:0.00027034658705815673  max:0.0615386962890625  min:0.0

chunksize 4    float32   参数全部乘以10
difference mean:1.6901441313166288e-06  max:0.00045359134674072266  min:0.0

chunksize 1    float32
difference mean:0.0002703463542275131  max:0.0615386962890625  min:0.0

chunksize 2    float32
difference mean:0.00027034658705815673  max:0.0615386962890625  min:0.0

chunksize 6    float32
difference mean:0.00027034670347347856  max:0.0615386962890625  min:0.0

chunksize 7    float32
difference mean:0.00027034670347347856  max:0.0615386962890625  min:0.0

chunksize 8    float32
difference mean:0.0002710675762500614  max:1.736608862876892  min:0.0

chunksize 9    float32
difference mean:nan  max:nan  min:nan

chunksize 8    float32   scale and rAts add minimal value
difference mean:0.0002710675762500614  max:1.736608862876892  min:0.0

chunksize 9    float32   scale and rAts add minimal value
difference mean:nan  max:nan  min:nan

chunksize 10    float32   scale and rAts add minimal value
difference mean:nan  max:nan  min:nan

chunksize 4    float64   scale and rAts add minimal value
difference mean:0.00027034658705815673  max:0.0615386962890625  min:0.0

chunksize 8    float64   scale and rAts add minimal value
difference mean:0.00027034658705815673  max:0.0615386962890625  min:0.0

chunksize 16    float64   scale and rAts add minimal value
difference mean:0.00034254053025506437  max:66.87391662597656  min:0.0

chunksize 32    float64   scale and rAts add minimal value
difference mean:0.0031069384422153234  max:75.61922454833984  min:0.0

chunksize 16    float32  no_amp
difference mean:3.2664271287785596e-08  max:7.62939453125e-06  min:0.0

chunksize 24    float32  no_amp
difference mean:nan  max:nan  min:nan

chunksize 32    float32  no_amp
difference mean:nan  max:nan  min:nan

chunksize 16    float32   scale and rAts add minimal value  no_amp
difference mean:3.2664363658341244e-08  max:7.62939453125e-06  min:0.0

chunksize 20    float32   scale and rAts add minimal value  no_amp
difference mean:1.0275289241690189e-06  max:2.3622395992279053  min:0.0

chunksize 21    float32   scale and rAts add minimal value  no_amp
difference mean:4.032215272786743e-08  max:0.0007474422454833984  min:0.0

chunksize 22    float32   scale and rAts add minimal value  no_amp
difference mean:1.8854823338187998e-06  max:1.8059234619140625  min:0.0

chunksize 23    float32   scale and rAts add minimal value  no_amp
difference mean:5.019609943701653e-06  max:1.4689669609069824  min:0.0

chunksize 24    float32   scale and rAts add minimal value  no_amp
difference mean:nan  max:nan  min:nan

chunksize 32    float32   scale and rAts add minimal value  no_amp
difference mean:nan  max:nan  min:nan

chunksize 16    float32   scale and rAts add 1e-10  no_amp
difference mean:0.0006421871948987246  max:2.501288414001465  min:-2.5502662658691406
relative_error mean:0.025566674768924713  max:43476.375  min:-1259.57568359375

chunksize 20    float32   scale and rAts add 1e-10  no_amp
difference mean:0.0017073124181479216  max:3.2233946323394775  min:-6.2792158126831055
relative_error mean:0.1589478999376297  max:72512.5546875  min:-201702.3125

chunksize 16    float32   scale and rAts add 1e-20  no_amp
difference mean:3.2804539529252e-08  max:0.0003216266632080078  min:-7.62939453125e-06
relative_error mean:1.7290254845647723e-06  max:0.1666666716337204  min:-0.17449665069580078

chunksize 20    float32   scale and rAts add 1e-20  no_amp
difference mean:5.734113983635325e-06  max:2.3622395992279053  min:-0.05455142259597778
relative_error mean:7.0750206759839784e-06  max:1.0518473386764526  min:-0.1111111119389534

chunksize 24    float32   scale and rAts add 1e-20  no_amp
difference mean:0.00013749866047874093  max:2.5226364135742188  min:-0.9116148352622986
relative_error mean:0.00016734623932279646  max:13.090376853942871  min:-5.308173179626465

chunksize 32    float32   scale and rAts add 1e-20  no_amp
difference mean:0.000552899029571563  max:2.6379313468933105  min:-1.8089827299118042
relative_error mean:0.005558881442993879  max:3533.392822265625  min:-1668.236328125

chunksize 48    float32   scale and rAts add 1e-20  no_amp
difference mean:0.0033360535744577646  max:2.5226364135742188  min:-5.8783063888549805
relative_error mean:0.1893329918384552  max:72505.34375  min:-38994.76171875

chunksize 16    float32   scale and rAts add 1e-30  no_amp
difference mean:3.266580250738116e-08  max:7.62939453125e-06  min:-7.62939453125e-06
relative_error mean:1.7193584653796279e-06  max:0.1666666716337204  min:-0.17449665069580078

chunksize 20    float32   scale and rAts add 1e-30  no_amp
difference mean:1.0275356316924444e-06  max:2.3622395992279053  min:-9.775161743164062e-06
relative_error mean:2.3270729343494168e-06  max:1.020663857460022  min:-0.1111111119389534

chunksize 24    float32   scale and rAts add 1e-30  no_amp
difference mean:1.8601500414661132e-05  max:1.8623613119125366  min:-0.2578394114971161
relative_error mean:2.24835366680054e-05  max:1.380490779876709  min:-0.1666666716337204

chunksize 32    float32   scale and rAts add 1e-30  no_amp
difference mean:0.000357122509740293  max:2.6379313468933105  min:-1.8089827299118042
relative_error mean:0.002776981331408024  max:3533.392822265625  min:-400.3826599121094

chunksize 48    float32   scale and rAts add 1e-30  no_amp
difference mean:0.001848453190177679  max:2.522199869155884  min:-5.265798091888428
relative_error mean:0.10006838291883469  max:72505.34375  min:-8769.0361328125

chunksize 64    float32   scale and rAts add 1e-30  no_amp
difference mean:0.006645903456956148  max:2.9474456310272217  min:-8.870993614196777
relative_error mean:0.34806525707244873  max:72505.34375  min:-38030.421875

chunksize 24    float32   scale and rAts add 1e-35  no_amp
difference mean:1.8601493138703518e-05  max:1.8623613119125366  min:-0.2578394114971161
relative_error mean:2.2490547053166665e-05  max:1.380490779876709  min:-0.1666666716337204

chunksize 32    float32   scale and rAts add 1e-35  no_amp
difference mean:0.000357122509740293  max:2.6379313468933105  min:-1.8089827299118042
relative_error mean:0.002776985289528966  max:3533.392822265625  min:-400.3826599121094

#### 测试推理速度  
# chunksize12
# Final Throughput:2.902301289057099 image/second
# Time for one image:0.34455416595458993 second/image

# chunksize16
# Final Throughput:3.8644601913544645 image/second
# Time for one image:0.25876835327148434 second/image

# chunksize20
# Final Throughput:4.674827765880439 image/second
# Time for one image:0.21391162414550774 second/image

# chunksize24
# Final Throughput:6.400144303253587 image/second
# Time for one image:0.15624647705078126 second/image

# chunksize32
# Final Throughput:8.41997131268905 image/second
# Time for one image:0.11876525024414059 second/image

# chunksize48
# Final Throughput:8.950930132635007 image/second
# Time for one image:0.11172023300170889 second/image

# chunksize64
# Final Throughput:14.750665526970534 image/second
# Time for one image:0.06779355129241943 second/image
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29506 inference_lhk.py --inference_time_pytorch


#### 测准确率 
# chunksize12  Acc@1 82.548 Acc@5 95.992
# chunksize16  Acc@1 82.494 Acc@5 95.962
# chunksize20  Acc@1 82.500 Acc@5 95.954
# chunksize24  Acc@1 82.408 Acc@5 95.906
# chunksize32  Acc@1 82.272 Acc@5 95.866
# chunksize48  Acc@1 81.586 Acc@5 95.528
# chunksize64  Acc@1 80.886 Acc@5 95.198
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29506 inference_lhk.py --eval


# 统计onnx模型节点数量
python calculate_op.py --onnx_path model_lhk_0611.onnx



# 20240826 尝试导出selectivescan过程为一个onnx节点
# 参考 https://www.jb51.net/python/317288aug.htm
CUDA_VISIBLE_DEVICES=1 python convert_sub_model_to_onnx.py --batch_size 4 --simplify_onnx_path submodel_lhk_chunk_simplify_new_op_20240909.onnx --token_H_W 8 --chunksize 24 --custom_operator 


# CrossScan_onnx time: 24.826879501342773 ms
# After CrossScan_onnx, before selective_scan time: 205.76153564453125 ms
# Selective_scan time: 77.233154296875 ms
# CrossMerge time: 0.27750399708747864 ms
# After CrossMerge time: 0.027648000046610832 ms
CUDA_VISIBLE_DEVICES=1 python convert_sub_model_to_onnx.py --batch_size 4 --simplify_onnx_path submodel_lhk_chunk_simplify_new_op2.onnx --token_H_W 8 --chunksize 24 --custom_operator --test_pytorch_speed


# CrossScan_onnx time: 17.26464080810547 ms
# After CrossScan_onnx, before selective_scan time: 233.10643005371094 ms
# Selective_scan time: 214.76864624023438 ms
# CrossMerge time: 0.27750399708747864 ms
# After CrossMerge time: 0.026623999699950218 ms
CUDA_VISIBLE_DEVICES=1 python convert_sub_model_to_onnx.py --batch_size 4 --simplify_onnx_path submodel_lhk_chunk_simplify_new_op2.onnx --token_H_W 8 --chunksize 24 --custom_operator --test_pytorch_speed --half


# exp size [20, 20, 20, 20, 50] with fp32 format:0.34966400265693665 ms
# exp size [20, 20, 20, 20, 50] with fp16 format:0.25040000677108765 ms

# exp size [20, 20, 20, 20, 20] with fp32 format:0.2585279941558838 ms
# exp size [20, 20, 20, 20, 20] with fp16 format:0.10105600208044052 ms

# exp size [20, 20, 20, 20, 5] with fp32 format:0.04867200180888176 ms
# exp size [20, 20, 20, 20, 5] with fp16 format:0.07097599655389786 ms

# exp size [20, 20, 20, 20, 1] with fp32 format:0.051231998950242996 ms
# exp size [20, 20, 20, 20, 1] with fp16 format:0.08451200276613235 ms

# exp size [20, 20, 20, 200, 1] with fp32 format:0.27350398898124695 ms
# exp size [20, 20, 20, 200, 1] with fp16 format:0.08419200032949448 ms

# exp size [24, 1, 4, 192, 1] with fp32 format:0.01692800037562847 ms
# exp size [24, 1, 4, 192, 1] with fp16 format:0.05734400078654289 ms


# 20240918 尝试导出selectivescan过程为一个onnx节点，并导出只有一个selectivescan节点的onnx模型，节点参考函数为selective_scan_ref()
# 当分类模型输入的图像分辨率为224*224时，对于不同的block, [D, L]会有4种不同的取值组合，因此导出了四个onnx，对应四种类型
CUDA_VISIBLE_DEVICES=3 python convert_sub_model_to_onnx.py --custom_operator --only_selectivescan 

# 读取模型真是推理时的算子输入，比对该算子的pytorch和onnx的计算结果
CUDA_VISIBLE_DEVICES=3 python convert_sub_model_to_onnx.py