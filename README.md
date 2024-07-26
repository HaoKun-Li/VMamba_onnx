## Export to ONNX

* Please try to see the bash in [run.sh](run.sh). 

* This project is still developing and welcome everyone to discuss together. 
* We use [vssm_tiny_0230_ckpt_epoch_262.pth](https://github.com/MzeroMiko/VMamba/releases/download/%2320240316/vssm_tiny_0230_ckpt_epoch_262.pth) as baseline model.

* The chunk24 technology is modified from [mamba-mini](https://github.com/MzeroMiko/mamba-mini). It is an efficient implementation of selective scan in one file, works with both cpu and gpu, with corresponding mathematical derivation. It is probably the code which is the most close to selective_scan_cuda in mamba.

* The chunkCumsum technology is modified according to the requirement from BITMAIN. THis modification makes sure that the input of cumsum operations can be moved to TPU memory in one time. In this way, the cumsum operation can use the LayerGroup optimization strategy to speedup the model. 

## Image Classification on ImageNet with VMamba
| Model | Hardware|  batchsize | Image/Second | Second/Image | Top1 Acc(%) | Onnx Cells | 
| :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Oiginal PyTorch with Trion and Cuda | Nvidia-A100-40G | 1 | 48.8317 | 0.0205 | 82.490 | -- |
| Pure PyTorch | Nvidia-A100-40G | 1 | 0.7058 | 1.4168 | 82.490 | -- |
| Onnx | Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz | 1 | 0.2667 | 3.7491 |	82.490 | 70386 |
| Onnx+chunk24 | Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz | 1 | 1.3399 | 0.7463 | 82.400 | 24707 |
| SimplifyOnnx+chunk24 | Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz | 1 | 1.3250 | 0.7547 | 82.400 | 12853 |
| Onnx+chunk24| Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz | 16 | 2.2994 | 0.4349 | 82.400 | 24707 |
| SimplifyOnnx+chunk24 | Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz | 16 | 2.3013 | 0.4345 | 82.400 | 12853 |
| Onnx+chunk24+chunkCumsum| Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz | 1 | 2.1616 | 0.4626 | 82.400 | 50408 |
| SimplifyOnnx+chunk24+chunkCumsum | Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz | 1 | 2.1677 | 0.4613 | 82.400 | 24388 |
| Onnx+chunk24+chunkCumsum| Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz | 16 | 2.7462 | 0.3641 | 82.400 | 50408 |
| SimplifyOnnx+chunk24+chunkCumsum | Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz | 16 |  2.7084 | 0.3692 | 82.400 | 24388 |


* The majority code and following Readme is clone from previous https://github.com/MzeroMiko/VMamba at May 21, 2024.
* We will support new version of VMamba soon.

<div align="center">
<h1>VMamba </h1>
<h3>VMamba: Visual State Space Model</h3>

[Yue Liu](https://github.com/MzeroMiko)<sup>1</sup>,[Yunjie Tian](https://sunsmarterjie.github.io/)<sup>1</sup>,[Yuzhong Zhao](https://scholar.google.com.hk/citations?user=tStQNm4AAAAJ&hl=zh-CN&oi=ao)<sup>1</sup>, [Hongtian Yu](https://github.com/yuhongtian17)<sup>1</sup>, [Lingxi Xie](https://scholar.google.com.hk/citations?user=EEMm7hwAAAAJ&hl=zh-CN&oi=ao)<sup>2</sup>, [Yaowei Wang](https://scholar.google.com.hk/citations?user=o_DllmIAAAAJ&hl=zh-CN&oi=ao)<sup>3</sup>, [Qixiang Ye](https://scholar.google.com.hk/citations?user=tjEfgsEAAAAJ&hl=zh-CN&oi=ao)<sup>1</sup>, [Yunfan Liu](https://scholar.google.com.hk/citations?user=YPL33G0AAAAJ&hl=zh-CN&oi=ao)<sup>1</sup>

<sup>1</sup>  University of Chinese Academy of Sciences, <sup>2</sup>  HUAWEI Inc.,  <sup>3</sup> PengCheng Lab.

Paper: ([arXiv 2401.10166](https://arxiv.org/abs/2401.10166))

</div>

* [**updates**](#white_check_mark-updates)
* [**abstract**](#abstract)
* [**overview**](#overview--derivations)
* [**main results**](#main-results)
* [**getting started**](#getting-started)
* [**star history**](#star-history)
* [**citation**](#citation)
* [**acknowledgment**](#acknowledgment)

## :white_check_mark: Updates
* **`May. 7th, 2024`**: Update: **Important!** using `torch.backends.cudnn.enabled=True` in downstream tasks is quite slow, disable it in vmamba.py!

* **` April. 10th, 2024`**: Update: we have released [arXiv 2401.10166v2](https://arxiv.org/abs/2401.10166v2), which contains lots of updates we made related to VMambav2!
 
* **` March. 20th, 2024`**: Update: we have released the `configs/logs/checkpoints` for `classification/detection/segmentation` of VMambav2. We'are still working on VMambav3! 

* **` March. 16th, 2024`**: Improvement: we implemented models with channel_first data layout, which GREATLY raises the `throughput` of the model on A100 (On V100, due to the slow implementation of F.conv2d compared to F.linear, it would not speed up.), Try using `norm_layer="ln2d"` (when inferencing or training) rather than `norm_layer="ln"` to unlock this feature with almost no performance cost!

* **` March. 8th, 2024`**: Update + Improvement: we update the performance of `VMamba-T`, `Vmamba-S`, `VMamba-B` with nightly build, checkpoints and logs are coming soon. (Note that these models are trained without `CrossScanTriton` or `forwardtype=v4`, you can modify those configs yourself to raise the speed with almost no cost!)

* **` March. 8th, 2024`**: Improvement: we implemented `CrossScan` and `CrossMerge` in `triton`, which speed the training up again. `CrossScan` and `CrossMerge` implemented in triton is ~2x faster than implemented in pytorch. Meanwhile, use `v4` rather than `v3` or `v2` in forwardtype also raise the speed GREATLY!.

* **` Feb. 26th, 2024`:** Improvement: we now support flexible output of `selective scan`. That means whatever type the input is, the output can always be float32. The feature is useful as when training with float16, the loss often get nan due to the overflow over float16. In the meantime, training with float32 costs more time. Input with float16 and output with float32 can be fast, but in the meantime, the loss is less likely to be NaN.   Try `SelectiveScanOflex` with float16 input and float32 output to enjoy that feature!

* **` Feb. 22th, 2024`:** Pre-Release: we set a pre-release to share nightly-build checkpoints in classificaion. Feel free to enjoy those new features with faster code and higher performance! 

* **` Feb. 18th, 2024`:** Release: all the checkpoints and logs of `VMamba` (`VSSM version 0`) in classification have been released. These checkpoints correspond to the experiments done before date #20240119, if there is any mismatch to the latest code in main, please let me know, and I'll fix that. This is related to issue#1 and issue#37.

* **` Feb. 16th, 2024`:** Fix bug + Improvement: `SS2D.forward_corev1` is deprecated. Fixed some bugs related to issue#30 (in test_selective scan.py, we now compare `ours` with `mamba_ssm` rather than `selective_scan_ref`), issue#32, issue#31. `backward nrow` has been added and tested in selective_scan.

* **` Feb. 4th, 2024`:** Fix bug + Improvement: Do not use `SS2D.forward_corev1` with `float32=False` for training (testing is ok), as it's unstable training in float16 for selective scan. We released `SS2D.forward_corev2`, which is in float32, and is faster than `SS2D.forward_corev1`.

* **` Feb. 1st, 2024`:** Fix bug: we now calculate FLOPs with the algrithm @albertgu [provides](https://github.com/state-spaces/mamba/issues/110), which will be bigger than previous calculation (which is based on the `selective_scan_ref` function, and ignores the hardware-aware algrithm).

* **` Jan. 31st, 2024`:** ~~Add feature: `selective_scan` now supports an extra argument `nrow` in `[1, 2, 4]`. If you find your device is strong and the time consumption keeps as `d_state` rises, try this feature to speed up `nrows` x without any cost ! Note this feature is actually a `bug fix` for [mamba](https://github.com/state-spaces/mamba).~~

* **` Jan. 28th, 2024`:** Add feature: we cloned main into a new branch called `20240128-achieve`, the main branch has experienced a great update now. The code now are much easier to use in your own project, and the training speed is faster! This new version is totally compatible with original one, and you can use previous checkpoints without any modification. But if you want to use exactly the same models as original ones, just change `forward_core = self.forward_corev1` into `forward_core = self.forward_corev0` in `classification/models/vmamba/vmamba.py#SS2D` or you can change into the branch `20240128-archive` instead.

* **` Jan. 23th, 2024`:** Add feature:  we add an alternative for mamba_ssm and causal_conv1d. Typing `pip install .` in `selective_scan` and you can get rid of those two packages. ~~Just turn `self.forward_core = self.forward_corev0` to `self.forward_core = self.forward_corev1` in `classification/models/vmamba/vmamba.py#SS2D.__init__` to enjoy that feature.~~ The training speed is expected to raise from 20min/epoch for tiny in 8x4090GPU to 17min/epoch, GPU memory cost reduces too.

* **` Jan. 22th, 2024`:** We have released VMamba-T/S pre-trained weights. The ema weights should be converted before transferring to downstream tasks to match the module names using [get_ckpt.py](analyze/get_ckpt.py).

* **` Jan. 19th, 2024`:** The source code for classification, object detection, and semantic segmentation are provided. 

## Abstract

Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) stand as the two most popular foundation models for visual representation learning. While
CNNs exhibit remarkable scalability with linear complexity w.r.t. image resolution, ViTs surpass them in fitting capabilities despite contending with quadratic
complexity. A closer inspection reveals that ViTs achieve superior visual modeling performance through the incorporation of global receptive fields and dynamic
weights. This observation motivates us to propose a novel architecture that inherits these components while enhancing computational efficiency. To this end, we draw
inspiration from the recently introduced state space model and propose the Visual State Space Model (VMamba), which achieves linear complexity without sacrificing global receptive fields. To address the encountered direction-sensitive issue, we introduce the Cross-Scan Module (CSM) to traverse the spatial domain and convert any non-causal visual image into order patch sequences. Extensive experimental results substantiate that VMamba not only demonstrates promising capabilities across various visual perception tasks, but also exhibits more pronounced advantages over established benchmarks as the image resolution increases. 

## Overview & Derivations

* [**VMamba**](https://arxiv.org/abs/2401.10166) serves as a general-purpose backbone for computer vision with linear complexity and shows the advantages of global receptive fields and dynamic weights.

<p align="center">
  <img src="assets/acc_flow_comp.png" alt="accuracy" width="80%">
</p>

* **2D-Selective-Scan of VMamba**

<p align="center">
  <img src="assets/ss2d.png" alt="arch" width="60%">
</p>

* **VMamba has global effective receptive field**

<p align="center">
  <img src="assets/erf_comp.png" alt="erf" width="50%">
</p>


## Main Results
:book: 
***Attention: The configs/logs/checkpoints of `Classification on ImageNet-1K`, `Object Detection on COCO`, `Semantic Segmentation on ADE20K` listed below corresponds to VMambav2[`arXiv 2401.10166v2`](https://arxiv.org/abs/2401.10166v2), which is also named `V9` in section `Accelerating VMamba`.***

:book:
***Attention: The configs/logs/checkpoints of `Classification on ImageNet-1K`, `Object Detection on COCO`, `Semantic Segmentation on ADE20K` corresponding to [`arXiv 2401.10166v1`](https://arxiv.org/abs/2401.10166v1) has been moved [`here`](assets/performance_stage0.md).***

<!-- :book: 
***The checkpoints of some of the models listed below will be released in weeks!*** -->

### **Classification on ImageNet-1K with VMambav2**

| name | pretrain | resolution |acc@1 | #params | FLOPs | configs/logs/ckpts | best epoch | use ema | GPU Mem | time/epoch |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |:---: |
| DeiT-S | ImageNet-1K | 224x224 | 79.8 | 22M | 4.6G | -- | -- | -- | -- | -- |
| DeiT-B | ImageNet-1K | 224x224 | 81.8 | 86M | 17.5G | -- | -- | -- | -- | -- |
| DeiT-B | ImageNet-1K | 384x384 | 83.1 | 86M | 55.4G | -- | -- | -- | -- | -- |
| Swin-T | ImageNet-1K | 224x224 | 81.2 | 28M | 4.5G | -- | -- | -- | -- | -- |
| Swin-S | ImageNet-1K | 224x224 | 83.2 | 50M | 8.7G | -- | -- | -- | -- | -- |
| Swin-B | ImageNet-1K | 224x224 | 83.5 | 88M | 15.4G | -- | -- | -- | -- | -- |
| VMamba-T(0230) | ImageNet-1K | 224x224 | 82.5 | 30M | 4.8G | [config](classification/configs/vssm/vmambav2_tiny_224.yaml)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240316/vssm_tiny_0230.txt)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240316/vssm_tiny_0230_ckpt_epoch_262.pth) | 262 | true | 18234M | 8.12min |
| VMamba-S | ImageNet-1K | 224x224 | 83.6 | 50M | 8.7G | [config](classification/configs/vssm/vmambav2_small_224.yaml)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240316/vssm_small_0229.txt)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240316/vssm_small_0229_ckpt_epoch_222.pth) | 222 | true | 27634M | 11.86min |
| VMamba-B | ImageNet-1K | 224x224 | 83.9 | 89M | 15.4G | [config](classification/configs/vssm/vmambav2_base_224.yaml)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240316/vssm_base_0229.txt)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240316/vssm_base_0229_ckpt_epoch_237.pth) | 237 | true | 37122M | 15.08min |

* *Models in this subsection is trained from scratch with random or manual initialization.*

* *We use ema because our model is still under development.*

* *we now calculate FLOPs with the algrithm @albertgu [provides](https://github.com/state-spaces/mamba/issues/110), which will be bigger than previous calculation (which is based on the `selective_scan_ref` function, and ignores the hardware-aware algrithm).*

### **Object Detection on COCO with VMambav2**
  
| Backbone | #params | FLOPs | Detector | box mAP | mask mAP | configs/logs/ckpts | best epoch |
| :---: | :---: | :---: | :---: | :---: | :---: |:---: |:---: |
| Swin-T | 48M | 267G | MaskRCNN@1x | 42.7| 39.3 |-- |-- |
| VMamba-T | 50M | 270G | MaskRCNN@1x | 47.4| 42.7 | [config](detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_tiny.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_tiny.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_tiny_epoch_12.pth) | 12 |
| Swin-S | 69M | 354G | MaskRCNN@1x | 44.8| 40.9 |-- |-- |
| VMamba-S | 70M | 384G | MaskRCNN@1x | 48.7| 43.7 | [config](detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_small.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_small.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_small_epoch_11.pth) | 11 |
| Swin-B | 107M | 496G | MaskRCNN@1x | 46.9| 42.3 |-- |-- |
| VMamba-B* | 108M | 485G | MaskRCNN@1x | 49.2| 43.9 | [config](detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_base.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_base.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_base_epoch_12.pth) | 12 |
| Swin-T | 48M | 267G | MaskRCNN@3x | 46.0| 41.6 |-- |-- |
| VMamba-T | 50M | 270G | MaskRCNN@3x | 48.9| 43.7 | [config](detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_tiny_ms_3x.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_tiny_ms_3x.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_tiny_ms_3x_epoch_36.pth) | 36 |
| Swin-S | 69M | 354G | MaskRCNN@3x | 48.2| 43.2 |-- |-- |
| VMamba-S | 70M | 384G | MaskRCNN@3x | 49.9| 44.2 | [config](detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_small_ms_3x.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_small_ms_3x.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_small_ms_3x_epoch_32.pth) | 32 |

* *Models in this subsection is initialized from the models trained in `classfication`.*

* *The total batch size of VMamba-B in COCO is `8`, which is supposed to be `16` as in other experiments. This is a `mistake`, not feature. We may fix that later.*

* *we now calculate FLOPs with the algrithm @albertgu [provides](https://github.com/state-spaces/mamba/issues/110), which will be bigger than previous calculation (which is based on the `selective_scan_ref` function, and ignores the hardware-aware algrithm).*

### **Semantic Segmentation on ADE20K with VMambav2**

| Backbone | Input|  #params | FLOPs | Segmentor | mIoU(SS) | mIoU(MS) | configs/logs/logs(ms)/ckpts | best iter |
| :---: | :---: | :---: | :---: | :---: | :---: |:---: |:---: |:---: |
| Swin-T | 512x512 | 60M | 945G | UperNet@160k | 44.4| 45.8| -- | -- |
| VMamba-T| 512x512 | 62M | 948G | UperNet@160k | 48.3| 48.6| [config](segmentation/configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240319/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.log)/[log(ms)](https://github.com/MzeroMiko/VMamba/releases/download/%2320240319/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_tta.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240319/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_iter_160000.pth) | 160k |
| Swin-S | 512x512 | 81M | 1039G | UperNet@160k | 47.6| 49.5| -- | -- |
| VMamba-S| 512x512 | 82M | 1028G | UperNet@160k | 50.6| 51.2|[config](segmentation/configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_small.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240319/upernet_vssm_4xb4-160k_ade20k-512x512_small.log)/[log(ms)](https://github.com/MzeroMiko/VMamba/releases/download/%2320240319/upernet_vssm_4xb4-160k_ade20k-512x512_small_tta.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240319/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth) | 144k |
| Swin-B | 512x512 | 121M | 1188G | UperNet@160k | 48.1| 49.7|-- |
| VMamba-B| 512x512 | 122M | 1170G | UperNet@160k | 51.0| 51.6|[config](segmentation/configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_base.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240319/upernet_vssm_4xb4-160k_ade20k-512x512_base.log)/[log(ms)](https://github.com/MzeroMiko/VMamba/releases/download/%2320240319/upernet_vssm_4xb4-160k_ade20k-512x512_base_tta.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240319/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth) | 160k |
<!-- | Swin-S | 640x640 | 81M | 1614G | UperNet@160k | 47.9| 48.8| -- | -- |
| VMamba-S| 640x640 | 82M | 1607G | UperNet@160k | 50.7| 51.2| [config](segmentation/configs/vssm1/upernet_vssm_4xb4-160k_ade20k-640x640_small.py)/[log]/[log(ms)]/[ckpt] | 160k | -->

* *Models in this subsection is initialized from the models trained in `classfication`.*

* *we now calculate FLOPs with the algrithm @albertgu [provides](https://github.com/state-spaces/mamba/issues/110), which will be bigger than previous calculation (which is based on the `selective_scan_ref` function, and ignores the hardware-aware algrithm).*


## Getting Started

### Installation

**Step 1: Clone the VMamba repository:**

To get started, first clone the VMamba repository and navigate to the project directory:

```bash
git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba
```

**Step 2: Environment Setup:**

VMamba recommends setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:
Also, We recommend using the pytorch>=2.0, cuda>=11.8. But lower version of pytorch and CUDA are also supported.

***Create and activate a new conda environment***

```bash
conda create -n vmamba
conda activate vmamba
```

***Install Dependencies***

```bash
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
```
<!-- cd kernels/cross_scan && pip install . -->

***Check Selective Scan (optional)***

* If you want to check the modules compared with `mamba_ssm`, install [`mamba_ssm`](https://github.com/state-spaces/mamba) first!

* If you want to check if the implementation of `selective scan` of ours is the same with `mamba_ssm`, `selective_scan/test_selective_scan.py` is here for you. Change to `MODE = "mamba_ssm_sscore"` in `selective_scan/test_selective_scan.py`, and run `pytest selective_scan/test_selective_scan.py`.

* If you want to check if the implementation of `selective scan` of ours is the same with reference code (`selective_scan_ref`), change to `MODE = "sscore"` in `selective_scan/test_selective_scan.py`, and run `pytest selective_scan/test_selective_scan.py`.

* `MODE = "mamba_ssm"` stands for checking whether the results of `mamba_ssm` is close to `selective_scan_ref`, and `"sstest"` is preserved for development. 

* If you find `mamba_ssm` (`selective_scan_cuda`) or `selective_scan` ( `selctive_scan_cuda_core`) is not close enough to `selective_scan_ref`, and the test failed, do not worry. Check if `mamba_ssm` and `selective_scan` are close enough [instead](https://github.com/state-spaces/mamba/pull/161).

* ***If you are interested in selective scan, you can check [mamba](https://github.com/state-spaces/mamba), [mamba-mini](https://github.com/MzeroMiko/mamba-mini), [mamba.py](https://github.com/alxndrTL/mamba.py) [mamba-minimal](https://github.com/johnma2006/mamba-minimal) for more information.***

***Dependencies for `Detection` and `Segmentation` (optional)***

```bash
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
```

### Model Training and Inference

**Classification**

To train VMamba models for classification on ImageNet, use the following commands for different configurations:

```bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg </path/to/config> --batch-size 128 --data-path </path/of/dataset> --output /tmp
```

If you only want to test the performance (together with params and flops):

```bash
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg </path/to/config> --batch-size 128 --data-path </path/of/dataset> --output /tmp --pretrained </path/of/checkpoint>
```

***please refer to [modelcard](./modelcard.sh) for more details.***

**Detection and Segmentation**

To evaluate with `mmdetection` or `mmsegmentation`:
```bash
bash ./tools/dist_test.sh </path/to/config> </path/to/checkpoint> 1
```
*use `--tta` to get the `mIoU(ms)` in segmentation*

To train with `mmdetection` or `mmsegmentation`:
```bash
bash ./tools/dist_train.sh </path/to/config> 8
```

For more information about detection and segmentation tasks, please refer to the manual of [`mmdetection`](https://mmdetection.readthedocs.io/en/latest/user_guides/train.html) and [`mmsegmentation`](https://mmsegmentation.readthedocs.io/en/latest/user_guides/4_train_test.html). Remember to use the appropriate backbone configurations in the `configs` directory.

### Analysis Tools

VMamba includes tools for visualizing mamba "attention" and effective receptive field, analysing throughput and train-throughput. Use the following commands to perform analysis:

```bash
# Visualize Mamba "Attention"
CUDA_VISIBLE_DEVICES=0 python analyze/attnmap.py

# Analyze the effective receptive field
CUDA_VISIBLE_DEVICES=0 python analyze/erf.py

# Analyze the throughput and train throughput
CUDA_VISIBLE_DEVICES=0 python analyze/tp.py

```

***We also included other analysing tools that we may use in this project. Thanks to all who have contributes to these tools.***


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MzeroMiko/VMamba&type=Date)](https://star-history.com/#MzeroMiko/VMamba&Date)

## Citation

```
@article{liu2024vmamba,
  title={VMamba: Visual State Space Model},
  author={Liu, Yue and Tian, Yunjie and Zhao, Yuzhong and Yu, Hongtian and Xie, Lingxi and Wang, Yaowei and Ye, Qixiang and Liu, Yunfan},
  journal={arXiv preprint arXiv:2401.10166},
  year={2024}
}
```

## Acknowledgment

This project is based on Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), Swin-Transformer ([paper](https://arxiv.org/pdf/2103.14030.pdf), [code](https://github.com/microsoft/Swin-Transformer)), ConvNeXt ([paper](https://arxiv.org/abs/2201.03545), [code](https://github.com/facebookresearch/ConvNeXt)), [OpenMMLab](https://github.com/open-mmlab),
and the `analyze/get_erf.py` is adopted from [replknet](https://github.com/DingXiaoH/RepLKNet-pytorch/tree/main/erf), thanks for their excellent works.

* **We release [Fast-iTPN](https://github.com/sunsmarterjie/iTPN/tree/main/fast_itpn) recently, which reports the best performance on ImageNet-1K at Tiny/Small/Base level models as far as we know. (Tiny-24M-86.5%, Small-40M-87.8%, Base-85M-88.75%)**
