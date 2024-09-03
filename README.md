# Pytorch OpenCL backend based on dlprimitives

DLPrimitives-OpenCL out of tree backend for pytorch

It is only beginning, but you can train some vision nets using OpenCL devices.

Supported pytorch versions are 1.13 and torch `>=` 2.4

# Validated Networks

Following torchvision networks were validated:

| Network               |   Notes                                   |
|-----------------------|-------------------------------------------|
| `alexnet`             |                                           |
| `resnet18`            |                                           |
| `resnet50`            |                                           |
| `convnext_small`      |                                           |
| `vgg16`               |                                           |
| `squeezenet1_0`       |                                           |
| `googlenet`           |                                           |
| `densenet161`         |                                           |
| `inception_v3`        | fwd only - backward fails on cuda/cpu     |
| `shufflenet_v2_x1_0`  |                                           |
| `mobilenet_v2`        |                                           |
| `mobilenet_v3_large`  |                                           |
| `mobilenet_v3_small`  | fwd only - same failure on bwd on cuda.   |
| `resnext50_32x4d`     |                                           |
| `wide_resnet50_2`     |                                           |
| `mnasnet1_0`          |                                           |
| `efficientnet_b0`     |                                           |
| `efficientnet_b4`     |                                           |
| `regnet_y_400mf`      |                                           |
| `vit_b_16`            |                                           |


Calculations validated agaist CPU reference for both forward and backward popogation.

# Tested Devices

DLPrimitves itself is tested on following devies: 

- AMD rx 6600XT with ROCM drivers, rx560 16cu with AMDGPU-pro drivers
- Nvidia: GTX 960
- Intel:  HD 530, UHD 770, Arc A380


# Installation

The simplest way is to download `whl` package from release page - builds for Windows and Linux are provided.

Create virtual evironment, install _CPU_ version of pytorch >= 2.4, download `whl` file that corresponds
to python version, pytorch version and architecture and install it. For example: python 3.10, torch 2.4 on Linux it is:

    pip install pytorch_ocl-0.1.0+torch2.4-cp310-none-linux_x86_64.whl

Now all you need is to import `pytorch_ocl` and now you can use device 'ocl' instead of cuda:

    torch.randn(10,10,device='ocl:0')

If you don't have prebuilt whl file for your enviromnent, you want use latest updates or modify the code,
please refer to README-build.md
    
## How to Use
    
Keep it mind... it is earky version that misses a lot of functionality and it isn't fully tested yet.
So if something fails. It is either not implemented or it is implemented incorrectly

Note: pytorch backend is based on dlprimitives library that actually implements all the operators and
it is relatively well tested.

If you still want to try: import package `pytorch_ocl`

Keep in mind you may have several OpenCL devices. Refer to `clinfo --list` to list
of the devices and their order. Now instead of calling `something.to('cuda')` you call `something.to('ocl:0')` or 
`something.to('privateuseone:0' for pytorch 1.13)` or another `ocl:1` etc.

## Known Issues

1. Many operators not implemented and there may be fallbacks to CPU. Sometimes it is minor but sometimes it may hamper the performance, some may just fail
2. When you save/restore the model move it to CPU. Currently there is an issue with loading back saved state dictionary if it was saved from ocl device


## `pytorch_ocl` specific API

Some functions specific to `pytorch_ocl`. When using pytorch >= 2.4 they are accessible from `torch.ocl` and `pytorch_ocl`, for 1.13 you must use `pytorch_ocl`

- `torch.ocl.empty_cache()`: Same as `torch.cuda.empty_cache()` remove all cached GPU memory
- `torch.ocl.synchronize(device=None)`: synchronize all operations queue on the device, if device is None - all of them same as `torch.cuda.synchonize`
- `torch.ocl.manual_seed_all(seed)`: reset random number generator state. `torch.manual_seed` - it calls automatically for pytorch >= 2.4. Note for pytorch 1.13 you must call `pytorch_ocl.manual_seed_all`


