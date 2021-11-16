# Pytorch OpenCL backend based on dlprimitives

DLPrimitives-OpenCL out of tree backend for pytorch

It is only beginning, but you can train some vision nets using OpenCL devices.


# Validated Networks

Following networks were validated in computations of 

| Network               |   Notes                                   |
|-----------------------|-------------------------------------------|
| `alexnet`             |                                           |
| `resnet18`            |                                           |
| `resnet50`            |                                           |
| `vgg16`               |                                           |
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


Results of inference validated agaist CPU reference for both forward and backward popogation.

Out of pretrained networks following two are failing: `squeezenet1_0`, `googlenet` since
ceil pooling mode isn't implemented yet

# Tested Devices

DLPrimitves itself is tested on following devies: 

- Nvidia: gtx 1080, rtx 2060s, gtx 960
- AMD: rx 6600 xt and in past rx 560
- Intel: HD530

# Benchmarks

All benchmarks done on gtx 960/4G to get comparison to native cuda speed.

## Test

Test includes copy of data to/from device and forward calculations

| Framework       | alexnet  | resnet18 | resnet50 | vgg16  |  mobilenet |
|-----------------|----------|----------|----------|--------|------------|
|pytorch/cuda     | 15.253   | 38.745   | 114.348  | 169.038| 46.110     |     
|pytorch/opencl   | 22.989   | 50.272   | 167.050  | 258.751| 82.044     |     
|dlprimitives     | 22.688   | 49.193   | 158.789  | 238.802| 82.080     |     
|keras/tf2-cuda   | 29.104   | 74.215   | 161.704  | 158.084| 88.851     |     
|keras/plaidml    | 43.004   | 91.533   | -        | -      | 45.693     |     

## Full Train 

Train includes - io to/from device, zero gadients, forward, backward and optimizer update step. Adam used as optimizer.


| Framework       | alexnet  | resnet18 | resnet50 | vgg16  |  mobilenet |
|-----------------|----------|----------|----------|--------|------------|
|pytorch/cuda     | 107.108  | 129.456  | 388.951  | N/A    | 177.434    |     
|pytorch/opencl   | 147.814  | 213.319  | 651.216  | N/A    | 382.590    |     
|dlprimitives     | 106.033  | 198.092  | 605.541  |1107.756| 344.599    |     
|keras/tf2-cuda   |  90.005  | 183.447  | 501.362  | 550.063| 322.416    |     
|keras/plaidml    | 222.166  | 507.116  | -        | -      | 571.438    |     

- vgg16 batch 16 failed to run to to lack of memory on pytorch.
- some setups with plaidml not tested due to lack of performance/memory



# Build

## In the nutshell

- Build customised version of pytorch (tiny change from main version)
- Build dlprimitives
- Build dlprim\_backend
- Load shared library in pytorch and start using it.

## Now in details

1.  The most complex part is to build pytorch - make sure you can build and install your own version of pytorch.

    Now take this git repo: <https://github.com/artyom-beilis/pytorch> it differs from the Original pytorch with a
    single modification of mapping OpenCL devices to PrivateUse dispatch key... If it is Greek to you, ignore, just
    build pytorch from this repo as an official one. Of course you can disable cuda by setting environment 
    variable `USE_CUDA=0`

    After you build pytorch and installed it into a virtual environment.

    **Don't try to skip this step. It wouldn't work.**

2.  Build dlprimitives <https://github.com/artyom-beilis/dlprimitives> and install it, lets say to `/opt/dlprim`

    Follow the instructions there: <https://github.com/artyom-beilis/dlprimitives/blob/master/docs/build.md>

3.  Build the backend.

        mkdir build
        cd build
        cmake -DCMAKE_PREFIX_PATH=$VIRTUAL_ENV/lib/python3.6/site-packages/torch/share/cmake/Torch ..
        make

    If cmake does not find dlprimitives provide `-DCMAKE_INCLUDE_PATH=path/to/dlprim/include` and `-DCMAKE_LIBRARY_PATH=path/to/dlprim/lib`
    to make sure it finds `libdlprim_core.so` and its header files

    Test it runs:

        python mnist.py --device opencl:0

    
## How to Use
    
Keep it mind... it is very very initial version that misses a lot of functionality and it isn't fully tested yet.
So if something fails. It is either not implemented or it is implemented incorrectly

Note: pytorch backend is based on dlprimitives library that actually implements all the operators and
it is relatievely well tested.


If you still want to try:

-   Before you begin in python code, load the library `libpt_ocl.so`:

        torch.ops.load_library("/path/to/libpt_ocl.so")

    It would enable you to use opencl devices. Keep in mind you may have several. Refer to `clinfo --list` to list
    of the devices and their order. Now instead of calling `something.to('cuda')` you call `something.to('opencl:0')`
    or another `opencl:1` etc.

-   Try to do only essential tasks on GPU, handle preparations and outputs on CPU since many ops may not be implemented
    for example printing




