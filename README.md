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


Calculations validated agaist CPU reference for both forward and backward popogation.

# Tested Devices

DLPrimitves itself is tested on following devies: 

- Nvidia: gtx 960
- AMD: rx 6600 xt and in past rx 560
- Intel: HD530


# Build

## Changes From previous

## In the nutshell

- Setup pip virtual enviromnet with _CPU_ version of pytorch. Supported pytorch version are 2.4 and above. Also pytorch 1.13 is supprted.
- Build dlprim\_backend and install at location you want
- import `pytorch_ocl` and use `ocl` device instead of `cuda`

## Now in details

1.  Setup pip virtual environment and install CPU version of pytorch - 2.4 is recommended. Pytorch 1.13 is still supported.

    Install CPU variant since you don't need CUDA support for OpenCL backend to work.

2.  Make sure you have OpenCL headers and library. It should include `opencl.hpp` or`cl2.hpp` - not the old one `cl.hpp`

3.  It is strongly recommended to have SQLite3 library and headers avalible as well, it would improve startup times by caching OpenCL kernels on disk.

4. Clone The repository

        git clone --recurse-submodules https://github.com/artyom-beilis/pytorch_dlprim.git

5.  Build the backend.

## Building the on Linux

Make sure you are in the virtual environment

	mkdir build
	cd build
	cmake -DCMAKE_PREFIX_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/torch/share/cmake/Torch -DCMAKE_INSTALL_PREFIX=/path/to/install/location ..
	make
    make install

Note: if you use python version that is different from 3.10 just fix the path above

Test it runs:

    export PYTHONPATH=/path/to/install/location/python
	python mnist.py --device ocl:0

If you want to test it in build environment use `export PYTHONPATH=build`

Note: for pytorch 1.13 use privateuseone device instead of ocl

## Building on Windows

Note: Windows support is even more experimental than Linux support. It was tested using MSVC 2022 using ninja build tool. 

You will nead OpenCL headers and `x86_64` import library. It is also strongly recommended to get sqlite3
library. You can download 64 bit def and dll files and headers from official web site. You can convert def file
to import library by running `lib /def:sqlite3.def /out:sqlite3.lib /machine:x64`

Put all the dependencies in a layout you can use with ease, something like:

    c:\deps
	c:\deps\include\
	c:\deps\include\CL\cl2.hpp
	c:\deps\include\sqlite3.h
	...
	c:\deps\lib\
	c:\deps\lib\OpenCL.lib
	c:\deps\lib\sqlite3.lib
	c:\deps\lib\sqlite3.dll

Make sure you put there 64 release versions only.

Setup virtual pip environment with pytorch. Lets assume you put it into `c:\venv\torch`

Open "x64 Native Tools Command Prompt for VS 2022" and activate virtual environment by running `c:\venv\torch\Scripts\activate` 
Change current directory to location of the `pytorch_dlprim` project

And run:

    mkdir build
	cd build
	cmake -DCMAKE_PREFIX_PATH=%VIRTUAL_ENV%\Lib\site-packages\torch\share\cmake\Torch -DCMAKE_BUILD_TYPE=RelWithDebInfo   -DCMAKE_C_COMPILER="cl.exe" -DCMAKE_CXX_COMPILER="cl.exe" -G Ninja -DCMAKE_INCLUDE_PATH=c:\deps\include\include -DCMAKE_LIBRARY_PATH=c:\deps\lib  -DCMAKE_INSTALL_PREFIX=c:\path\to\install ..
	ninja
	
Make sure that sqlite3 dll is in the path by calling

    set PATH=%PATH%;c:\deps\lib	
	
Once build is complete go back to previous directory and run mnist example

    cd ..
	python mnist.py --device=ocl:0
	
    
## How to Use
    
Keep it mind... it is very very initial version that misses a lot of functionality and it isn't fully tested yet.
So if something fails. It is either not implemented or it is implemented incorrectly

Note: pytorch backend is based on dlprimitives library that actually implements all the operators and
it is relatively well tested.

If you still want to try:

Import package pytorch_ocl

    torch.ops.load_library("/path/to/libpt_ocl.so")
	
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


