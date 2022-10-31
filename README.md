# Pytorch OpenCL backend based on dlprimitives

DLPrimitives-OpenCL out of tree backend for pytorch

It is only beginning, but you can train some vision nets using OpenCL devices.


# Validated Networks

Following torchvision networks were validated:

| Network               |   Notes                                   |
|-----------------------|-------------------------------------------|
| `alexnet`             |                                           |
| `resnet18`            |                                           |
| `resnet50`            |                                           |
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

## Changes From previous

Note the build procedure was significantly simplified - so READ again

1. You don't need to build custom pytorch
2. You should use ocl name for device rather than opencl (see details below)

## In the nutshell

- Setup pip virtual enviromnet with pytorch 1.13 or nighyly version for CPU
- Build dlprim\_backend
- Load shared library in pytorch and start using it.

## Now in details

1.  Setup pip virtual environment and install CPU version of pytorch - either 1.13 stable or
    nightly build of pytorch:: <https://pytorch.org/get-started/locally/>

    Install CPU variant since you don't need CUDA support for OpenCL backend to work.

2.  Make sure you have OpenCL headers and library. It should be `cl2.hpp` - not the old one `cl.hpp`

3.  It is strongly recommended to have SQLite3 library and headers avalible as well, it would improve startup times by caching OpenCL kernels on disk.

4. Clone The repository

        git clone --recurse-submodules https://github.com/artyom-beilis/pytorch_dlprim.git

5.  Build the backend.

## Building the on Linux

Make sure you are in the virtual environment

	mkdir build
	cd build
	cmake -DCMAKE_PREFIX_PATH=$VIRTUAL_ENV/lib/python3.8/site-packages/torch/share/cmake/Torch ..
	make

Note: if you use python version that is different from 3.8 just fix the path above

Test it runs:

	python mnist.py --device ocl:0

Note from previous build procedure, now dlprimitives is submodule of the project. No need to build it separatly.

## Building on Windows

Note: Windows support is even more experimental than Linux support. It was tested using pytorch 1.13, MSVC 2022 using ninja build tool. 

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

Setup virtual pip enviromnet with pytorch. Lets assume you put it into `c:\venv\torch`

Open "x64 Native Tools Command Prompt for VS 2022" and activate virtual envornment by running `c:\venv\torch\Scripts\activate` 
Change current directory to location of the `pytorch_dlprim` project

And run:

    mkdir build
	cd build
	cmake -DCMAKE_PREFIX_PATH=%VIRTUAL_ENV%\Lib\site-packages\torch\share\cmake\Torch -DCMAKE_BUILD_TYPE=RelWithDebInfo   -DCMAKE_C_COMPILER="cl.exe" -DCMAKE_CXX_COMPILER="cl.exe" -G Ninja -DCMAKE_INCLUDE_PATH=c:\deps\include\include -DCMAKE_LIBRARY_PATH=c:\deps\lib  ..
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
it is relatievely well tested.

If you still want to try:

-   Before you begin in python code, load the library `libpt_ocl.so` :

        torch.ops.load_library("/path/to/libpt_ocl.so")
	
	Or on Windows
	
	    torch.ops.load_library("/path/to/pt_ocl.dll")
		
	It enables useing opencl devices as `privateuseone` device.
		
	If you use nighly version `>= 1.14` you can rename `privateuseone` device to `ocl`
	
        torch.utils.rename_privateuse1_backend('ocl')

    Keep in mind you may have several. Refer to `clinfo --list` to list
    of the devices and their order. Now instead of calling `something.to('cuda')` you call `something.to('ocl:0')` or 
	`something.to('privateuseone:0')` or another `ocl:1` etc.

-   Try to do only essential tasks on GPU, handle preparations and outputs on CPU since many ops may not be implemented
    for example printing

