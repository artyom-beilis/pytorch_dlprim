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

It was tested using MSVC 2022, pytorch 2.4, python 3.12 with ninja build tool. 

### Dependencies

Organize your dependencies directory:


-   Download ninja from https://ninja-build.org/ it would make the life much easier. All instructions here refer to use of Ninja build tool
-   You will nead OpenCL headers and `64` import library. You can get them here: https://github.com/KhronosGroup/OpenCL-SDK/releases
-   SQLite3 is strongly recommended. I recommend to a simple static build and use it:

    Download sqlite-amalgamation-XXXXX.zip file from https://www.sqlite.org/download.html, open "x64 native tool command prompt" shell and 
    complile the library:

        cl /c /EHsc sqlite3.c
        lib sqlite3.obj
    
    Now you have sqlite3.lib and sqlite3.h/sqlite3ext.h you need for build

Put all the dependencies in a layout you can use with ease, something like:

    c:\deps
	c:\deps\include\
	c:\deps\include\CL\opencl.hpp
	c:\deps\include\sqlite3.h
	...
	c:\deps\lib\
	c:\deps\lib\OpenCL.lib
	c:\deps\lib\sqlite3.lib

Addtionally find the location of your python installation, for example `c:\Python\Python312`, you'll need to point to its location in CMake to make sure it find
Make sure you put there 64 release versions only.

Setup virtual pip environment with pytorch. Lets assume you put it into `c:\venv\torch`

Open "x64 Native Tools Command Prompt for VS 2022" and activate virtual environment by running `c:\venv\torch\Scripts\activate` 
Change current directory to location of the `pytorch_dlprim` project

And run:

    mkdir build
	cd build
	cmake -DCMAKE_PREFIX_PATH=%VIRTUAL_ENV%\Lib\site-packages\torch\share\cmake\Torch -DCMAKE_BUILD_TYPE=RelWithDebInfo  -DCMAKE_C_COMPILER="cl.exe" -DCMAKE_CXX_COMPILER="cl.exe" -G Ninja -DCMAKE_INCLUDE_PATH="c:\deps\include\include;C:\Python\Python312\include" -DCMAKE_LIBRARY_PATH="c:\deps\lib;C:\Python\Python312\Libs"  -DCMAKE_INSTALL_PREFIX=c:\path\to\install ..
	ninja
    ninja install
	
Please note: `-DCMAKE_LIBRARY_PATH` and `-DCMAKE_INCLUDE_PATH` point to both dependencies directory and python directory! 

Once build is complete go back to previous directory and run mnist example to test

    cd ..
    set PYTHONPATH=build
	python mnist.py --device=ocl:0
	
For your daily use `set PYTHONPATH=c:\path\to\install\python` and include `import pytorch_ocl`
