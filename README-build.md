## In a nutshell

- Setup a python environment (using `uv` is recommended).
- Build the project using `uv build` or `pip install .`.
- The build system uses **CPM (CMake Package Manager)** to automatically fetch dependencies like `dlprimitives` and OpenCL headers.
- import `pytorch_ocl` and use `ocl` device instead of `cuda`.

## Dependencies

The following dependencies are automatically handled by the build system via CPM:
- **dlprimitives**: Core deep learning primitives for OpenCL.
- **OpenCL-Headers**: Official Khronos OpenCL headers.
- **OpenCL-CLHPP**: Official Khronos OpenCL C++ headers.

You still need to have:
- **OpenCL Drivers**: Ensure your GPU/CPU has OpenCL drivers installed (e.g., `intel-opencl-icd`, `rocm-opencl-runtime`, or NVIDIA drivers).
- **SQLite3**: (Recommended) For kernel caching.

## Build Optimization

- **CPM Cache**: To avoid downloading dependencies for every new build directory, it is recommended to set the `CPM_SOURCE_CACHE` environment variable:
  ```bash
  export CPM_SOURCE_CACHE=$HOME/.cache/CPM
  ```

## Building on Linux / macOS / Windows

The recommended way to build and install is using `uv`:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/artyom-beilis/pytorch_dlprim.git
    cd pytorch_dlprim
    ```

2.  **Build the wheel**:
    ```bash
    uv build
    ```

3.  **Install the wheel**:
    ```bash
    uv pip install dist/pytorch_ocl-0.1.0-*.whl
    ```

4.  **Verify the installation**:
    ```bash
    python mnist.py --device ocl:0
    ```

### Manual CMake Build (Advanced)

If you prefer building manually with CMake:

```bash
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/your/torch/cmake -DCMAKE_INSTALL_PREFIX=/path/to/install
make -j$(nproc)
make install
```

## Troubleshooting

### OpenCL not found
If CMake fails to find OpenCL, ensure the OpenCL library is in your system's library path. On Linux, this is typically `/usr/lib/libOpenCL.so`.

### Python/PyTorch version
Ensure you are using a supported PyTorch version (2.4+ recommended, 1.13 also supported).
