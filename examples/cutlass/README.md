# Building cutlass
Build with:

    cd cutlass-src
    mkdir build
    cd build
    cmake .. -DCUTLASS_NVCC_ARCHS=35 -DCUTLASS_ENABLE_CUBLAS=OFF -DCUTLASS_ENABLE_CUDNN=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_ENABLE_PROFILER=OFF
    make -j8

# Building the example
Our goal is to generate a usable cubin library with the kernels we want along with a host library for creating the appropriate arguments to the kernels from more generic (and KaaS compatible) user inputs.

    ./build.sh

This should generate out.cubin which contains the compiled kernel from test.cu.
