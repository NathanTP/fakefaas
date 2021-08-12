#!/bin/bash

if [ -z "$CUTLASS_PATH" ]; then
    CUTLASS_PATH=./cutlass
fi

CUTLASS_PATH=$(realpath $CUTLASS_PATH)

# Most of these arguments were taken from running "make VERBOSE=1" in cutlass/build/examples/
/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler \
    -I$CUTLASS_PATH/include \
    -I$CUTLASS_PATH/g/examples/common \
    -I$CUTLASS_PATH/g/build/include \
    -I/usr/local/cuda/include \
    -I$CUTLASS_PATH/g/tools/util/include \
    -O0 -DNDEBUG -Xcompiler=-fPIC \
    --cubin \
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
    -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing \
    -gencode=arch=compute_35,code=sm_35 -std=c++11 \
    -x cu -c kern.cu -o kern.cubin

# Here's a starting point for a command to run a host program linked against
# out.cubin. It won't work out of the box, you'll need to modify it to link
# against out.cubin and stuff.
# /usr/bin/g++ basic_gemm.cu.o -o 00_basic_gemm  \
#     -lcudadevrt -lcudart_static -lrt -lpthread -ldl \
#     -L"/usr/local/cuda/targets/x86_64-linux/lib/stubs" \
#     -L"/usr/local/cuda/targets/x86_64-linux/lib"

/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler \
-I/scratch/jasonding/fakefaas/examples/cutlass/cutlass/include \
-I/scratch/jasonding/fakefaas/examples/cutlass/cutlass/examples/00_basic_gemm \
-I/scratch/jasonding/fakefaas/examples/cutlass/cutlass/g/examples/common \
-I/scratch/jasonding/fakefaas/examples/cutlass/cutlass/g/build/include \
-I/usr/local/cuda/include \
-I/scratch/jasonding/fakefaas/examples/cutlass/cutlass/g/tools/util/include \
-O0 -DNDEBUG -Xcompiler=-fPIC \
-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
-Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing \
-gencode=arch=compute_35,code=sm_35 -std=c++11 \
-x cu -o lib/out.so --shared cutlassAdaptors.cu

/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler \
-I/scratch/jasonding/fakefaas/examples/cutlass/cutlass/include \
-I/scratch/jasonding/fakefaas/examples/cutlass/cutlass/examples/00_basic_gemm \
-I/scratch/jasonding/fakefaas/examples/cutlass/cutlass/g/examples/common \
-I/scratch/jasonding/fakefaas/examples/cutlass/cutlass/g/build/include \
-I/usr/local/cuda/include \
-I/scratch/jasonding/fakefaas/examples/cutlass/cutlass/g/tools/util/include \
-O0 -DNDEBUG -Xcompiler=-fPIC \
--cubin \
-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
-Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing \
-gencode=arch=compute_35,code=sm_35 -std=c++11 \
-x cu -c kern.cu -o kern.cubin

/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler \
-I/scratch/jasonding/fakefaas/examples/cutlass/cutlass/include \
-I/scratch/jasonding/fakefaas/examples/cutlass/cutlass/examples/00_basic_gemm \
-I/scratch/jasonding/fakefaas/examples/cutlass/cutlass/g/examples/common \
-I/scratch/jasonding/fakefaas/examples/cutlass/cutlass/g/build/include \
-I/usr/local/cuda/include \
-I/scratch/jasonding/fakefaas/examples/cutlass/cutlass/g/tools/util/include \
--cubin  -rdc=true -lcudadevrt \
-gencode=arch=compute_35,code=sm_35 -std=c++11 \
-x cu -c lib/kern.cu -o kern.cubin