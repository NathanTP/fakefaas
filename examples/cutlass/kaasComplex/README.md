# Quickstart

    make
    ./basic_gemm
    python test.py

# Files
## basic\_gemm.cu
This is a pure C example of some of the components. Useful for testing stuff
out before messing with python.

## cutlassAdapters.cu/h
These are utility functions on the host for working with cutlass.

## kern.cu/h
These are the various sgemm kernels that we use. This includes the real cutlass
one, a reference kernel, and a super simple kernel for trying stuff out.

## test.py
This is example python code for invoking the various cutlass kernels and utilities.
