import sys
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule, DynamicModule
import numpy as np
import ctypes
import ctypes.util
import pathlib


def test(func):
    np.random.seed(0)
    a = np.random.randn(4,4)
    a = a.astype(np.float32)
    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)

    # func.call(a_gpu, block=(4,4,1))
    grid = (1, 1)
    block = (4, 4, 1)

    func.prepared_call(grid, block, a_gpu, 5)

    a_scaled = np.empty_like(a)
    cuda.memcpy_dtoh(a_scaled, a_gpu)
    print(a_scaled)
    print(a)


def getCubin():
    mod = cuda.module_from_file("kerns/kerns.cubin")
    func = mod.get_function("doublifyKern")
    func.prepare("P")
    return func


def getProdJit():
    mod = SourceModule("""
        #include <stdint.h>
        __global__ void prodKern(uint32_t *v0, uint32_t *v1, uint32_t *vout, uint64_t *len)
        {
            int id = blockIdx.x*blockDim.x+threadIdx.x;
            if (id < *len) {
                vout[id] = v0[id] * v1[id];    
            }
        }
        """, options=["-std=c++14"])

    func = mod.get_function("prodKern")
    func.prepare(["P"]*4)
    return func

def getJitted():
    mod = SourceModule("""
      #include <stdint.h>
      __global__ void multiply(float *a, float v)
      {
	int idx = threadIdx.x + threadIdx.y*4;
	a[idx] *= v;
      }
      """)
    func = mod.get_function("multiply")
    func.prepare(['P', 'f'])
    return func

def testProd():
    f = getProdJit()

    np.random.seed(0)
    a = np.random.randint(0, 256, size=8, dtype=np.uint32)
    b = np.random.randint(0, 256, size=8, dtype=np.uint32)

    a_d = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_d, a)
    b_d = cuda.mem_alloc(b.nbytes)
    cuda.memcpy_htod(b_d, b)
    c_d = cuda.mem_alloc(b.nbytes)
    len_d = cuda.mem_alloc(8)
    cuda.memcpy_htod(len_d, ctypes.c_uint64(8))

    grid = (1, 1)
    block = (8,1,1)

    bufs = [a_d, b_d, c_d, len_d]

    f.prepared_call(grid, block, *bufs, shared_size=0)

    c_prod = np.empty_like(a)
    cuda.memcpy_dtoh(c_prod, c_d)
    print(a)
    print(b)
    print(a*b)
    print(c_prod)


# testProd()
# cuF = getCubin()
jitF = getJitted()

# test(cuF)
test(jitF)
