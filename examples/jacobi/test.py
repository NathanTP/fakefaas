import pycuda.driver as cuda
import pycuda.autoinit  # NOQA
import numpy as np
import ctypes as ct
import random
import os

ROWS_PER_CTA = 8

def loadKerns():
    mod = cuda.module_from_file("jacobi.ptx")
    jacobiKern = mod.get_function("JacobiMethodAllIter")
    jacobiKern.prepare("PPiPPPi")

    return jacobiKern


class kernelConfig(ct.Structure):
    """This mirrors the CudaConfig struct defined in cutlassAdapters.h"""
    _fields_ = [
        ("gridX", ct.c_int),
        ("gridY", ct.c_int),
        ("gridZ", ct.c_int),
        ("blockX", ct.c_int),
        ("blockY", ct.c_int),
        ("blockZ", ct.c_int),
        ("smem_size", ct.c_int)
    ]

# @Params:
# N: Number of rows (columns); iters: Maximum iterations
def testJacobi(N, iters):

    jacobiKern = loadKerns()

    rng = np.random.default_rng(50)
    A = rng.random((N, N), dtype=np.float32)
    fill_arr = np.sum(np.abs(A), axis=1) + 1
    np.fill_diagonal(A, fill_arr)
    b = rng.random((N, 1), dtype=np.float64)
    x = np.zeros(shape=(N, 1), dtype=np.float64)
    x_new = np.zeros(shape=(N, 1), dtype=np.float64)
    d_sum = np.zeros(1, dtype=np.float64)

    A_d = cuda.mem_alloc(A.nbytes)
    cuda.memcpy_htod(A_d, A)

    b_d = cuda.mem_alloc(b.nbytes)
    cuda.memcpy_htod(b_d, b)

    x_d = cuda.mem_alloc(x.nbytes)
    cuda.memset_d8(x_d, 0, x.nbytes)

    x_new_d = cuda.mem_alloc(x_new.nbytes)
    cuda.memset_d8(x_new_d, 0, x_new.nbytes)

    d_d = cuda.mem_alloc(d.nbytes)
    cuda.memset_d8(d_d, 0, d.nbytes)

    grid = (256, 1, 1)
    block = ((N // ROWS_PER_CTA) + 2, 1, 1)

    print("Grid is: ", grid)
    print("Block is: ", block)

    jacobiKern.prepared_call(grid, block, A_d, b_d, N, x_d, x_new_d, d_d, iters)

    if iters % 2 == 0:
        # x_new
        cuda.memcpy_dtoh(x_new, x_new_d)
        print(x_new)
    else:
        cuda.memcpy_dtoh(x, x_d)
        print(x)

testJacobi(512, 1000)
# mod = cuda.module_from_file("jacobi.ptx")
# kern = loadKerns()
