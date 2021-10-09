import pycuda.driver as cuda
import pycuda.autoinit  # NOQA
import numpy as np
import ctypes as ct
import random
import os

ROWS_PER_CTA = 8

def loadKerns():
    mod = cuda.module_from_file("jacobi.ptx")
    jacobiKern = mod.get_function("JacobiMethod")
    jacobiKern.prepare("iPPPPP")

    return jacobiKern

def test(A, b, N, iters):
    x = np.zeros((len(A[0]), 1))
    x_new = np.zeros_like(x)
    for k in range(iters):
        for i in range(N):
            temp = b[i]
            for j in range(N):
                temp -= A[i, j] * x[j]
                temp /= A[i, i]
                x_new[i] += temp
        x = np.copy(x_new)
    return x

# Solves the equation Ax=b via the Jacobi iterative method.
def np_jacobi(A,b,N):
    
    x = np.zeros((len(A[0]), 1))
                                                                                                                                                                 
    D = np.diagflat(np.diag(A))
    R = A - D

    # Iterate for N times                                                                                                                                                                          
    for i in range(N):
        x = np.linalg.inv(D) @ (b - (R @ x))
    return x

# @Params:
# N: Number of rows (columns); iters: Maximum iterations
def testJacobi(N, iters):

    jacobiKern = loadKerns()

    rng = np.random.default_rng(40)
    A = rng.random((N, N), dtype=np.float32)
    fill_arr = np.sum(np.abs(A), axis=1)
    np.fill_diagonal(A, fill_arr)
    b = rng.random((N, 1), dtype=np.float64)
    # A = np.zeros(shape=(N, N), dtype=np.float32)
    # b = np.zeros(shape=(N, 1), dtype=np.float64)
    # for i in range(N):
    #     b[i] = 2 * N
    #     for j in range(N):
    #         A[i, j] = 1
    #         A[i, i] = N + 1
    x = np.zeros(shape=(N, 1), dtype=np.float64)
    x_new = np.zeros(shape=(N, 1), dtype=np.float64)
    d = np.zeros(1, dtype=np.float64)

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

    for k in range(iters):
        # if k == iters-1:
        #     cuda.memset_d8(d_d, 0, d.nbytes)
        if k % 2 == 0:
            jacobiKern.prepared_call(grid, block, N, A_d, b_d, x_d, x_new_d, d_d, shared_size=8*N)
        else:
            jacobiKern.prepared_call(grid, block, N, A_d, b_d, x_new_d, x_d, d_d, shared_size=8*N)

    if iters % 2 == 0:
        cuda.memcpy_dtoh(x_new, x_new_d)
        print("CUDA result is:")
        print(x_new)

    else:
        cuda.memcpy_dtoh(x, x_d)
        print("CUDA result is:")
        print(x)
    
    # Relative difference between numpy and cuda result
    np_res = np.linalg.solve(A, b)
    print("Diff between numpy and cuda is:")
    if iters % 2 == 0:
        print(np.abs((np_res - x_new) / np_res))
    else:
        print(np.abs((np_res - x) / np_res))

    # This should print out the error
    cuda.memcpy_dtoh(d, d_d)
    print("CUDA error is:")
    print(d)

testJacobi(512, 3000)
