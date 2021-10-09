import pycuda.driver as cuda
import pycuda.autoinit  # NOQA
import numpy as np
import ctypes as ct
import random
import os

ROWS_PER_CTA = 8

def loadKerns():
    mod = cuda.module_from_file("jacobi.ptx")
    jacobiKern = mod.get_function("JacobiMethodOuter")
    jacobiKern.prepare("PPPPPi")

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

def np_jacobi(A,b,N,x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed                                                                                                                                                            
    if x is None:
        x = np.zeros((len(A[0]), 1))

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
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

    # rng = np.random.default_rng(40)
    # A = rng.random((N, N), dtype=np.float32)
    # fill_arr = np.sum(np.abs(A), axis=1)
    # np.fill_diagonal(A, fill_arr)
    # b = rng.random((N, 1), dtype=np.float64)
    A = np.zeros(shape=(N, N), dtype=np.float32)
    b = np.zeros(shape=(N, 1), dtype=np.float64)
    for i in range(N):
        b[i] = 2 * N
        for j in range(N):
            A[i, j] = 1
            A[i, i] = N + 1
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

    jacobiKern.prepared_call(grid, block, A_d, b_d, x_d, x_new_d, d_d, iters)

    # np_act = np.linalg.solve(A, b)
    # np_res = np_jacobi(A, b, iters)
    # tes = test(A, b, N, iters)

    # print(np.linalg.norm(np.abs((np_act - np_res) / np_res)))

    if iters % 2 == 0:
        cuda.memcpy_dtoh(x_new, x_new_d)
        print("CUDA result is:")
        print(sum(x_new))
        # print(np_res)
        # err = np.linalg.norm(np.abs((x_new - np_res) / np_res))
        # print("error is:")
        # print(err)

    else:
        cuda.memcpy_dtoh(x, x_d)
        print("CUDA result is:")
        print(x)
        # err = np.linalg.norm(np.abs((x - np_res) / np_res))
        # print("error is:")
        # print(err)

    cuda.memcpy_dtoh(d, d_d)
    print(d)

testJacobi(512, 3000)
