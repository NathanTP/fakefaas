import os
import ctypes as ct

dir_path = os.path.dirname(os.path.realpath(__file__))


def loadSgemmAdapter():
    libc = ct.cdll.LoadLibrary(str(dir_path) + "/cutlass/cutlassAdapters.so")
    getArg = libc.adaptSGEMMArgs
    c_float_p = ct.POINTER(ct.c_float)
    getArg.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_float, c_float_p, ct.c_int,
                       c_float_p, ct.c_int, ct.c_float, c_float_p, ct.c_int]
    getArg.restype = ct.POINTER(ct.c_byte*320)
    return getArg


def parseSgemmArgs(literalVals, dAddrs, cutlassAdapter):
    M = literalVals[2]
    N = literalVals[3]
    K = literalVals[4]
    lda = literalVals[5]
    ldb = literalVals[6]
    ldc = literalVals[7]
    alpha = literalVals[0]
    beta = literalVals[1]
    a_d = dAddrs[0]
    b_d = dAddrs[1]
    c_d = dAddrs[2]
    params = cutlassAdapter(M, N, K, alpha,
                            ct.cast(int(a_d), ct.POINTER(ct.c_float)), lda,
                            ct.cast(int(b_d), ct.POINTER(ct.c_float)), ldb,
                            beta,
                            ct.cast(int(c_d), ct.POINTER(ct.c_float)), ldc)
    return params
