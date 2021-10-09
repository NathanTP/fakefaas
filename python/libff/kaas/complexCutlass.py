import os
import ctypes as ct
from numpy import ctypeslib

# define complex ctype as a python class
class complex(ct.Structure):
    _fields_ = [('real', ct.c_float), ('imag', ct.c_float)]

c_complex_p = ct.POINTER(complex)


dir_path = os.path.dirname(os.path.realpath(__file__))

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

def loadAdapter():
    libc = ct.cdll.LoadLibrary(str(dir_path) + "/complexCutlass/cutlassAdapters.so")
    getArg = libc.adaptSGEMMArgs
    getArg.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_float, c_complex_p, ct.c_int,
                       c_complex_p, ct.c_int, ct.c_float, c_complex_p, ct.c_int]
    # Instead of trying to define the Params struct in python, we just pretend
    # that it's a byte array of the same size (320 bytes in this case)
    getArg.restype = ct.POINTER(ct.c_byte*328)

    getDims = libc.getCudaConfig
    # M, N, K
    getDims.argtypes = [ct.c_int, ct.c_int, ct.c_int]
    getDims.restype = ct.POINTER(kernelConfig)

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
    thing = ct.cast(int(a_d), c_complex_p)
    params = cutlassAdapter(M, N, K, alpha,
                    thing, lda,
                    ct.cast(int(b_d), c_complex_p), ldb,
                    beta,
                    ct.cast(int(c_d), c_complex_p), ldc)
    return params
