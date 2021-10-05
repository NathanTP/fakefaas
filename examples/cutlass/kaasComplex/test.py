import pycuda.driver as cuda
import pycuda.autoinit  # NOQA
import numpy as np
from numpy import ctypeslib
import ctypes as ct

# c_complex_p = ctypeslib.ndpointer(np.csingle, ndim=2, flags='C')
class complex(ct.Structure):
    _fields_ = [('real', ct.c_float), ('imag', ct.c_float)]

# c_complex_p = ct.POINTER(ct.c_byte*8)
c_complex_p = ct.POINTER(complex)

def loadKerns():
    mod = cuda.module_from_file("./cutlass.cubin")

    # Since the cutlass kernel came from a template, the name is crazy long.
    # Unfortunately, extern "C" doesn't fix the issue. This string is obtained
    # by running "nm" on the cubin
    cutlassKern = mod.get_function("_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock12MmaPipelinedINS1_9GemmShapeILi128ELi128ELi8EEENS_9transform11threadblock22PredicatedTileIteratorINS_11MatrixShapeILi128ELi8EEENS_7complexIfEENS_6layout8RowMajorELi1ENS8_30PitchLinearStripminedThreadMapINSF_16PitchLinearShapeILi8ELi128EEELi256ELi1EEELi1EEENS9_19RegularTileIteratorISC_SE_NSF_11ColumnMajorELi1ENS8_33TransposePitchLinearThreadMapSimtISK_EELi8EEENSA_INSB_ILi8ELi128EEESE_SG_Li0ENSH_INSI_ILi128ELi8EEELi256ELi1EEELi1EEENSM_ISR_SE_SG_Li0EST_Li8EEESE_SG_NS4_9MmaPolicyINS1_4warp7MmaSimtINS6_ILi32ELi64ELi8EEESE_SN_SE_SG_SE_SG_NSX_13MmaSimtPolicyINSB_ILi4ELi8EEENSF_19RowMajorInterleavedILi2EEENS6_ILi2ELi2ELi1EEEEELi1ELNS_16ComplexTransformE0ELS16_0EbEENSB_ILi2ELi0EEENSB_ILi0ELi0EEELi1EEENS_21NumericArrayConverterISE_SE_Li4ELNS_15FloatRoundStyleE2EEES1D_bEENS_8epilogue11threadblock8EpilogueIS7_S17_Li1ENS1G_22PredicatedTileIteratorINS1G_26OutputTileOptimalThreadMapINS1G_15OutputTileShapeILi128ELi1ELi4ELi4ELi1EEENS1K_ILi1ELi2ELi4ELi1ELi8EEELi256ELi1ELi64EEESE_EENS1F_4warp20FragmentIteratorSimtISZ_NS1_6thread3MmaINS6_ILi8ELi8ELi1EEESE_SN_SE_SG_SE_SG_NS_4arch13OpMultiplyAddEbEESG_S15_EENS1P_16TileIteratorSimtISZ_S1W_SE_SG_S15_EENS1G_18SharedLoadIteratorINS1N_18CompactedThreadMapESE_Li8EEENS1F_6thread17LinearCombinationISE_Li1ESE_SE_LNS23_9ScaleType4KindE0ELS1C_2EEENSB_ILi0ELi9EEELi1EEENS4_30GemmIdentityThreadblockSwizzleILi1EEELb0EEEEEvNT_6ParamsE")

    # The kernel takes a Params struct as an argument (by value). Rather than
    # try to define that struct in python, we instead find its size manually
    # (in cuda using sizeof()) and then specify a byte array of the same size
    # here. Pycuda doesn't care about the type in practice, it only needs the
    # size. This type string is defined by python's "struct" module.
    cutlassKern.prepare("328s")

    refKern = mod.get_function("ReferenceGemm_kernel")
    # See python's struct module for a description of this type string
    refKern.prepare("iiifPiPifPi")

    return refKern, cutlassKern


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
    libc = ct.cdll.LoadLibrary("./cutlassAdapters.so")
    getArg = libc.adaptSGEMMArgs
    getArg.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_float, c_complex_p, ct.c_int,
                       c_complex_p, ct.c_int, ct.c_float, c_complex_p, ct.c_int]
    # getArg.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_float, ct.c_void_p, ct.c_int, 
    #                     ct.c_void_p, ct.c_int, ct.c_float, ct.c_void_p, ct.c_int]
    # Instead of trying to define the Params struct in python, we just pretend
    # that it's a byte array of the same size (320 bytes in this case)
    getArg.restype = ct.POINTER(ct.c_byte*328)

    getDims = libc.getCudaConfig
    # M, N, K
    getDims.argtypes = [ct.c_int, ct.c_int, ct.c_int]
    getDims.restype = ct.POINTER(kernelConfig)

    return (getArg, getDims)


def testKern():
    """This kernel is useful for trying out ideas without all the complexity of
    cutlass"""
    mod = cuda.module_from_file("./cutlass.cubin")
    kern = mod.get_function("testKernel")
    kern.prepare("16s")

    lib = ct.cdll.LoadLibrary("./cutlassAdapters.so")
    getStruct = lib.getTestStruct
    getStruct.argtypes = [ct.c_int, ct.POINTER(complex)]
    getStruct.restype = ct.POINTER(ct.c_byte*16)

    anInt = 42
    hArr = np.arange(4096*2, dtype=np.csingle) * (1+1j)
    print("numpy sum is")
    print(np.sum(hArr))

    dArr = cuda.mem_alloc(hArr.nbytes)
    cuda.memcpy_htod(dArr, hArr)

    arg = getStruct(anInt, ct.cast(int(dArr), ct.POINTER(complex)))

    kern.prepared_call((1, 1, 1), (1, 1, 1), arg.contents)


def testSgemm(M, N, K, alpha, beta):
    lda = M
    ldb = K
    ldc = M

    getArg, getDims = loadAdapter()
    refKern, cutlassKern = loadKerns()

    rng = np.random.default_rng(5)
    a = np.asfortranarray(rng.random((M, K), dtype=np.float32) * (1+1j))
    b = np.asfortranarray(rng.random((K, N), dtype=np.float32) * (1+1j))
    c = np.asfortranarray(np.zeros(shape=(M, N), dtype=np.csingle))

    a_d = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_d, a)

    b_d = cuda.mem_alloc(b.nbytes)
    cuda.memcpy_htod(b_d, b)

    c_d = cuda.mem_alloc(c.nbytes)
    cuda.memset_d8(c_d, 0, c.nbytes)

    cfg = getDims(M, N, K).contents
    grid = (cfg.gridX, cfg.gridY, cfg.gridZ)
    block = (cfg.blockX, cfg.blockY, cfg.blockZ)

    print("Grid is: ", grid)
    print("Block is: ", block)
    print("Smem Size is: ", cfg.smem_size)

    params = getArg(M, N, K, alpha,
                    ct.cast(int(a_d), c_complex_p), lda,
                    ct.cast(int(b_d), c_complex_p), ldb,
                    beta,
                    ct.cast(int(c_d), c_complex_p), ldc)            

    cutlassKern.prepared_call(grid, block, params.contents, shared_size=cfg.smem_size)

    cuda.memcpy_dtoh(c, c_d)

    refC_d = cuda.mem_alloc(c.nbytes)

    refBlock = (16, 16, 1)
    refGrid = (((M + 16 - 1) // 16), ((N + 16 - 1) // 16), 1)
    refKern.prepared_call(refBlock, refGrid, M, N, K, alpha, a_d, lda, b_d, ldb, beta, refC_d, ldc)

    refC_h = np.asfortranarray(np.zeros(shape=(M, N), dtype=np.csingle))
    cuda.memcpy_dtoh(refC_h, refC_d)

    print("Cutlass Kern Result:")
    print(c)

    print("Reference Kern Result:")
    print(refC_h)

    print("NP Result: ")
    # np_res = np.matmul(a.T, b.T).T
    np_res = np.matmul(a, b)
    print(np_res)

    # Check difference
    print("Difference between numpy and reference result:")
    print(np_res - refC_h)
    print("Difference between cutlass and numpy result:")
    print(c - np_res)

# a = complex(1.0, 0.0)
# b = complex(0.0, 0.0)
# testSgemm(128, 128, 128, a, b)
testSgemm(128, 128, 128, 1, 0)
# testKern()
