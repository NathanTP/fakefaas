import pycuda.driver as cuda
import pycuda.autoinit  # NOQA
import numpy as np
import ctypes as ct
import libff.kaas as kaas
import libff.kv
import libff.invoke
import libff.kaas.kaasFF
import libff as ff

redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"


def loadKerns():
    mod = cuda.module_from_file("./cutlass.cubin")

    # Since the cutlass kernel came from a template, the name is crazy long.
    # Unfortunately, extern "C" doesn't fix the issue. This string is obtained
    # by running "nm" on the cubin
    cutlassKern = mod.get_function("_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock12MmaPipelinedINS1_9GemmShapeILi128ELi128ELi8EEENS_9transform11threadblock22PredicatedTileIteratorINS_11MatrixShapeILi128ELi8EEEfNS_6layout8RowMajorELi1ENS8_30PitchLinearStripminedThreadMapINSD_16PitchLinearShapeILi8ELi128EEELi256ELi1EEELi1EEENS9_19RegularTileIteratorISC_fNSD_11ColumnMajorELi1ENS8_33TransposePitchLinearThreadMapSimtISI_EELi4EEENSA_INSB_ILi8ELi128EEEfSE_Li0ENSF_INSG_ILi128ELi8EEELi256ELi1EEELi1EEENSK_ISP_fSE_Li0ESR_Li4EEEfSE_NS4_9MmaPolicyINS1_4warp7MmaSimtINS6_ILi32ELi64ELi8EEEfSL_fSE_fSE_NSV_13MmaSimtPolicyINSB_ILi4ELi8EEENSD_19RowMajorInterleavedILi2EEENS6_ILi4ELi4ELi1EEEEELi1ELNS_16ComplexTransformE0ELS14_0EbEENSB_ILi4ELi0EEENSB_ILi0ELi0EEELi1EEENS_21NumericArrayConverterIffLi4ELNS_15FloatRoundStyleE2EEES1B_bEENS_8epilogue11threadblock8EpilogueIS7_S15_Li1ENS1E_22PredicatedTileIteratorINS1E_26OutputTileOptimalThreadMapINS1E_15OutputTileShapeILi128ELi1ELi4ELi4ELi1EEENS1I_ILi1ELi4ELi2ELi1ELi8EEELi256ELi1ELi32EEEfEENS1D_4warp20FragmentIteratorSimtISX_NS1_6thread3MmaINS6_ILi8ELi8ELi1EEEfSL_fSE_fSE_NS_4arch13OpMultiplyAddEbEESE_S13_EENS1N_16TileIteratorSimtISX_S1U_fSE_S13_EENS1E_18SharedLoadIteratorINS1L_18CompactedThreadMapEfLi4EEENS1D_6thread17LinearCombinationIfLi1EffLNS21_9ScaleType4KindE0ELS1A_2EEENSB_ILi0ELi17EEELi1EEENS4_30GemmIdentityThreadblockSwizzleILi1EEELb0EEEEEvNT_6ParamsE")

    # The kernel takes a Params struct as an argument (by value). Rather than
    # try to define that struct in python, we instead find its size manually
    # (in cuda using sizeof()) and then specify a byte array of the same size
    # here. Pycuda doesn't care about the type in practice, it only needs the
    # size. This type string is defined by python's "struct" module.
    cutlassKern.prepare("320s")

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
    c_float_p = ct.POINTER(ct.c_float)
    getArg.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_float, c_float_p, ct.c_int,
                       c_float_p, ct.c_int, ct.c_float, c_float_p, ct.c_int]
    # Instead of trying to define the Params struct in python, we just pretend
    # that it's a byte array of the same size (320 bytes in this case)
    getArg.restype = ct.POINTER(ct.c_byte*320)

    getDims = libc.getCudaConfig
    # M, N, K
    getDims.argtypes = [ct.c_int, ct.c_int, ct.c_int]
    getDims.restype = ct.POINTER(kernelConfig)

    return (getArg, getDims)


def getCtx(remote=False):
    if remote:
        objStore = ff.kv.Redis(pwd=redisPwd, serialize=True)
    else:
        objStore = ff.kv.Local(copyObjs=False, serialize=False)

    return libff.invoke.RemoteCtx(None, objStore)


def testKern():
    """This kernel is useful for trying out ideas without all the complexity of
    cutlass"""
    mod = cuda.module_from_file("./cutlass.cubin")
    kern = mod.get_function("testKernel")
    kern.prepare("16s")

    lib = ct.cdll.LoadLibrary("./cutlassAdapters.so")
    getStruct = lib.getTestStruct
    getStruct.argtypes = [ct.c_int, ct.POINTER(ct.c_float)]
    getStruct.restype = ct.POINTER(ct.c_byte*16)

    anInt = 42
    hArr = np.arange(4096*2, dtype=np.float32)

    dArr = cuda.mem_alloc(hArr.nbytes)
    cuda.memcpy_htod(dArr, hArr)

    arg = getStruct(anInt, ct.cast(int(dArr), ct.POINTER(ct.c_float)))

    kern.prepared_call((1, 1, 1), (1, 1, 1), arg.contents)


def testSgemmKaas(M, N, K, alpha, beta):
    lda = M
    ldb = K
    ldc = M

    libffCtx = getCtx(remote=False)

    rng = np.random.default_rng(0)
    a = rng.random((M, K), dtype=np.float32)
    b = rng.random((K, N), dtype=np.float32)
    c = np.zeros(shape=(M, N), dtype=np.float32)

    getArg, getDims = loadAdapter()

    cfg = getDims(M, N, K).contents
    grid = (cfg.gridX, cfg.gridY, cfg.gridZ)
    block = (cfg.blockX, cfg.blockY, cfg.blockZ)

    smem = cfg.smem_size

    libffCtx.kv.put('a', a)
    aBuf = kaas.bufferSpec('a', a.nbytes)

    libffCtx.kv.put('b', b)
    bBuf = kaas.bufferSpec('b', b.nbytes)

    libffCtx.kv.put('c', c)
    cBuf = kaas.bufferSpec('c', c.nbytes)
    literals = [kaas.literalSpec('f', alpha), kaas.literalSpec('f', beta),
                kaas.literalSpec('f', M), kaas.literalSpec('f', N), kaas.literalSpec('f', K), kaas.literalSpec('f', lda), kaas.literalSpec('f', ldb), kaas.literalSpec('f', ldc)]
    firstKern = kaas.kernelSpec(kaas.builtins["cutlass"], "sgemm0", grid, block, sharedSize=smem, arguments=[(aBuf, 'i'), (bBuf, 'i'), (cBuf, 'o')], literals=literals)

    req = kaas.kaasReq([firstKern])
    kaasHandle = kaas.kaasFF.getHandle("direct", libffCtx)
    kaasHandle.Invoke(req.toDict())

    c = np.frombuffer(libffCtx.kv.get('c'), dtype=np.float32)
    print(c)


def testSgemm(M, N, K, alpha, beta):
    lda = M
    ldb = K
    ldc = M

    getArg, getDims = loadAdapter()
    refKern, cutlassKern = loadKerns()

    rng = np.random.default_rng(0)
    a = np.asfortranarray(rng.random((M, K), dtype=np.float32))
    b = np.asfortranarray(rng.random((K, N), dtype=np.float32))
    c = np.asfortranarray(np.zeros(shape=(M, N), dtype=np.float32))

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
    import time
    timeStart = time.time()

    params = getArg(M, N, K, alpha,
                    ct.cast(int(a_d), ct.POINTER(ct.c_float)), lda,
                    ct.cast(int(b_d), ct.POINTER(ct.c_float)), ldb,
                    beta,
                    ct.cast(int(c_d), ct.POINTER(ct.c_float)), ldc)

    cutlassKern.prepared_call(grid, block, params.contents, shared_size=cfg.smem_size)
    cuda.Context.synchronize()
    print(time.time() - timeStart)

    cuda.memcpy_dtoh(c, c_d)

    refC_d = cuda.mem_alloc(c.nbytes)
    cuda.memset_d8(refC_d, 0, c.nbytes)

    refBlock = (16, 16, 1)
    refGrid = (((M + 16 - 1) // 16), ((N + 16 - 1) // 16), 1)
    refKern.prepared_call(refGrid, refBlock, M, N, K, alpha, a_d, lda, b_d, ldb, beta, refC_d, ldc)

    refC_h = np.asfortranarray(np.zeros(shape=(M, N), dtype=np.float32))
    cuda.memcpy_dtoh(refC_h, refC_d)

    print("Cutlass Kern Result:")
    print(c)

    #print("Reference Kern Result:")
    #print(refC_h)

    a = np.reshape(a, (K, M), order='F')
    b = np.reshape(b, (N, K), order='F')
    print("NP Result: ")
    # np_res = np.matmul(a.T, b.T).T
    np_res = np.matmul(a, b)
    print(np_res)

    # Check difference
    print("Difference between numpy and reference result:")
    print(np_res - refC_h)
    print("Difference between numpy and cutlass result:")
    print(np_res - c)


testSgemm(128, 128, 128, 1.0, 0.0)
#testSgemm(10000, 8000, 10000, 1.0, 0.0) #-> 1.12 seconds
