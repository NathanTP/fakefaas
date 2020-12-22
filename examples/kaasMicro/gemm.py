import pathlib
import math
from pprint import pprint

import libff as ff
import libff.kv
import libff.invoke

import kaasServer as kaas
import numpy as np

redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"
testPath = pathlib.Path(__file__).resolve().parent
# colMajor = True
colMajor = False

def getCtx(remote=False):
    if remote:
        objStore = ff.kv.Redis(pwd=redisPwd, serialize=True)
    else:
        objStore = ff.kv.Local(copyObjs=False, serialize=False)

    return libff.invoke.RemoteCtx(None, objStore)


rng = np.random.default_rng(0)
def generateArr(shape):
    arr = rng.random(shape, dtype=np.float32)
    if colMajor:
        arr = np.asfortranarray(arr)

    return arr

mmKern = "sgemm"
# mmKern = "matmul"
class mmFunc():
    def _bufArgToBuf(self, arg, bname, shape):
        """Buffer args to various mmFunc methods can be ndarray, or kaasBuf,
        this converts them all to a kaasBuf as appropriate"""
        if isinstance(arg, kaas.bufferSpec):
            return arg
        elif isinstance(arg, np.ndarray):
            self.ffCtx.kv.put(self.name + "_" + bname, arg)
        elif arg is not None:
            raise RuntimeError("Unrecognized type (must be either ndarray or kaas.bufferSpec): " + str(type(arg)))

        # None and ndarray need to generate a bufferSpec
        return kaas.bufferSpec(self.name + "_" + bname, shape[0]*shape[1] * 4)


    def __init__(self, name, shape, libffCtx, kaasHandle, constB=None, mode='direct'):
        """Create a matmul function invoker. Shape should be the two matrix
        dimensions [(arows, acols), (brows, bcols)]. mode can be either
        'direct' or 'process' depending on how you'd like kaas to run.
        
        constB (a numpy array or kaas.bufferSpec) may be provided to associate
        a permanent B associated with this multiplier rather than a dynamic
        one."""
        self.name = name
        self.ffCtx = libffCtx
        self.kHandle = kaasHandle

        self.aShape = shape[0]
        self.bShape = shape[1]
        self.cShape = (shape[0][0], shape[1][1])

        if constB is not None:
            self.constB = self._bufArgToBuf(constB, 'b', self.bShape)
        else:
            self.constB = None

        # dims is a property of a multiplier function, not any particular
        # invocation. We upload it at registration time.
        self.ffCtx.kv.put(self.name+"_dims", np.asarray(list(self.aShape) + list(self.bShape), dtype=np.uint64))
        self.dimBuf = kaas.bufferSpec(self.name + "_dims", 4*8)

        if mmKern == 'sgemm':
            tileN = 32
            tile_tb_height = 16 
            tileM = (tileN * tile_tb_height)
            if self.aShape[0] % tileM != 0 or self.bShape[1] % tileN != 0:
                raise RuntimeError("Arrays must be a multiple of tile size")

            self.gridDim = (self.aShape[0] // tileM, self.bShape[1] // tileN, 1)
            self.blockDim = (tileN, tile_tb_height, 1)
            self.sharedSize = tile_tb_height * tileN * 4
        else:
            threadBlock = 32
            self.gridDim = (math.ceil(self.bShape[0] / threadBlock), math.ceil(self.aShape[0] / threadBlock), 1)
            self.blockDim = (threadBlock, threadBlock, 1)
            self.sharedSize = (2 * blockDim[0] * blockDim[1])*4

       
    def getKernFunc(self, aData=None, bData=None, cData=None):
        """Returns the kernel function object for this multiplier. You can use
        this to compose larger operations. aData, bData, and cData may be:
            None: the buffer will be assumed to exist in the kv store with name 'NAME_BUFNAME' (e.g. 'foo_a')
            np.ndarray: The contents of the array will be uploaded to the kv as 'NAME_BUFNAME'
            kaas.bufferSpec: the buffer spec will be used directly

        Using np.ndarray for cData does nothing as the array will be
        overwritten anyway. If bData is None and the function was created with
        constB, constB will be used.
        """
        aBuf = self._bufArgToBuf(aData, 'a', self.aShape)

        if bData is None and self.constB is not None:
            bBuf = self.constB
        else:
            bBuf = self._bufArgToBuf(aData, 'b', self.aShape)

        cBuf = self._bufArgToBuf(cData, 'c', self.cShape)

        return kaas.kernelSpec(testPath / 'kerns' / 'gemm.cubin',
            mmKern,
            self.gridDim, self.blockDim, sharedSize=self.sharedSize,
            inputs = [self.dimBuf, aBuf, bBuf],
            outputs = [cBuf])


    def invoke(self, aData=None, bData=None, cData=None, times=None):
        """Invoke this multiplier with inputs uploaded with 'name' (using
        uploadArrays()). Invoke returns the name of the result in libff.kv.
        Times may be provided to record invocation statistics. See
        getKernFunc() for details of the data arguments."""
        kern = self.getKernFunc(aData, bData, cData)

        req = kaas.kaasReq([ kern ])
        with libff.timer("invoke", times):
            self.kHandle.Invoke(req.toDict())

        return kern.outputs[0].name 


    def destroy(self):
        """Free any remote resources associated with this function. Any A, B,
        or C buffers (even bConst) are the responsibility of the caller to
        free."""
        # Right now there is nothing else to do as the system doesn't support
        # de-registration of functions. Everything related to this function
        # will eventually get flushed out of the caches anyway.
        self.ffCtx.kv.delete(self.name+"_dims")


def getData(ffCtx, name, shape):
    raw = ffCtx.kv.get(name)
    arr = np.frombuffer(raw, dtype=np.float32)
    if colMajor:
        arr = arr.reshape(shape[0], shape[1], order="F")
    else:
        arr = arr.reshape(shape[0], shape[1])
    
    return arr

def testMM(mode='direct'):
    libffCtx = getCtx(remote=(mode == 'process'))
    kaasHandle = kaas.getHandle(mode, libffCtx)

    # arrA = generateArr((32,32))
    # arrB = generateArr((32,32))
    arrA = generateArr((1024,256))
    arrB = generateArr((256,512))
    libffCtx.kv.put("test_a", arrA)
    libffCtx.kv.put("test_b", arrB)

    shape = [arrA.shape, arrB.shape]
    func = mmFunc('test', shape, libffCtx, kaasHandle, mode=mode)

    rKey = func.invoke()
    
    arrC = getData(libffCtx, rKey, func.cShape)

    libffCtx.kv.delete('test_a')
    libffCtx.kv.delete('test_b')
    libffCtx.kv.delete('test_c')
    func.destroy()

    arrC = arrC.round(4)
    npC = np.matmul(arrA, arrB).round(4)

    # Single precision accumulates errors really fast so the differences can be
    # big. The euclidean distance seems to get us close enough for a pretty
    # good guess at correctness
    diff = arrC - npC
    dist = np.linalg.norm(diff)
    if dist > 1:
        print("FAIL:")
        print("Euclidean Dist: " + str(dist))
        print(np.unique(diff))
        # print("A:\n", arrA)
        # print("B:\n", arrB)
        print("KaaS Result:")
        print(arrC[0][:10])
        print("")

        print("Numpy Result:")
        print(npC[0][:10])
    else:
        print("PASS")

if __name__ == "__main__":
    print("MatMul Test")
    testMM('direct')
