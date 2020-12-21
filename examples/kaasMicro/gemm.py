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

def getCtx(remote=False):
    if remote:
        objStore = ff.kv.Redis(pwd=redisPwd, serialize=True)
    else:
        objStore = ff.kv.Local(copyObjs=False, serialize=False)

    return libff.invoke.RemoteCtx(None, objStore)


rng = np.random.default_rng(0)
def generateArr(shape):
    arr = rng.random(shape, dtype=np.float32)
    return arr


class mmFunc():
    def __init__(self, shape, libffCtx, kaasHandle, mode='direct'):
        """Create a matmul function invoker. Shape should be the two matrix
        dimensions [(arows, acols), (brows, bcols)]. mode can be either
        'direct' or 'process' depending on how you'd like kaas to run."""
        self.ffCtx = libffCtx
        self.kHandle = kaasHandle

        threadBlock = 32

        self.aShape = shape[0]
        self.bShape = shape[1]

        # These are just templates, the names will be overwritten when invoked
        self.aBuf = kaas.bufferSpec('A', self.aShape[0] * self.aShape[1] * 4)
        self.bBuf = kaas.bufferSpec('B', self.bShape[0] * self.bShape[1] * 4)
        self.cBuf = kaas.bufferSpec('C', self.aShape[0] * self.bShape[1] * 4)

        # 4 uint64s
        self.dimBuf = kaas.bufferSpec("dims", 4*8)

        gridDim = (math.ceil(self.bShape[0] / threadBlock), math.ceil(self.aShape[0] / threadBlock), 1)
        blockDim = (threadBlock, threadBlock, 1)
        sharedSize = 2 * blockDim[0] * blockDim[1]

        self.kern = kaas.kernelSpec(testPath / 'kerns' / 'libkaasMicro.cubin',
            'matmulKern',
            gridDim, blockDim, sharedSize=sharedSize,
            inputs = [self.dimBuf, self.aBuf, self.bBuf],
            outputs = [self.cBuf])
       

    def invoke(self, name):
        """Invoke this multiplier with inputs uploaded with 'name' (using
        uploadArrays()). Invoke returns the name of the result in libff.kv."""

        # References to these are stored in self.kern
        self.aBuf.name = name+"_a" 
        self.bBuf.name = name+"_b" 
        self.cBuf.name = name+"_c"
        self.dimBuf.name = name+"_dims"

        req = kaas.kaasReq([ self.kern ])
        self.kHandle.Invoke(req.toDict())

        return name+"_c"


class mmData():
    def __init__(self, name, shape, ctx: libff.invoke.RemoteCtx):
        """Represents a set of matrices will be used for the kaas matmul
        routine.  Name will be used to prefix all elements in the kv store.
        Shape represents both A and B arrays: ((a_nrow, a_ncol), (b_nrow,
        b_ncol))."""
        self.name = name
        self.shape = shape
        self.ctx = ctx


    def upload(self, a, b):
        """Upload a and b matrices. No local reference will be maintained to a
        or b. They should be numpy 2d arrays of float32.""" 
        self.ctx.kv.put(self.name+"_dims", np.asarray(list(self.shape[0]) + list(self.shape[1]), dtype=np.uint64))
        self.ctx.kv.put(self.name+"_a", a)
        self.ctx.kv.put(self.name+"_b", b)


    def getResult(self):
        cRaw = self.ctx.kv.get(self.name+"_c")
        cArr = np.frombuffer(cRaw, dtype=np.float32)
        cArr = cArr.reshape(self.shape[0][0], self.shape[1][1])
        # cArr = cArr.reshape(self.shape[0][0], self.shape[1][1], order="F")
        return cArr

    
    def clean(self):
        self.ctx.kv.delete(self.name+"_a")
        self.ctx.kv.delete(self.name+"_b")
        self.ctx.kv.delete(self.name+"_c")
        self.ctx.kv.delete(self.name+"_dims")


def testMM(mode='direct'):
    libffCtx = getCtx(remote=(mode == 'process'))
    kaasHandle = kaas.getHandle(mode, libffCtx)

    arrA = generateArr((3,2))
    arrB = generateArr((2,7))
    shape = [arrA.shape, arrB.shape]

    data = mmData("test", shape, libffCtx)
    data.upload(arrA, arrB)

    func = mmFunc(shape, libffCtx, kaasHandle, mode)

    func.invoke("test")

    arrC = data.getResult()
    data.clean()

    arrC = arrC.round(4)
    npC = np.matmul(arrA, arrB).round(4)
    if(not np.array_equal(arrC, npC)):
        print("FAIL:")
        print("A:\n", arrA)
        print("B:\n", arrB)
        print("KaaS Result:")
        print(arrC)
        print("")

        print("Numpy Result:")
        print(npC)
    else:
        print("PASS")

if __name__ == "__main__":
    print("MatMul Test")
    testMM('direct')
