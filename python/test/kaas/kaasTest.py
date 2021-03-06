import pathlib
import math
import sys
import subprocess as sp

import libff as ff
import libff.kv
import libff.invoke

import libff.kaas as kaas
import libff.kaas.kaasFF

import numpy as np

redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"
testPath = pathlib.Path(__file__).resolve().parent


def getCtx(remote=False):
    if remote:
        objStore = ff.kv.Redis(pwd=redisPwd, serialize=True)
    else:
        objStore = ff.kv.Local(copyObjs=False, serialize=False)

    return libff.invoke.RemoteCtx(None, objStore)


def testDoublify(mode='direct'):
    stats = ff.util.profCollection()
    libffCtx = getCtx(remote=(mode == 'process'))
    kaasHandle = kaas.kaasFF.getHandle(mode, libffCtx, stats=stats)

    testArray = np.random.randn(16).astype(np.float32)
    origArray = testArray.copy()

    libffCtx.kv.put("input", testArray)

    # doublify is in-place
    arguments = [(kaas.bufferSpec('input', testArray.nbytes), 'io')]

    kern = kaas.kernelSpec(testPath / 'kerns' / 'libkaasMicro.cubin',
                           'doublifyKern',
                           (1, 1), (16, 1, 1),
                           arguments=arguments)

    req = kaas.kaasReq([kern])

    # This is just for the test, a real system would use libff to invoke the
    # kaas server
    kaasHandle.Invoke(req.toDict())

    doubledBytes = libffCtx.kv.get('input')
    doubledArray = np.frombuffer(doubledBytes, dtype=np.float32)

    libffCtx.kv.delete("input")

    expect = origArray*2
    if not np.array_equal(doubledArray, expect):
        print("FAIL")
        print("Expected:")
        print(expect)
        print("Got:")
        print(doubledArray)
    else:
        print("PASS")

    print("Stats: ")
    print(kaasHandle.getStats())


def testDotProd(mode='direct'):
    nElem = 1024
    # nElem = 32
    nByte = nElem*4

    libffCtx = getCtx(remote=(mode == 'process'))
    kaasHandle = kaas.kaasFF.getHandle(mode, libffCtx)

    aArr = np.arange(0, nElem, dtype=np.uint32)
    bArr = np.arange(nElem, nElem*2, dtype=np.uint32)
    # arrLen = np.uint64(nElem)

    libffCtx.kv.put('a',   aArr)
    libffCtx.kv.put('b',   bArr)
    # libffCtx.kv.put('len', arrLen)

    aBuf = kaas.bufferSpec('a', nByte)
    bBuf = kaas.bufferSpec('b', nByte)
    # lenBuf = kaas.bufferSpec('len', arrLen.nbytes)
    prodOutBuf = kaas.bufferSpec('prodOut', nByte, ephemeral=True)
    cBuf = kaas.bufferSpec('c', 8)

    args_prod = [(aBuf, 'i'), (bBuf, 'i'), (prodOutBuf, 'o')]

    prodKern = kaas.kernelSpec(testPath / 'kerns' / 'libkaasMicro.cubin',
                               'prodKern',
                               (1, 1), (nElem, 1, 1),
                               literals=[kaas.literalSpec('Q', nElem)],
                               arguments=args_prod)

    args_sum = [(prodOutBuf, 'i'), (cBuf, 'o')]

    sumKern = kaas.kernelSpec(testPath / 'kerns' / 'libkaasMicro.cubin',
                              'sumKern',
                              (1, 1), (nElem // 2, 1, 1),
                              arguments=args_sum)

    req = kaas.kaasReq([prodKern, sumKern])

    kaasHandle.Invoke(req.toDict())

    c = np.frombuffer(libffCtx.kv.get('c'), dtype=np.uint32)[0]

    libffCtx.kv.delete("a")
    libffCtx.kv.delete("b")
    # libffCtx.kv.delete("len")
    libffCtx.kv.delete("c")

    expect = np.dot(aArr, bArr)
    if c != expect:
        print("Failure:")
        print("Expected: ", np.dot(aArr, bArr))
        print("Got: ", c)
    else:
        print("PASS")


rng = np.random.default_rng()


def generateArr(shape):
    return rng.random(shape, dtype=np.float32)


def testMatMul(mode='direct'):
    libffCtx = getCtx(remote=(mode == 'process'))
    kaasHandle = kaas.kaasFF.getHandle(mode, libffCtx)

    arrA = generateArr((32, 32))
    arrB = generateArr((32, 32))
    dims = np.asarray(list(arrA.shape) + list(arrB.shape), dtype=np.uint64)

    libffCtx.kv.put('dims', dims)
    libffCtx.kv.put("A", arrA)
    libffCtx.kv.put("B", arrB)

    aBuf = kaas.bufferSpec('A', arrA.nbytes)
    bBuf = kaas.bufferSpec('B', arrB.nbytes)
    dimBuf = kaas.bufferSpec("dims", dims.nbytes)

    # C is aRows x bCols
    cSize = arrA.shape[0]*arrB.shape[1]*4
    cBuf = kaas.bufferSpec('C', cSize)

    threadBlock = 32
    gridDim = (math.ceil(arrB.shape[0] / threadBlock), math.ceil(arrA.shape[0] / threadBlock), 1)
    blockDim = (threadBlock, threadBlock, 1)
    sharedSize = 2 * blockDim[0] * blockDim[1]

    args = [(dimBuf, 'i'), (cBuf, 'o'), (bBuf, 'i'), (aBuf, 'i')]

    kern = kaas.kernelSpec(testPath / 'kerns' / 'libkaasMicro.cubin',
                           'matmulKern',
                           gridDim, blockDim, sharedSize=sharedSize,
                           arguments=args)

    req = kaas.kaasReq([kern])

    kaasHandle.Invoke(req.toDict())

    cRaw = libffCtx.kv.get('C')
    cArr = np.frombuffer(cRaw, dtype=np.float32)
    cArr = cArr.reshape(arrA.shape[0], arrB.shape[1])

    libffCtx.kv.delete("dims")
    libffCtx.kv.delete("A")
    libffCtx.kv.delete("B")
    libffCtx.kv.delete("C")

    npArr = np.matmul(arrA, arrB).round(4)

    # lots of rounding error in float32 matmul, use euclidean distance to make
    # sure we're close. Usually if something goes wrong, it goes very wrong.
    dist = np.linalg.norm(cArr - npArr)
    if dist > 10:
        print("FAIL:")
        print("Distance: ", dist)
        print("A:\n", arrA)
        print("B:\n", arrB)
        print("KaaS Result:")
        print(cArr)
        print("")

        print("Numpy Result:")
        print(npArr)
    else:
        print("PASS")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./kaasTest.py ['direct'|'process']")
        sys.exit(1)
    else:
        mode = sys.argv[1]

    if not (testPath / 'kerns' / 'libkaasMicro.cubin').exists():
        print("Test cubin not available, building now:")
        sp.call(["make"], cwd=(testPath / 'kerns'))
        print("libkaasMicro.cubin built sucessefully\n")

    if not libff.invoke.cudaAvailable:
        print("KaaS requires CUDA support from libff")
        sys.exit(1)

    print("Double Test:")
    with ff.testenv('simple', mode):
        testDoublify(mode)

    print("Dot Product Test:")
    with ff.testenv('simple', mode):
        testDotProd(mode)

    print("MatMul Test")
    with ff.testenv('simple', mode):
        testMatMul(mode)
