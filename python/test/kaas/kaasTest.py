import pathlib
import math
import sys
import subprocess as sp
from pprint import pprint

import time
import pickle

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


def runDoublify(kaasHandle, libffCtx, testArray=None):
    if testArray is None:
        testArray = np.random.randn(16).astype(np.float32)

    origArray = testArray.copy()

    libffCtx.kv.put("input", testArray)

    # doublify is in-place
    arguments = [(kaas.bufferSpec('input', testArray.nbytes), 'io')]

    kern = kaas.kernelSpec(testPath / 'kerns' / 'libkaasMicro.cubin',
                           'doublifyKern',
                           (1, 1), (16, 1, 1),
                           arguments=arguments)

    req = kaas.kaasReqDense([kern])

    # This is just for the test, a real system would use libff to invoke the
    # kaas server
    kaasHandle.Invoke(req)

    doubledBytes = libffCtx.kv.get('input')
    doubledArray = np.frombuffer(doubledBytes, dtype=np.float32)

    libffCtx.kv.delete("input")

    return (doubledArray, origArray)


def testStats(mode='direct'):
    stats = ff.util.profCollection()
    libffCtx = getCtx(remote=(mode == 'process'))
    kaasHandle = kaas.kaasFF.getHandle(mode, libffCtx, stats=stats)

    runDoublify(kaasHandle, libffCtx)

    kStats = kaasHandle.getStats()

    # This is supposed to update the original stats object directly. The
    # returned reference is just for convenience or if you didn't pass in a
    # pre-existing stats object
    if kStats is not stats:
        print("FAIL: getStats returned a new object.")
        pprint(kStats.report())
        return

    if 't_init' not in stats:
        print("FAIL: didn't receive any stats from RemoteFunc")
        pprint(kStats.report())
        return

    if 't_zero' not in stats.mod('worker'):
        print("FAIL: didn't receive any stats from KaaS server")
        pprint(kStats.report())
        return

    print("PASS")


def testDoublify(mode='direct'):
    stats = ff.util.profCollection()
    libffCtx = getCtx(remote=(mode == 'process'))
    kaasHandle = kaas.kaasFF.getHandle(mode, libffCtx, stats=stats)

    doubledArray, origArray = runDoublify(kaasHandle, libffCtx)

    expect = origArray*2
    if not np.array_equal(doubledArray, expect):
        print("FAIL")
        print("Expected:")
        print(expect)
        print("Got:")
        print(doubledArray)
    else:
        print("PASS")

    kaasHandle.Close()


def getDotProdReq(nElem):
    nByte = nElem*4
    aBuf = kaas.bufferSpec('inpA', nByte, key="a")
    bBuf = kaas.bufferSpec('inpB', nByte, key="b")
    # lenBuf = kaas.bufferSpec('len', arrLen.nbytes)
    prodOutBuf = kaas.bufferSpec('prodOut', nByte, ephemeral=True)
    cBuf = kaas.bufferSpec('output', 8, key='c')

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

    return kaas.kaasReqDense([prodKern, sumKern])


def testDotProd(mode='direct'):
    nElem = 1024
    # nElem = 32

    libffCtx = getCtx(remote=(mode == 'process'))
    kaasHandle = kaas.kaasFF.getHandle(mode, libffCtx)

    aArr = np.arange(0, nElem, dtype=np.uint32)
    bArr = np.arange(nElem, nElem*2, dtype=np.uint32)

    libffCtx.kv.put('a',   aArr)
    libffCtx.kv.put('b',   bArr)

    req = getDotProdReq(nElem)

    kaasHandle.Invoke(req)
    kaasHandle.Close()

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


def testRekey(mode='direct'):
    nElem = 1024

    libffCtx = getCtx(remote=(mode == 'process'))
    kaasHandle = kaas.kaasFF.getHandle(mode, libffCtx)

    aArr = np.arange(0, nElem, dtype=np.uint32)
    bArr = np.arange(nElem, nElem*2, dtype=np.uint32)
    libffCtx.kv.put('a',   aArr)
    libffCtx.kv.put('b',   bArr)

    req = getDotProdReq(nElem)

    kaasHandle.Invoke(req)

    c = np.frombuffer(libffCtx.kv.get('c'), dtype=np.uint32)[0]

    aNew = np.arange(nElem*2, nElem*3, dtype=np.uint32)
    bNew = np.arange(nElem*3, nElem*4, dtype=np.uint32)
    libffCtx.kv.put('aNew',   aNew)
    libffCtx.kv.put('bNew',   bNew)

    req.reKey({'inpA': 'aNew', 'inpB': 'bNew', 'output': "cNew"})
    kaasHandle.Invoke(req)
    c2 = np.frombuffer(libffCtx.kv.get('cNew'), dtype=np.uint32)[0]

    expect1 = np.dot(aArr, bArr)
    expect2 = np.dot(aNew, bNew)

    if c != expect1:
        print("Failure:")
        print("Original Call Expected: ", expect1)
        print("Got: ", c)
        return False

    if c2 != expect2:
        print("Failure:")
        print("Renamed Call Expected: ", expect2)
        print("Got: ", c2)
        return False

    kaasHandle.Close()
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

    req = kaas.kaasReqDense([kern])

    kaasHandle.Invoke(req)
    kaasHandle.Close()

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


def bigReq():
    nByte = 128

    kerns = []
    lastOut = kaas.bufferSpec('inpFirst', nByte)
    for i in range(1000):
        inBuf = lastOut
        constBuf = kaas.bufferSpec('const' + str(i), nByte)
        outBuf = kaas.bufferSpec('out' + str(i), nByte)

        prodKern = kaas.kernelSpec(testPath / 'kerns' / 'libkaasMicro.cubin',
                                   'prodKern',
                                   (1, 1), (nByte / 4, 1, 1),
                                   literals=[kaas.literalSpec('Q', nByte / 4)],
                                   arguments=[(inBuf, 'i'), (constBuf, 'i'), (outBuf, 'o')])

        kerns.append(prodKern)
        lastOut = outBuf

    keyMap = {}
    keyMap['inpFirst'] = 'rekeyedInpFirst'
    for i in range(1000):
        keyMap['out' + str(i)] = 'reKeyedOut' + str(i)

    fullReq = kaas.kaasReq(kerns)
    denseReq = kaas.kaasReqDense(kerns)

    start = time.time()
    total = 0
    for kern in denseReq.kernels:
        kern = kaas.denseKern(*kern[:-1])
        total += len(kern[0])
    print("parsing dense took: ", time.time() - start)

    start = time.time()
    fullReqSer = pickle.dumps(fullReq)
    print("Serializing full took: ", time.time() - start)

    start = time.time()
    denseReqSer = pickle.dumps(denseReq)
    print("Serializing dense took: ", time.time() - start)

    start = time.time()
    fullReqSer = pickle.loads(fullReqSer)
    print("Deserializing full took: ", time.time() - start)

    start = time.time()
    denseReqDes = pickle.loads(denseReqSer)
    print("Deserializing dense took: ", time.time() - start)

    start = time.time()
    fullReq.reKey(keyMap)
    print("Reykeying full took: ", time.time() - start)

    start = time.time()
    denseReq.reKey(keyMap)
    print("Reykeying dense took: ", time.time() - start)


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

    print("Double Test:")
    with ff.testenv('simple', mode):
        testDoublify(mode)

    print("Stats Test:")
    with ff.testenv('simple', mode):
        testStats(mode)

    print("Dot Product Test:")
    with ff.testenv('simple', mode):
        testDotProd(mode)

    print("MatMul Test")
    with ff.testenv('simple', mode):
        testMatMul(mode)

    print("Rekey:")
    with ff.testenv('simple', mode):
        testRekey(mode)
