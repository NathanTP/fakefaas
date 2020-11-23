import pathlib

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


def testDoublify(mode='direct'):
    libffCtx = getCtx(remote=True)
    kaasHandle = kaas.getHandle(mode, libffCtx)

    testArray = np.random.randn(16).astype(np.float32)
    origArray = testArray.copy()

    libffCtx.kv.put("input", testArray)

    # doublify is in-place
    inputs  = [ kaas.bufferSpec('input', testArray.nbytes) ]
    outputs = [ kaas.bufferSpec('input', testArray.nbytes) ]

    kern = kaas.kernelSpec(testPath / 'kerns' / 'libkaasMicro.so',
            'doublify',
            4, 4,
            inputs=inputs,
            outputs=outputs)

    req = kaas.kaasReq([ kern ])

    # This is just for the test, a real system would use libff to invoke the
    # kaas server
    kaasHandle.Invoke(req.toDict())

    doubledBytes = libffCtx.kv.get('input')
    doubledArray = np.frombuffer(doubledBytes, dtype=np.float32)

    expect = origArray*2
    if not np.array_equal(doubledArray, expect):
        print("FAIL")
        print("Expected:")
        print(expect)
        print("Got:")
        print(doubledArray)
    else:
        print("PASS")


def testDotProd(mode='direct'):
    nElem = 1024
    # nElem = 32 
    nByte = nElem*4

    libffCtx = getCtx(remote=True)
    kaasHandle = kaas.getHandle(mode, libffCtx)

    aArr = np.arange(0,nElem, dtype=np.uint32)
    bArr = np.arange(nElem,nElem*2, dtype=np.uint32)
    arrLen = np.uint64(nElem)

    # Getting zcopy bytes out of numpy arrays is tricky. Even these memoryviews
    # don't act like bytes (len returns nelem, not nbyte). You have to use
    # buf.nbytes to get the byte length. array.tobytes(), bytes(array), and
    # bytebuffer(array) give copies.
    # aBytes = memoryview(aArr)
    # bBytes = memoryview(bArr)
    # lenBytes = memoryview(arrLen)

    libffCtx.kv.put('a',   aArr)
    libffCtx.kv.put('b',   bArr)
    libffCtx.kv.put('len', arrLen)
    
    aBuf = kaas.bufferSpec('a', nByte)
    bBuf = kaas.bufferSpec('b', nByte)
    lenBuf = kaas.bufferSpec('len', arrLen.nbytes)
    prodOutBuf = kaas.bufferSpec('prodOut', nByte, ephemeral=True)
    cBuf = kaas.bufferSpec('c', 8)

    prodKern = kaas.kernelSpec(testPath / 'kerns' / 'libkaasMicro.so',
            'prod',
            1, nElem,
            inputs=[aBuf, bBuf, lenBuf],
            outputs=[prodOutBuf])

    sumKern = kaas.kernelSpec(testPath / 'kerns' / 'libkaasMicro.so',
            'sum',
            1, nElem // 2,
            inputs = [prodOutBuf],
            outputs = [cBuf])

    req = kaas.kaasReq([ prodKern, sumKern ])

    kaasHandle.Invoke(req.toDict())

    c = np.frombuffer(libffCtx.kv.get('c'), dtype=np.uint32)[0]
    expect = np.dot(aArr, bArr)
    if c != expect:
        print("Failure:")
        print("Expected: ", np.dot(aArr, bArr))
        print("Got: ", c)
    else:
        print("PASS")

if __name__ == "__main__":
    print("Dot Product Test:") 
    testDotProd('process')
    print("Double Test:")
    testDoublify('process')
