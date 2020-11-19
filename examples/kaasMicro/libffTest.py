import pathlib

import libff as ff
import libff.kv
import libff.invoke

import kaasServer as kaas
import numpy as np

redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"
testPath = pathlib.Path(".").resolve()

def getCtx(remote=False):
    if remote:
        objStore = ff.kv.Redis(pwd=redisPwd, serialize=True)
    else:
        objStore = ff.kv.Local(copyObjs=False, serialize=False)

    return libff.invoke.RemoteCtx(None, objStore)


def testDoublify():
    libffCtx = getCtx(remote=False)

    testArray = np.random.randn(4,4).astype(np.float32)
    print("Orig:")
    print(testArray)

    testBytes = memoryview(testArray)
    libffCtx.kv.put("input", testBytes)
    print(testBytes.nbytes)

    # doublify is in-place
    inputs  = [ kaas.bufferSpec('input', testBytes.nbytes) ]
    outputs = [ kaas.bufferSpec('input', testBytes.nbytes) ]

    kern = kaas.kernelSpec(testPath / 'kerns' / 'libkaasMicro.so',
            'doublify',
            4, 4,
            inputs=inputs,
            outputs=outputs)

    req = kaas.kaasReq([ kern ])

    # This is just for the test, a real system would use libff to invoke the
    # kaas server
    kaas.kaasServe(req.toDict(), libffCtx)

    doubledBytes = libffCtx.kv.get('input')
    doubledArray = np.frombuffer(doubledBytes, dtype=np.float32)

    print("Doubled:")
    print(doubledArray)


def testDotProd():
    nElem = 1024
    # nElem = 32 
    nByte = nElem*4

    libffCtx = getCtx(remote=False)

    aArr = np.arange(0,nElem, dtype=np.uint32)
    bArr = np.arange(nElem,nElem*2, dtype=np.uint32)
    arrLen = np.uint64(nElem)

    # Getting zcopy bytes out of numpy arrays is tricky. Even these memoryviews
    # don't act like bytes (len returns nelem, not nbyte). You have to use
    # buf.nbytes to get the byte length. array.tobytes(), bytes(array), and
    # bytebuffer(array) give copies.
    aBytes = memoryview(aArr)
    bBytes = memoryview(bArr)
    lenBytes = memoryview(arrLen)

    libffCtx.kv.put('a',   aBytes)
    libffCtx.kv.put('b',   bBytes)
    libffCtx.kv.put('len', lenBytes)
    
    aBuf = kaas.bufferSpec('a', nByte)
    bBuf = kaas.bufferSpec('b', nByte)
    lenBuf = kaas.bufferSpec('len', lenBytes.nbytes)
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

    kaas.kaasServe(req.toDict(), libffCtx)

    c = np.frombuffer(libffCtx.kv.get('c'), dtype=np.uint32)[0]
    print("Expected: ", np.dot(aArr, bArr))
    print("Got: ", c)

if __name__ == "__main__":
   testDotProd()
   # testDoublify()
