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


def testDirect():
    libffCtx = getCtx(remote=False)

    testArray = np.random.randn(4,4).astype(np.float32)
    print("Orig:")
    print(testArray)

    testBytes = bytearray(testArray)
    libffCtx.kv.put("input", testBytes)

    # doublify is in-place
    inputs  = [ kaas.bufferSpec('input', len(testBytes)) ]
    outputs = [ kaas.bufferSpec('input', len(testBytes)) ]

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


if __name__ == "__main__":
   testDirect() 
