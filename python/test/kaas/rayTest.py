import libff.kaas as kaas
import libff.kaas.kaasRay as kaasRay
import numpy as np
import pathlib

import ray

testPath = pathlib.Path(__file__).resolve().parent


def testDoublify():
    ray.init()

    testArray = np.random.randn(16).astype(np.float32)
    origArray = testArray.copy()

    inpRef = ray.put(testArray.data)

    # doublify is in-place
    # XXX kaas currently conflates GPU buffers and KV objects. This means you
    # can't have an in-place kernel that writes the output to a new key. Gotta
    # fix this, but this works around it for now since Ray ignores the output
    # key but KaaS uses it to identify the buffer.
    arguments = [(kaas.bufferSpec(inpRef, testArray.nbytes), 'io')]

    kern = kaas.kernelSpec(testPath / 'kerns' / 'libkaasMicro.cubin',
                           'doublifyKern',
                           (1, 1), (16, 1, 1),
                           arguments=arguments)

    req = kaas.kaasReq([kern])

    # This is just for the test, a real system would use libff to invoke the
    # kaas server
    outs = kaasRay.kaasServeRay.remote(req.toDict())
    outs = ray.get(outs)
    assert len(outs) == 1

    doubledBytes = ray.get(outs[0])
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


testDoublify()
