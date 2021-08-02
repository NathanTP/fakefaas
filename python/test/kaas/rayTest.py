import libff.kaas as kaas
import libff.kaas.kaasRay as kaasRay
import numpy as np
import pathlib

import ray

testPath = pathlib.Path(__file__).resolve().parent


@ray.remote(num_gpus=1)
def runKaasTask(req):
    returns = kaasRay.kaasServeRay(req)
    return returns


def testMultipleOut():
    nElem = 16

    testArray = np.arange(0, nElem, dtype=np.uint32)

    inpRef = ray.put(testArray.data)

    args = [(kaas.bufferSpec(inpRef, testArray.nbytes), 'i'),
            (kaas.bufferSpec('outIncremented', testArray.nbytes, ephemeral=False), 'o'),
            (kaas.bufferSpec('outDoubled', testArray.nbytes, ephemeral=False), 'o')]

    kern = kaas.kernelSpec(testPath / 'kerns' / 'libkaasMicro.cubin',
                           'multipleOut',
                           (1, 1), (nElem, 1, 1),
                           literals=[kaas.literalSpec('Q', nElem)],
                           arguments=args)

    req = kaas.kaasReq([kern])

    retRef = runKaasTask.remote(req.toDict())

    outRefs = ray.get(retRef)

    incremented = np.frombuffer(ray.get(outRefs[0]), dtype=np.uint32)
    doubled = np.frombuffer(ray.get(outRefs[1]), dtype=np.uint32)

    expect = testArray + 1
    if not np.array_equal(incremented, expect):
        print("Fail: increment didn't work")
        print("\tExpect: ", expect)
        print("\tGot: ", incremented)
        # return False

    expect = testArray * 2
    if not np.array_equal(doubled, expect):
        print("Fail: double didn't work")
        print("\tExpect: ", expect)
        print("\tGot: ", doubled)
        return False

    print("PASS")
    return True


def testSum():
    testArray = np.arange(0, 16, dtype=np.uint32)
    origArray = testArray.copy()

    inpRef = ray.put(testArray.data)

    args = [(kaas.bufferSpec(inpRef, testArray.nbytes), 'i'),
            (kaas.bufferSpec('out', 4, ephemeral=False), 'o')]

    kern = kaas.kernelSpec(testPath / 'kerns' / 'libkaasMicro.cubin',
                           'sumKern',
                           (1, 1), (8, 1, 1),
                           arguments=args)

    req = kaas.kaasReq([kern])

    retRef = runKaasTask.remote(req.toDict())

    outRef = ray.get(retRef)
    kaasSum = np.frombuffer(ray.get(outRef), dtype=np.uint32)[0]

    expect = origArray.sum()
    if kaasSum != expect:
        print("Fail: results don't match")
        print("\tExpect: ", expect)
        print("\tGot: ", kaasSum)
    else:
        print("PASS")


ray.init()
print("Sum Test:")
testSum()
print("Multiple Output Test:")
testMultipleOut()
