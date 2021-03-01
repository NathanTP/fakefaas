import pathlib
import math
from pprint import pprint
import time
import numpy as np
import argparse
import itertools
import sys

import libff as ff
import libff.kv
import libff.invoke
import libff.kaas as kaas

import pygemm
import pygemm.kaas

def _testChainedOne(name, mode, clientType, shapes, libffCtx, kaasHandle):
    inArr = pygemm.generateArr(shapes[0].a)

    if clientType == 'kaas':
        func = pygemm.kaas.ChainedMults("testchain"+name, shapes, libffCtx, kaasHandle)
    elif clientType == 'faas':
        func = pygemm.faas.ChainedMults("testchain"+name, shapes, libffCtx, mode=mode, useCuda=True)
    else:
        func = pygemm.local.ChainedMults("testchain"+name, shapes, ffCtx=libffCtx, useCuda=True)

    baseFunc = pygemm.local.ChainedMults("testchain_baseline", shapes, bArrs=func.bArrs, useCuda=False)
    baseRes = baseFunc.invoke(inArr)

    retKey = func.invoke(inArr)
    testOut = pygemm.getData(libffCtx, retKey, shapes[-1].c)
    dist = np.linalg.norm(testOut - baseRes)
    if dist > 10:
        return "First call returned wrong result\n" + "Distance: " + str(dist)

    retKey = func.invoke(inArr)
    testOut = pygemm.getData(libffCtx, retKey, shapes[-1].c)
    dist = np.linalg.norm(testOut - baseRes)
    if dist > 10:
        return "Second call returned wrong result\n" + "Distance: " + str(dist)

    func.destroy()
    return None


def testChained(mode, clientType):
    libffCtx = pygemm.getCtx(remote=(mode == 'process'))

    shapes = [
            pygemm.mmShape(128,128,128),
            pygemm.mmShape(128,256,128),
            pygemm.mmShape(128,512,256) ]

    if clientType == 'kaas':
        kaasHandle = kaas.getHandle(mode, libffCtx)
    else:
        kaasHandle = None

    res = _testChainedOne("0", mode, clientType, shapes, libffCtx, kaasHandle)
    if res is not None:
        raise TestError("first func " + "_".join(["chained", mode, clientType]), res)

    #===============================================================
    # Create a new function handle to make sure cleanup works
    #===============================================================
    shapes = [
            pygemm.mmShape(256,256,256),
            pygemm.mmShape(256,128,256),
            pygemm.mmShape(256,256,128) ]
    res = _testChainedOne("1", mode, clientType, shapes, libffCtx, kaasHandle)
    if res is not None:
        raise TestError("second func " + "_".join(["chained", mode, clientType]), res)

    libffCtx.kv.destroy()
    print("PASS")


def testClient(mode, clientType):
    libffCtx = pygemm.getCtx(remote=(mode == 'process'))

    stats = libff.profCollection()
    if clientType == 'faas':
        client = pygemm.faas.benchClient("benchClientTest", 4, 1024, libffCtx, mode=mode, stats=stats)
    elif clientType == 'kaas':
        client = pygemm.kaas.benchClient("benchClientTest", 4, 1024, libffCtx, mode=mode, stats=stats)
    else:
        client = pygemm.local.benchClient("benchClientTest", 4, 1024, ffCtx=libffCtx, stats=stats)

    # benchclients manage most of the experiment themselves so we can't really
    # verify correctness (we trust testChained to do that). But we will make
    # sure the basic API runs without crashing.

    inArr = pygemm.generateArr(client.shapes[0].a)
    client.invoke(inArr)
    testOut = client.getResult()
    
    client.resetStats()

    start = time.time()
    client.invokeN(2)
    tInvoke = time.time() - start 

    stats = client.getStats().report()
    client.destroy()
    
    reported = stats['t_invoke']
    measured = (tInvoke / 2)*1000
    if (measured / reported) < 0.5:
        print("FAIL")
        print("Measured time significantly different than reported time: ")
        print("\tMeasured: ", measured)
        print("\tReported: ", reported)
    else:
        print("PASS")


def testThroughput(mode, clientType):
    libffCtx = pygemm.getCtx(remote=(mode == 'process'))

    stats = libff.profCollection()
    if clientType == 'faas':
        client = pygemm.faas.benchClient("benchClientTest", 4, 1024, libffCtx, mode=mode, stats=stats)
    elif clientType == 'kaas':
        client = pygemm.kaas.benchClient("benchClientTest", 4, 1024, libffCtx, mode=mode, stats=stats)
    else:
        client = pygemm.local.benchClient("benchClientTest", 4, 1024, ffCtx=libffCtx, stats=stats)

    # benchclients manage most of the experiment themselves so we can't really
    # verify correctness (we trust testChained to do that). But we will make
    # sure the basic API runs without crashing.

    inArr = pygemm.generateArr(client.shapes[0].a)
    client.invoke(inArr)
    testOut = client.getResult()
    
    client.resetStats()

    start = time.time()
    client.invokeN(2)
    tInvoke = time.time() - start 

    stats = client.getStats().report()
    client.destroy()
    
    reported = stats['t_invoke']
    measured = (tInvoke / 2)*1000
    if (measured / reported) < 0.5:
        print("FAIL")
        print("Measured time significantly different than reported time: ")
        print("\tMeasured: ", measured)
        print("\tReported: ", reported)
    else:
        print("PASS")


if __name__ == "__main__":
    clientTypes = ['kaas', 'faas', 'local']
    modes = ['direct', 'process']

    # mode = 'direct'
    # clientType = 'local'
    # benchName = "_".join(["chained", mode, clientType])
    # with testenv(benchName, mode, clientType):
    #     print(benchName)
    #     testChained(mode, clientType)
    #     print("PASS")
    #
    # benchName = "_".join(["client", mode, clientType])
    # with testenv(benchName, mode, clientType):
    #     print(benchName)
    #     testClient(mode, clientType)
    # sys.exit() 

    for (mode, clientType) in itertools.product(modes, clientTypes):
        benchName = "_".join(["chained", mode, clientType])
        with ff.testenv(benchName, mode):
            print(benchName)
            testChained(mode, clientType)

        benchName = "_".join(["client", mode, clientType])
        with ff.testenv(benchName, mode):
            print(benchName)
            testClient(mode, clientType)
