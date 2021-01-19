import pathlib
import math
from pprint import pprint
import time
import numpy as np
import csv
import argparse
import subprocess as sp
import itertools
from contextlib import contextmanager
import sys

# Just to get its exception type
import redis

import libff as ff
import libff.kv
import libff.invoke

# import kaasServer as kaas
import kaasServer

import pygemm
import pygemm.kaas

class TestError(Exception):
    def __init__(self, testName, msg):
        self.testName = testName
        self.msg = msg

    def __str__(self):
        return "Test {} failed: {}".format(self.testName, self.msg)

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
        kaasHandle = kaasServer.getHandle(mode, libffCtx)
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
        kaasHandle = kaasServer.getHandle(mode, libffCtx)
        client = pygemm.kaas.benchClient("benchClientTest", 4, 1024, libffCtx, kaasHandle, stats=stats)
    else:
        client = pygemm.local.benchClient("benchClientTest", 4, 1024, ffCtx=libffCtx, stats=stats)

    # benchclients manage most of the experiment themselves so we can't really
    # verify correctness (we trust testChained to do that). But we will make
    # sure the basic API runs without crashing.

    inArr = pygemm.generateArr(client.shapes[0].a)
    client.invoke(inArr)
    testOut = client.getResult()
    
    stats.reset()

    start = time.time()
    client.invokeN(2)
    tInvoke = time.time() - start 

    stats = client.getStats()
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

    # s = kaasHandle.Stats()
    #
    # timeMetricsTotal = 0
    # for k,v in s['WorkerStats'].items():
    #     if k[:2] == 't_':
    #         timeMetricsTotal += v
    # print("Total Time: ", s['LocalStats']['t_libff_invoke'])
    # print("Time Metrics Total: ", timeMetricsTotal)
    # print("Missing Time: ", s['LocalStats']['t_libff_invoke'] - timeMetricsTotal)
    # pprint(s)

    # clientPoisson = benchClient("benchClientTest", 3, 128, libffCtx, kaasHandle, rng=benchClient.poisson(5))
    # clientPoisson.invokeN(5, inArrs=5)
    # clientPoisson.destroy()
    #
    # clientZipf = benchClient("benchClientTest", 3, 128, libffCtx, kaasHandle, rng=benchClient.zipf(2, maximum=100))
    # clientZipf.invokeN(5, inArrs = [ generateArr(clientZipf.shapes[0].a) for i in range(5) ])
    # clientZipf.destroy()

@contextmanager
def testenv(testName, mode, clientType):
    if mode == 'process':
        redisProc = sp.Popen(['redis-server', '../../redis.conf'], stdout=sp.PIPE, text=True)

    try:
        yield
    except redis.exceptions.ConnectionError as e:
        pass
        redisProc.terminate()
        serverOut = redisProc.stdout.read()
        raise TestError(testName, str(e) + ": " + serverOut)

    if mode == 'process':
        redisProc.terminate()


if __name__ == "__main__":
    clientTypes = ['kaas', 'faas', 'local']
    modes = ['direct', 'process']

    # mode = 'process'
    # clientType = 'faas'
    # benchName = "_".join(["chained", mode, clientType])
    # with testenv(benchName, mode, clientType):
    #     print(benchName)
    #     testChained(mode, clientType)
    #     print("PASS")
    #     time.sleep(0.5)

    # benchName = "_".join(["client", mode, clientType])
    # with testenv(benchName, mode, clientType):
    #     print(benchName)
    #     testClient(mode, clientType)
    #     # testChained(mode, clientType)
    # sys.exit() 

    for (mode, clientType) in itertools.product(modes, clientTypes):
        benchName = "_".join(["chained", mode, clientType])
        with testenv(benchName, mode, clientType):
            print(benchName)
            testChained(mode, clientType)
            # The OS takes a little bit to clean up the port reservation, gotta
            # wait before restarting redis
            time.sleep(0.5)

        benchName = "_".join(["client", mode, clientType])
        with testenv(benchName, mode, clientType):
            print(benchName)
            testClient(mode, clientType)
            time.sleep(0.5)
