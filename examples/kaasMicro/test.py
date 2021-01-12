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

# Just to get its exception type
import redis

import libff as ff
import libff.kv
import libff.invoke

# import kaasServer as kaas
import kaasServer

import pygemm
import pygemm.kaas

def testChained(mode, clientType):
    libffCtx = pygemm.getCtx(remote=(mode == 'process'))
    kaasHandle = kaasServer.getHandle(mode, libffCtx)

    shapes = [
            pygemm.mmShape(128,128,128),
            pygemm.mmShape(128,256,128),
            pygemm.mmShape(128,512,256) ]

    inArr = pygemm.generateArr(shapes[0].a)

    if clientType == 'kaas':
        func = pygemm.kaas.ChainedMults("testchain", shapes, libffCtx, kaasHandle)
    elif clientType == 'faas':
        func = pygemm.faas.ChainedMults("testchain", shapes, libffCtx, mode=mode, useCuda=True)
    else:
        func = pygemm.local.ChainedMults("testchain", shapes, ffCtx=libffCtx, useCuda=True)

    retKey = func.invoke(inArr)

    testOut = pygemm.getData(libffCtx, retKey, shapes[-1].c)

    baseFunc = pygemm.local.ChainedMults("testchain_baseline", shapes, bArrs=func.bArrs, useCuda=False)
    baseArr = baseFunc.invoke(inArr)

    func.destroy()
    libffCtx.kv.destroy()

    diff = testOut - baseArr
    dist = np.linalg.norm(diff)

    if dist > 10:
        print("FAIL")
        print("Distance: " + str(dist))
    else:
        print("PASS")


def testClient(mode, clientType):
    libffCtx = pygemm.getCtx(remote=(mode == 'process'))
    kaasHandle = kaasServer.getHandle(mode, libffCtx)

    if clientType == 'faas':
        clientPlain = pygemm.faas.benchClient("benchClientTest", 4, 1024, libffCtx, mode)
    elif clientType == 'kaas':
        clientPlain = pygemm.kaas.benchClient("benchClientTest", 4, 1024, libffCtx, kaasHandle)
    else:
        clientPlain = pygemm.local.benchClient("benchClientTest", 4, 1024, libffCtx)

    start = time.time()
    clientPlain.invokeN(2)
    tInvoke = time.time() - start 

    stats = clientPlain.getStats()
    clientPlain.destroy()
    
    reported = stats['LocalStats']['t_client_invoke']
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

class TestError(Exception):
    def __init__(self, testName, msg):
        self.testName = testName
        self.msg = msg

    def __str__(self):
        return "Test {} failed: {}".format(testName, msg)


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

    # mode = 'direct'
    # clientType = 'kaas'
    # benchName = "_".join(["client", mode, clientType])
    # with testenv(benchName, mode, clientType):
    #     print(benchName)
    #     testClient(mode, clientType)
        # time.sleep(0.5)

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
