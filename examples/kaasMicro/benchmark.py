import pathlib
import math
from pprint import pprint
import time
import numpy as np
import csv

import libff as ff
import libff.kv
import libff.invoke

# import kaasServer as kaas
import kaasServer

import pygemm
import pygemm.kaas

def testMMChained(mode='direct'):
    libffCtx = pygemm.getCtx(remote=(mode == 'process'))
    kaasHandle = kaasServer.getHandle(mode, libffCtx)

    shapes = [
            pygemm.mmShape(128,128,128),
            pygemm.mmShape(128,256,128),
            pygemm.mmShape(128,512,256) ]

    inArr = pygemm.generateArr(shapes[0].a)

    # func = pygemm.kaas.ChainedMults("testchain", shapes, libffCtx, kaasHandle)
    func = pygemm.faas.ChainedMults("testchain", shapes, libffCtx, mode=mode, useCuda=True)

    retKey = func.invoke(inArr)

    testOut = pygemm.getData(libffCtx, retKey, shapes[-1].c)
    print(testOut)

    baseFunc = pygemm.local.ChainedMults(shapes, bArrs=func.bArrs, useCuda=True)
    baseArr = baseFunc.invoke(inArr)
    print(baseArr)

    func.destroy()

    diff = testOut - baseArr
    dist = np.linalg.norm(diff)

    if dist > 10:
        print("FAIL")
        print("Distance: " + str(dist))
    else:
        print("PASS")


def testMMOne(mode='direct'):
    libffCtx = pygemm.getCtx(remote=(mode == 'process'))
    kaasHandle = kaasServer.getHandle(mode, libffCtx)

    # arrA = generateArr((32,32))
    # arrB = generateArr((32,32))
    arrA = pygemm.generateArr((1024,256))
    arrB = pygemm.generateArr((256,512))
    libffCtx.kv.put("test_a", arrA)
    libffCtx.kv.put("test_b", arrB)

    shape = [arrA.shape, arrB.shape]
    func = pygemm.kaas.mmFunc('test', shape, libffCtx, kaasHandle)

    rKey = func.invoke()
    
    arrC = pygemm.getData(libffCtx, rKey, func.cShape)

    libffCtx.kv.delete('test_a')
    libffCtx.kv.delete('test_b')
    libffCtx.kv.delete('test_c')
    func.destroy()

    arrC = arrC.round(4)
    npC = np.matmul(arrA, arrB).round(4)

    # Single precision accumulates errors really fast so the differences can be
    # big. The euclidean distance seems to get us close enough for a pretty
    # good guess at correctness
    diff = arrC - npC
    dist = np.linalg.norm(diff)
    if dist > 1:
        print("FAIL:")
        print("Euclidean Dist: " + str(dist))
    else:
        print("PASS")


def testClient(mode='direct'):
    libffCtx = pygemm.getCtx(remote=(mode == 'process'))
    kaasHandle = kaasServer.getHandle(mode, libffCtx)
    # clientPlain = benchClient("benchClientTest", 3, 128, libffCtx, kaasHandle)
    clientPlain = pygemm.benchClient("benchClientTest", 4, 1024, libffCtx, kaasHandle)
    clientPlain.invokeN(1)
    clientPlain.destroy()
    
    print("Stats: ")
    s = kaasHandle.Stats()

    timeMetricsTotal = 0
    for k,v in s['WorkerStats'].items():
        if k[:2] == 't_':
            timeMetricsTotal += v
    print("Total Time: ", s['LocalStats']['invoke'])
    print("Time Metrics Total: ", timeMetricsTotal)
    print("Missing Time: ", s['LocalStats']['invoke'] - timeMetricsTotal)
    pprint(s)

    # clientPoisson = benchClient("benchClientTest", 3, 128, libffCtx, kaasHandle, rng=benchClient.poisson(5))
    # clientPoisson.invokeN(5, inArrs=5)
    # clientPoisson.destroy()
    #
    # clientZipf = benchClient("benchClientTest", 3, 128, libffCtx, kaasHandle, rng=benchClient.zipf(2, maximum=100))
    # clientZipf.invokeN(5, inArrs = [ generateArr(clientZipf.shapes[0].a) for i in range(5) ])
    # clientZipf.destroy()


def startKaas(mode='direct'):
    """Start the kaas server and run some trivial computation to ensure the kaas server is warm."""
    libffCtx = pygemm.getCtx(remote=(mode == 'process'))
    kaasHandle = kaasServer.getHandle(mode, libffCtx)

    kern = kaasServer.kernelSpec(pygemm.kernsDir / 'noop.cubin', 'noop', (1,1,1), (1,1,1))
    kaasHandle.Invoke(kaasServer.kaasReq([kern]).toDict())
    kaasHandle.Stats(reset=True)

    return (libffCtx, kaasHandle)


def benchmark(name, depth, size, mode, nrepeat, outPath=None):
    """Run a benchmark, outputing a CSV named ${name}.csv with the name column
    set to name.  The benchmark will be run with the depth,size,mode and
    repeated nrpeat times + 1 (cold start + nrepeat warm starts).

    If outPath is set, the results will be appended to that CSV file instead
    of creating a new one."""

    clientType = 'faas'
    if clientType == 'kaas':
        ffCtx, kaasCtx = startKaas(mode)
        client = pygemm.kaas.benchClient('benchmark-'+ mode, depth, size, ffCtx, kaasCtx)
    elif clientType == 'faas':
        ffCtx = pygemm.getCtx(remote=(mode == 'process'))
        client = pygemm.faas.benchClient('benchmark-'+ mode, depth, size, ffCtx, mode=mode, useCuda=True)
    elif clientType == 'local':
        client = pygemm.local.benchClient('benchmark-'+ mode, depth, size, useCuda=False)

    configDict = { 'name' : name, 'mode' : mode, 'nrepeat' : nrepeat,
            'matDim' : size, 'depth' : depth, 'nbyte' :  pygemm.sizeFromSideLen(depth, size)}

    # Cold Start
    client.invokeN(1)
    # rawStats = kaasCtx.Stats(reset=True)
    rawStats = client.getStats(reset=True)
    coldStats = rawStats['WorkerStats']
    coldStats['t_e2e'] = rawStats['LocalStats']['invoke']
    coldStats['warm'] = False 
    coldStats = {**coldStats, **configDict}

    # Warm Start
    client.invokeN(nrepeat)
    # rawStats = kaasCtx.Stats(reset=True)
    rawStats = client.getStats(reset=True)
    warmStats = rawStats['WorkerStats']
    warmStats['warm'] = True
    warmStats['t_e2e'] = rawStats['LocalStats']['invoke']
    warmStats = {**warmStats, **configDict}

    if outPath is not None:
        outPath = pathlib.Path('.').resolve() / outPath
    else:
        outPath = pathlib.Path('.').resolve() / name + ".csv"

    newFile = not outPath.exists()
    with open(outPath, 'a') as csvF:
        writer = csv.DictWriter(csvF, fieldnames=warmStats.keys())

        if newFile:
            writer.writeheader()

        writer.writerow(warmStats)
        writer.writerow(coldStats)

if __name__ == "__main__":
    # print(benchClient.sizeFromSideLen(3, 1024*8) / (1024*1024*1024))
    # testMMOne('direct')
    # testMMChained('direct')
    # testClient('direct')
    # benchmark('testingBench', 1, 128, 'direct', 2, outPath='test.csv')

    # benchmark('smallDirect', 4, 1024,   'direct', 5, outPath='matmul.csv')
    # benchmark('largeDirect', 4, 1024*8, 'direct', 5, outPath='matmul.csv')

    benchmark('smallRemote', 4, 1024,   'process', 5, outPath='matmul.csv')
    # benchmark('largeRemote', 4, 1024*8, 'process', 5, outPath='matmul.csv')
