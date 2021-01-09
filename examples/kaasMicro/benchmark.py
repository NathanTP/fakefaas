import pathlib
import math
from pprint import pprint
import time
import numpy as np
import csv
import argparse
import subprocess as sp

# Just to get its exception type
import redis

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
    # func = pygemm.faas.ChainedMults("testchain", shapes, libffCtx, mode=mode, useCuda=True)
    func = pygemm.local.ChainedMults("testchain", shapes, ffCtx=libffCtx, useCuda=True)

    retKey = func.invoke(inArr)

    testOut = pygemm.getData(libffCtx, retKey, shapes[-1].c)

    baseFunc = pygemm.local.ChainedMults("testchain_baseline", shapes, bArrs=func.bArrs, useCuda=False)
    baseArr = baseFunc.invoke(inArr)

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


def cleanStats(rawStats, config):
    # Have to flatten for CSV which could lead to accidental name conflicts
    overlappingKeys = set(rawStats['WorkerStats'].keys()) & set(rawStats['LocalStats'].keys())
    if len(overlappingKeys) != 0:
        print("WARNING: key conflict between client and worker stats:")
        print(overlappingKeys)

    stats = {**rawStats['WorkerStats'], **rawStats['LocalStats']}
    stats = {**stats, **config}

    return stats

def benchmark(name, depth, size, mode, nrepeat, clientType, outPath=None):
    """Run a benchmark, outputing a CSV named ${name}.csv with the name column
    set to name.  The benchmark will be run with the depth,size,mode and
    repeated nrpeat times + 1 (cold start + nrepeat warm starts).

    If outPath is set, the results will be appended to that CSV file instead
    of creating a new one."""

    if clientType == 'kaas':
        ffCtx, kaasCtx = startKaas(mode)
        client = pygemm.kaas.benchClient('benchmark-'+ mode, depth, size, ffCtx, kaasCtx)
    elif clientType == 'faas':
        ffCtx = pygemm.getCtx(remote=(mode == 'process'))
        client = pygemm.faas.benchClient('benchmark-'+ mode, depth, size, ffCtx, mode=mode, useCuda=True)
    elif clientType == 'local':
        client = pygemm.local.benchClient('benchmark-'+ mode, depth, size, useCuda=True)

    configDict = { 'name' : name, 'mode' : mode, 'n_repeat' : nrepeat,
            'matDim' : size, 'depth' : depth, 's_matrix' :  pygemm.sizeFromSideLen(depth, size),
            'client' : clientType}

    # Cold Start
    client.invokeN(1)
    coldStats = cleanStats(client.getStats(reset=True), configDict)
    coldStats['warm'] = False

    # Warm Start
    client.invokeN(nrepeat, fetchResult=True)
    warmStats = cleanStats(client.getStats(reset=True), configDict)
    warmStats['warm'] = True

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
    # testMMOne('direct')
    # testMMChained('direct')
    # testClient('direct')
    # benchmark('testingBench', 1, 128, 'direct', 2, outPath='test.csv')
    # sys.exit()

    #=================================================================================
    # Results are only valid if you run one benchmark. Running multiple
    # benchmarks in the same process may skew results, you probably also want
    # to restart redis for the process mode.
    #=================================================================================

    parser = argparse.ArgumentParser(description="Benchmark the gemm kernel")
    parser.add_argument("-w", "--worker", type=str, default='local', choices=['local', 'kaas', 'faas'],
            help="Which backend worker to use.")
    parser.add_argument("-s", "--size", type=str, default='small', choices=['small', 'large'],
            help="Which experiment size to use.")
    parser.add_argument("-m", "--mode", type=str, default='direct', choices=['direct', 'process'],
            help="Which libff mode to use")
    parser.add_argument("-n", "--niter", type=int, default=5, help="Number of experiment iterations to run for the warm results")

    args = parser.parse_args()

    if args.size == 'small': 
        size = 1024
    else:
        size = 1024*8

    # Each worker may collect different statistics and therefor must use a different CSV output
    outPath = args.worker+".csv"

    if args.mode == 'process':
        redisProc = sp.Popen(['redis-server', '../../redis.conf'], stdout=sp.PIPE, text=True)

    try:
        benchmark("_".join([args.size, args.mode, args.worker]), 4, size, args.mode, args.niter, args.worker, outPath=outPath)
    except redis.exceptions.ConnectionError as e:
        print("Redis error:")
        print(e)
        redisProc.terminate()
        print(redisProc.stdout.read())

    if args.mode == 'process':
        redisProc.terminate()
