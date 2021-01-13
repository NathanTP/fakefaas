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

# def benchmarkWithPreprocess(name, depth, size, mode, nrepeat, clientType, outPath=None):
#     """Run a host-based preprocessing phase before running the kernel(s) to
#     simulate a decoupled application"""
#     pass
#
def benchmark(name, depth, size, mode, nrepeat, clientType, preprocessTime=None, outPath=None):
    """Run a benchmark, outputing a CSV named ${name}.csv with the name column
    set to name.  The benchmark will be run with the depth,size,mode and
    repeated nrpeat times + 1 (cold start + nrepeat warm starts).

    If outPath is set, the results will be appended to that CSV file instead
    of creating a new one."""

    if clientType == 'kaas':
        ffCtx, kaasCtx = startKaas(mode)
        client = pygemm.kaas.benchClient('benchmark-'+ mode, depth, size, ffCtx, kaasCtx,
                preprocessTime=preprocessTime)
    elif clientType == 'faas':
        ffCtx = pygemm.getCtx(remote=(mode == 'process'))
        client = pygemm.faas.benchClient('benchmark-'+ mode, depth, size, ffCtx,
                preprocessTime=preprocessTime, mode=mode, useCuda=True)
    elif clientType == 'local':
        client = pygemm.local.benchClient('benchmark-'+ mode, depth, size,
                preprocessTime=preprocessTime, useCuda=True)

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
    parser.add_argument("-p", "--preprocess", type=int, default=None,
            help="Number of ms of preprocessing time to simulate. If not specified, no preprocessing will be simulated.")

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
        benchmark("_".join([args.size, args.mode, args.worker]), 4, size, args.mode, args.niter,
                args.worker, outPath=outPath,
                preprocessTime=args.preprocess)
    except redis.exceptions.ConnectionError as e:
        print("Redis error:")
        print(e)
        redisProc.terminate()
        print(redisProc.stdout.read())

    if args.mode == 'process':
        redisProc.terminate()
