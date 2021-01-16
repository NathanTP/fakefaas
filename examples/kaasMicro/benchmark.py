import pathlib
import math
from pprint import pprint
import time
import numpy as np
import json
import argparse
import subprocess as sp
import sys
from datetime import datetime

# Just to get its exception type
import redis

import libff as ff
import libff.kv
import libff.invoke

# import kaasServer as kaas
import kaasServer

import pygemm
import pygemm.kaas

def startKaas(mode='direct', stats=None):
    """Start the kaas server and run some trivial computation to ensure the kaas server is warm."""
    libffCtx = pygemm.getCtx(remote=(mode == 'process'))
    kaasHandle = kaasServer.getHandle(mode, libffCtx, stats=stats)

    kern = kaasServer.kernelSpec(pygemm.kernsDir / 'noop.cubin', 'noop', (1,1,1), (1,1,1))
    kaasHandle.Invoke(kaasServer.kaasReq([kern]).toDict())
    kaasHandle.Stats(reset=True)

    return (libffCtx, kaasHandle)


def cleanStats(rawStats, config):
    stats = {**rawStats, **config}
    return stats


def writeStats(stats, path: pathlib.Path):
    """stats should be a flat dictionary of statistics to write to the JSON file at path"""
    if path.exists():
        with open(path, 'r') as f:
            allStats = json.load(f)
    else:
        allStats = {}

    expName = stats['name']
    if expName in allStats:
        allStats[expName].append(stats)
    else:
        allStats[expName] = [stats]

    with open(path, 'w') as f:
        json.dump(allStats, f, indent=2)


def benchmark(name, depth, size, mode, nrepeat, clientType, preprocessTime=None, outPath=None):
    """Run a benchmark, outputing a CSV named ${name}.csv with the name column
    set to name.  The benchmark will be run with the depth,size,mode and
    repeated nrpeat times + 1 (cold start + nrepeat warm starts).

    If outPath is set, the results will be appended to that CSV file instead
    of creating a new one."""

    topProfs = ff.profCollection()
    if clientType == 'kaas':
        ffCtx, kaasCtx = startKaas(mode, stats=topProfs.mod("kaas"))
        client = pygemm.kaas.benchClient('benchmark-'+ mode, depth, size, ffCtx, kaasCtx,
                preprocessTime=preprocessTime, stats=topProfs.mod('client'))
    elif clientType == 'faas':
        ffCtx = pygemm.getCtx(remote=(mode == 'process'))
        client = pygemm.faas.benchClient('benchmark-'+ mode, depth, size, ffCtx,
                preprocessTime=preprocessTime, mode=mode, useCuda=True, stats=topProfs.mod('client'))
    elif clientType == 'local':
        client = pygemm.local.benchClient('benchmark-'+ mode, depth, size,
                preprocessTime=preprocessTime, useCuda=True, stats=topProfs.mod('client'))

    configDict = { 'name' : name, 'mode' : mode, 'n_repeat' : nrepeat, 't_preprocess' : preprocessTime,
            'matDim' : size, 'depth' : depth, 's_matrix' :  pygemm.sizeFromSideLen(depth, size),
            'client' : clientType, 'date' : datetime.now().isoformat()}

    if outPath is not None:
        outPath = pathlib.Path('.').resolve() / outPath
    else:
        outPath = pathlib.Path('.').resolve() / name + ".json"

    # Cold Start
    client.invokeN(1)
    coldStats = cleanStats(topProfs.report(), configDict)
    coldStats['warm'] = False
    writeStats(coldStats, outPath)
    topProfs.reset()

    # Warm Start
    client.invokeN(nrepeat, fetchResult=True)
    warmStats = cleanStats(topProfs.report(), configDict)
    warmStats['warm'] = True
    writeStats(warmStats, outPath)
    topProfs.reset()

    # newFile = not outPath.exists()
    # with open(outPath, 'a') as csvF:
    #     writer = csv.DictWriter(csvF, fieldnames=warmStats.keys())
    #
    #     if newFile:
    #         writer.writeheader()
    #
    #     writer.writerow(warmStats)
    #     writer.writerow(coldStats)


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
    parser.add_argument("-p", "--preprocess", default=None,
            help="Preprocessing time. This may be either a number of ms of preprocessing time to simulate or 'high'/'low' to use a preconfigured time as a fraction of problem size. If not specified, no preprocessing will be simulated.")

    args = parser.parse_args()

    if args.preprocess.isdigit():
        preproc = int(args.preprocess)

    if args.size == 'small': 
        size = 1024
        if not args.preprocess.isdigit():
            # XXX ideally we'd actually calculate this somehow, but for now it's just hard-coded
            if args.preprocess == 'high':
                # roughly 2x the model runtime
                preproc = 150
            elif args.preprocess == 'low':
                # roughly 25% of the model runtime
                preproc = 20
    else:
        size = 1024*8
        if not args.preprocess.isdigit():
            if args.preprocess == 'high':
                # roughly 2x the model runtime
                preproc = 76000
            elif args.preprocess == 'low':
                # roughly 25% of the model runtime
                preproc = 9500


    # Each worker may collect different statistics and therefor must use a different CSV output
    # outPath = args.worker+".csv"
    outPath = 'results.json'

    if args.mode == 'process':
        redisProc = sp.Popen(['redis-server', '../../redis.conf'], stdout=sp.PIPE, text=True)

    try:
        benchmark("_".join([args.size, args.mode, args.worker]), 4, size, args.mode, args.niter,
                args.worker, outPath=outPath,
                preprocessTime=preproc)
    except redis.exceptions.ConnectionError as e:
        print("Redis error:")
        print(e)
        redisProc.terminate()
        print(redisProc.stdout.read())

    if args.mode == 'process':
        redisProc.terminate()
