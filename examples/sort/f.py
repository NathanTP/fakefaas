import subprocess as sp
import sys
import ctypes
import os
import random
import json
import base64
import pathlib
import tempfile
import functools
import operator
import time
import numpy
import pylibsort
import libff.array as farr

# ol-install: numpy

# from memory_profiler import profile
import cProfile
import pstats
import io

doProfile = True 

def printCSV(pr, path):
    path = pathlib.Path(path)

    result = io.StringIO()
    # pstats.Stats(pr,stream=result).print_stats()
    pstats.Stats(pr,stream=result).sort_stats(pstats.SortKey.CUMULATIVE).print_stats()
    result=result.getvalue()
    # chop the string into a csv-like buffer
    result='ncalls'+result.split('ncalls')[-1]
    result='\n'.join([','.join(line.rstrip().split(None,5)) for line in result.split('\n')])
    
    with open(path, 'w') as f:
        f.write(result)


def f(event):
    # Temporary limitation for testing
    if event['arrType'] != 'file':
        return {
                "success" : False,
                "err" : "Function currently only supports file distributed arrays"
                }

    refs = pylibsort.getPartRefs(event)
    rawBytes = farr.readPartRefs(refs)

    try:
        boundaries = pylibsort.sortPartial(rawBytes, event['offset'], event['width'])
    except Exception as e:
        return {
                "success" : False,
                "err" : str(e)
               }

    pylibsort.writeOutput(event, rawBytes, boundaries)
    
    return {
            "success" : True,
            "err" : "" 
           }

def selfTest():
    """Main only used for testing purposes"""

    import numpy as np

    # nElem = 256 
    # nElem = 256*(1024*1024)
    nElem = 1024*1024
    nbyte = nElem*4
    offset = 0
    width = 8
    # width = 16
    narr = 2
    npart = 2
    bytesPerPart = int(nbyte / (narr * npart))

    inBuf = pylibsort.generateInputs(nElem)

    with tempfile.TemporaryDirectory() as tDir:
        tDir = pathlib.Path(tDir)
        farr.SetFileMount(tDir)

        inArrName = "faasSortTestIn"
        outArrName = "faasSortTestOut"

        # Write source arrays
        inShape = farr.ArrayShape.fromUniform(bytesPerPart, npart)
        refs = []
        for arrX in range(narr):
            arrName = inArrName + str(arrX)
            inArr = farr.fileDistribArray.Create(arrName, inShape)

            start = (arrX*npart)*bytesPerPart
            end = start + (bytesPerPart*npart)
            inArr.WriteAll(inBuf[start:end])
            for partX in range(npart):
                refs.append({
                    'arrayName': arrName,
                    'partID' : partX,
                    'start' : 0,
                    'nbyte' : -1
                })
            inArr.Close()

        req = {
                "offset" : offset,
                "width" : width,
                "arrType" : "file",
                "input" : refs,
                "output" : outArrName
        }

        if doProfile:
            pr = cProfile.Profile()
            pr.enable()
            resp = f(req)
            pr.disable()
            printCSV(pr, "./faas{}b.csv".format(width))
            pr.dump_stats("./faas{}b.prof".format(width))
        else:
            start = time.time()
            resp = f(req)
            print(time.time() - start)

        if not resp['success']:
            print("FAILURE: Function returned error: " + resp['err'])
            exit(1)

        outArr = farr.fileDistribArray.Open(outArrName)
        outBuf = outArr.ReadAll()
        boundaries = outArr.shape.starts

        outArr.Destroy()

    # byte capacities to int boundaries 
    caps = np.array(outArr.shape.caps) // 4
    boundaries = np.cumsum(caps)
    boundaries = np.roll(boundaries, 1)
    boundaries[0] = 0

    pylibsort.checkPartial(inBuf, outBuf, boundaries, offset, width)

    print("PASS")


def directInvoke():
    """Call this to directly invoke the function from the command line instead
    of through a FaaS provider, it expects the arguments to be in JSON as
    argv[1]. This function behaves as a main(); it calls exit() directly and
    never returns."""

    dataDir = os.environ.get('OL_SHARED_VOLUME', "")
    if dataDir == "":
        print(json.dumps({ "success" : False, "err" : "OL_SHARED_VOLUME not set, set it to the shared directory for distrib arrays"}))
        exit(1)
    farr.SetFileMount(pathlib.Path(dataDir))

    
    try:
        jsonCmd = sys.stdin.read()
        cmd = json.loads(jsonCmd)
    except Exception as e:
        print(json.dumps({ "success" : False, "err" : "Argument parsing error: " + str(e) }))
        exit(1)

    resp = None
    if doProfile:
        pr = cProfile.Profile()
        pr.enable()
        resp = f(cmd)
        pr.disable()
        printCSV(pr, "./faas{}.csv".format(cmd['output']))
        pr.dump_stats("./faas{}.prof".format(cmd['output']))
    else:
        resp = f(cmd)

    print(json.dumps(resp))
    if resp['success']:
        exit(0)
    else:
        exit(1)
 

# @profile
def testGenerate():
    # sz = 256*1024*1024
    sz = 1024*1024
    testIn = pylibsort.generateInputs(sz)

    start = time.time()
    ints = pylibsort.bytesToInts(testIn)
    print(time.time() - start)
    # with tempfile.TemporaryFile() as f:
    #     f.write(testIn[:])


if __name__ == "__main__":
    selfTest()
    # directInvoke()
    # testGenerate()
