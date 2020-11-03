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
import libff.invoke

# ol-install: numpy

# from memory_profiler import profile
import cProfile
import pstats
import io

doProfile = False 

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


def sortPartial(event):
    # Temporary limitation for testing
    if event['arrType'] != 'file':
        return { "error" : "Function currently only supports file distributed arrays" }

    refs = pylibsort.getPartRefs(event)
    rawBytes = pylibsort.readPartRefs(refs)

    try:
        boundaries = pylibsort.sortPartial(rawBytes, event['offset'], event['width'])
    except Exception as e:
        return { "error" : str(e) }

    pylibsort.writeOutput(event, rawBytes, boundaries)
    
    return { "error" : None }

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
        pylibsort.SetDistribMount(tDir)

        inArrName = "faasSortTestIn"
        outArrName = "faasSortTestOut"

        # Write source arrays
        inShape = pylibsort.ArrayShape.fromUniform(bytesPerPart, npart)
        refs = []
        for arrX in range(narr):
            arrName = inArrName + str(arrX)
            inArr = pylibsort.DistribArray.Create(arrName, inShape)

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
            resp = sortPartial(req)
            pr.disable()
            printCSV(pr, "./faas{}b.csv".format(width))
            pr.dump_stats("./faas{}b.prof".format(width))
        else:
            start = time.time()
            resp = sortPartial(req)
            print(time.time() - start)

        if resp['error'] is not None:
            print("FAILURE: Function returned error: " + resp['error'])
            exit(1)

        outArr = pylibsort.DistribArray.Open(outArrName)
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


def directInvoke(arrMnt):
    """Call this to directly invoke the function from the command line instead
    of through a FaaS provider, it expects the arguments to be in JSON, they
    are read from stdin. Function returns when stdin is closed."""

    pylibsort.SetDistribMount(arrMnt)

    for rawCmd in sys.stdin:
        try:
            cmd = json.loads(rawCmd)
        except json.decoder.JSONDecodeError as e:
            err = "Failed to parse command (must be valid JSON): " + str(e)
            print(json.dumps({ "error" : err }), flush=True)
            continue

        resp = None
        if doProfile:
            pr = cProfile.Profile()
            pr.enable()
            resp = sortPartial(cmd)
            pr.disable()
            printCSV(pr, "./faas{}.csv".format(cmd['output']))
            pr.dump_stats("./faas{}.prof".format(cmd['output']))
        else:
            resp = sortPartial(cmd)

        print(json.dumps(resp), flush=True)


def libffInvoke():
    """Use this as main when you are using libff.invoke to invoke the sort.
    See libff.invoke documentation for how this works"""

    libff.invoke.RemoteProcessServer({"sortPartial" : sortPartial}, sys.argv[1:])


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
    # selfTest()
    libffInvoke()
    # directInvoke()
    # testGenerate()
