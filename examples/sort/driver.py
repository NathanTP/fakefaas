import pylibsort
import numpy as np
import tempfile
import pathlib
import libff.invoke

def distribSort(inputName):
   pass

def testPartialDirect(nElem):
    """Basic sanity test: invoke worker directly (import it) and do one round
    of sorting with one client"""
    import worker

    offset = 0
    width = 8
    inSz = nElem*4
    inBuf = pylibsort.generateInputs(nElem)

    with tempfile.TemporaryDirectory() as tDir:
        tDir = pathlib.Path(tDir)
        pylibsort.SetDistribMount(tDir)

        inArrName = "distribSortTestInput"
        outArrName = "distribSortTestOutput"
        inShape = pylibsort.ArrayShape.fromUniform(inSz, 1)

        inArr = pylibsort.DistribArray.Create(inArrName, inShape)
        inArr.WriteAll(inBuf)
        inArr.Close()

        ref = { "arrayName" : inArrName,
                "partID" : 0,
                "start" : 0,
                "nbyte" : -1 }
        

        req = { "offset" : offset,
                "width" : width,
                "arrType" : "file",
                "input" : [ref],
                "output" : outArrName }

        worker.sortPartial(req)

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


def testPartialNoLibff(nElem):
    """Basic sanity test: invoke worker directly (import it) and do one round
    of sorting with one client"""
    import worker

    offset = 0
    width = 8
    inSz = nElem*4
    inBuf = pylibsort.generateInputs(nElem)

    with tempfile.TemporaryDirectory() as tDir:
        tDir = pathlib.Path(tDir)
        pylibsort.SetDistribMount(tDir)

        inArrName = "distribSortTestInput"
        outArrName = "distribSortTestOutput"
        inShape = pylibsort.ArrayShape.fromUniform(inSz, 1)

        inArr = pylibsort.DistribArray.Create(inArrName, inShape)
        inArr.WriteAll(inBuf)
        inArr.Close()

        ref = { "arrayName" : inArrName,
                "partID" : 0,
                "start" : 0,
                "nbyte" : -1 }
        

        req = { "offset" : offset,
                "width" : width,
                "arrType" : "file",
                "input" : [ref],
                "output" : outArrName }

        worker.sortPartial(req)

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


def testPartialRemote(funcClass, nElem):
    """Basic sanity test: invoke worker directly (import it) and do one round
    of sorting with one client"""

    offset = 0
    width = 8
    inSz = nElem*4
    inBuf = pylibsort.generateInputs(nElem)

    with tempfile.TemporaryDirectory() as tDir:
        tDir = pathlib.Path(tDir)
        pylibsort.SetDistribMount(tDir)
        remFunc = funcClass(pathlib.Path("./worker.py").resolve(), "sortPartial", tDir)

        inArrName = "distribSortTestInput"
        outArrName = "distribSortTestOutput"
        inShape = pylibsort.ArrayShape.fromUniform(inSz, 1)

        inArr = pylibsort.DistribArray.Create(inArrName, inShape)
        inArr.WriteAll(inBuf)
        inArr.Close()

        ref = { "arrayName" : inArrName,
                "partID" : 0,
                "start" : 0,
                "nbyte" : -1 }
        

        req = { "offset" : offset,
                "width" : width,
                "arrType" : "file",
                "input" : [ref],
                "output" : outArrName }

        remFunc.Invoke(req)

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


if __name__ == "__main__":
    # testPartialDirectNoLibff(1024)

    # remFunc = libff.invoke.ProcessRemoteFunc(pathlib.Path("./worker.py").resolve(), "sortPartial")
    funcClass = libff.invoke.ProcessRemoteFunc
    testPartialRemote(funcClass, 1024)
