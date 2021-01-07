import pathlib

import libff as ff
from libff import kv, invoke

from .util import *

workerPath = pathlib.Path(__file__).parent.parent.resolve() / "faasWorker.py"

class ChainedMults():
    def __init__(self, name, shapes, ffCtx, mode='direct', bArrs=None, useCuda=True):
        self.shapes = shapes
        self.useCuda = useCuda
        self.ffCtx = ffCtx
        self.name = name

        # List of keys we are responsible for destroying
        self.ownedKeys = []

        if mode == 'direct':
            self.remFunc = ff.invoke.DirectRemoteFunc(workerPath, 'sgemm', self.ffCtx)
        else:
            self.remFunc = ff.invoke.ProcessRemoteFunc(workerPath, 'sgemm', self.ffCtx)

        self.bArrs = []
        if bArrs is None:
            # prevShape is used to validate the chain dimensions, it's the C shape
            # of the previous layer. For the first layer, we just set it to A's
            # shape.
            prevShape = self.shapes[0].a
            for i,shape in enumerate(shapes):
                if shape.a != prevShape:
                    raise RuntimeError("Invalid input shape for layer " + str(i) + " (" + str(shape.a) + ") previous layer output " + str(prevShape))
                
                constB = generateArr(shape.b)
                self.bArrs.append(constB)
                prevShape = shape.c

        else:
            self.bArrs = bArrs

        self.bNames = [ name + "_b" + str(i) for i in range(len(self.bArrs)) ]
        for name,arr in zip(self.bNames, self.bArrs):
            self.ffCtx.kv.put(name, arr)
            self.ownedKeys += self.bNames


    def invoke(self, inArr, outBuf=None, times=None):
        cleanInput = False
        if not isinstance(inArr, str):
            inKey = self.name + "_input"
            self.ffCtx.kv.put(self.name + "_input", inArr)
            cleanInput = True
        else:
            inKey = inArr

        outKey = self.name+"_output"

        req = { "input" : inKey,
                "bArrs" : self.bNames,
                "shapes" : [ (s.M, s.N, s.K) for s in self.shapes ],
                "output" : outKey,
                "useCuda" : self.useCuda
              }

        self.remFunc.Invoke(req)

        if cleanInput:
            self.ffCtx.kv.delete(inKey)

        return outKey


    def destroy(self):
        for key in self.ownedKeys:
            self.ffCtx.kv.delete(key)


class benchClient():
    def __init__(self, name, depth, sideLen, ffCtx, mode='direct', rng=None, useCuda=True):
        self.name = name
        self.rng = rng
        self.nbytes = sizeFromSideLen(depth, sideLen)
        self.stats = ff.profCollection()
        self.ffCtx = ffCtx

        # Keys that we are responsible for deleting
        self.ownedKeys = []

        if self.nbytes > DEVICE_MEM_CAP:
            raise RuntimeError("Requested configuration would not fit on the device!")

        # Uniform shape for now
        self.shapes = [ mmShape(sideLen, sideLen, sideLen) ] * depth

        self.func = ChainedMults(name, self.shapes, ffCtx, mode, useCuda=useCuda)


    def invoke(self, inArr):
        if self.rng is not None:
            time.sleep(self.rng() / 1000)

        with ff.timer("invoke", self.stats):
            self.lastRetKey = self.func.invoke(inArr)


    def invokeN(self, n, inArrs=1, fetchResult=False):
        if isinstance(inArrs, int):
            inBufs = []
            for i in range(inArrs):
                arr = generateArr(self.shapes[0].a)
                inBufs.append(arr)
        else:
            inBufs = inArrs

        deleteInputs = False
        if not isinstance(inBufs[0], str):
            inNames = []
            for i,b in enumerate(inBufs):
                inName = self.name + "_input" + str(i)
                self.ffCtx.kv.put(inName, b)
                inNames.append(inName)
            inBufs = inNames
            deleteInputs = True

        for i in range(n):
            self.invoke(inBufs[ i % len(inBufs) ])
            if fetchResult:
                self.getResult()

            self.ffCtx.kv.delete(self.lastRetKey)

        if deleteInputs:
            for key in inBufs:
                self.ffCtx.kv.delete(key)


    def getStats(self, reset=False):
        return {
                "LocalStats" : self.stats.report(),
                "WorkerStats" : {}
        }


    def getResult(self):
        return getData(self.lastRetKey, shapes[-1].c)


    def destroy(self):
        self.func.destroy()
