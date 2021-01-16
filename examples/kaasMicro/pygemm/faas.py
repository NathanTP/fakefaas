import pathlib

import libff as ff
from libff import kv, invoke

from .util import *

workerPath = pathlib.Path(__file__).parent.parent.resolve() / "faasWorker.py"

class ChainedMults():
    def __init__(self, name, shapes, ffCtx, mode='direct', preprocessTime=None, bArrs=None, useCuda=True, stats=None):
        self.shapes = shapes
        self.useCuda = useCuda
        self.ffCtx = ffCtx
        self.name = name
        self.preprocessTime = preprocessTime
        self.stats = stats
        self.kvStats = self.stats.mod('kv')

        # List of keys we are responsible for destroying
        self.ownedKeys = []

        if mode == 'direct':
            self.remFunc = ff.invoke.DirectRemoteFunc(workerPath, 'sgemm', self.ffCtx,
                    stats=self.stats.mod('remfunc'))
        else:
            self.remFunc = ff.invoke.ProcessRemoteFunc(workerPath, 'sgemm', self.ffCtx,
                    stats=self.stats.mod('remfunc'))

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
            self.ffCtx.kv.put(name, arr, profile=self.kvStats)
            self.ownedKeys += self.bNames


    def invoke(self, inArr, outBuf=None):
        cleanInput = False
        if not isinstance(inArr, str):
            inKey = self.name + "_input"
            with ff.timer("t_write_input", self.stats):
                self.ffCtx.kv.put(self.name + "_input", inArr, profile=self.kvStats)
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

        if self.preprocessTime is not None:
            req['preprocess'] = self.preprocessTime

        with ff.timer("t_invoke", self.stats):
            self.remFunc.Invoke(req)

        if cleanInput:
            self.ffCtx.kv.delete(inKey, profile=self.kvStats)

        return outKey


    def destroy(self):
        for key in self.ownedKeys:
            self.ffCtx.kv.delete(key, profile=self.kvStats)
        self.remFunc.Close()


class benchClient():
    def __init__(self, name, depth, sideLen, ffCtx, preprocessTime=None, mode='direct', rng=None, useCuda=True, stats=None):
        """A general driver for the sgemm benchmark.
            - preprocessTime: amount of time to spend preprocessing on a
              host-only function, None skips the preprocess step entirely.

        METRICS: benchClient sets the following metrics in stats
            - t_e2e: The total time for one invocation of the prediction,
              including any data movement, pre/post processing, and the model itself.
        """
        self.name = name
        self.rng = rng
        self.nbytes = sizeFromSideLen(depth, sideLen)
        self.stats = stats 
        self.kvStats = self.stats.mod('kv')
        self.ffCtx = ffCtx
        self.preTime = preprocessTime

        # Keys that we are responsible for deleting
        self.ownedKeys = []

        if self.nbytes > DEVICE_MEM_CAP:
            raise RuntimeError("Requested configuration would not fit on the device!")

        # Uniform shape for now
        self.shapes = [ mmShape(sideLen, sideLen, sideLen) ] * depth

        self.func = ChainedMults(name, self.shapes, ffCtx, mode, useCuda=useCuda, preprocessTime=self.preTime, stats=self.stats)

        if self.preTime is not None:
            if mode == 'direct':
                self.preFunc = ff.invoke.DirectRemoteFunc(workerPath, 'preprocess', self.ffCtx)
            else:
                self.preFunc = ff.invoke.ProcessRemoteFunc(workerPath, 'preprocess', self.ffCtx)


    def invoke(self, inArr):
        if self.rng is not None:
            time.sleep(self.rng() / 1000)

        with ff.timer('t_e2e', self.stats):
            cleanInput = False
            if not isinstance(inArr, str):
                inKey = self.name + "_input"
                with ff.timer("t_write_input", self.stats):
                    self.ffCtx.kv.put(self.name + "_input", inArr, profile=self.kvStats)
                cleanInput = True
            else:
                inKey = inArr

            # if self.preTime is not None:
            #     self.preFunc.Invoke({'input': 
            self.lastRetKey = self.func.invoke(inKey)

            if cleanInput:
                self.ffCtx.kv.delete(inKey, profile=self.kvStats)


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
            with ff.timer("t_write_input", self.stats):
                for i,b in enumerate(inBufs):
                    inName = self.name + "_input" + str(i)
                    self.ffCtx.kv.put(inName, b, profile=self.kvStats)
                    inNames.append(inName)
            inBufs = inNames
            deleteInputs = True

        for i in range(n):
            self.invoke(inBufs[ i % len(inBufs) ])
            if fetchResult:
                self.getResult()

            self.ffCtx.kv.delete(self.lastRetKey, profile=self.kvStats)

        if deleteInputs:
            for key in inBufs:
                self.ffCtx.kv.delete(key, profile=self.kvStats)


    def getStats(self):
        report = self.stats.report()
        return report 


    def getResult(self):
        with ff.timer("t_read_output", self.stats):
            raw = getData(self.ffCtx, self.lastRetKey, self.shapes[-1].c)
        return raw


    def destroy(self):
        self.func.destroy()
