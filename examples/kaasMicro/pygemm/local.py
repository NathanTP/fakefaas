import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule, DynamicModule

import libff as ff

from .util import *

cudaLib = None
gemmFunc = None

# Profiling level sets how aggressive we are in profiling.
#   - 0 no profiling (only affects worker/cuda stats)
#   - 1 metrics that will have little effect on performance
#   - 2 All metrics. This may have a performance impact, particularly for small kernels and/or data sizes.  
profLevel=2
def updateProf(prof, name, val, level=1):
    if level <= profLevel:
        prof[name].update(val)


def checkLevel(prof, level=1):
    """Returns the profs object (ff.profCollection) for this level. If level is
    above the current profLevel, Non is returned. This is suitable for use in a
    ff.timer context."""
    if level <= profLevel:
        return prof
    else:
        return None

# A list of all metrics that must be finalized after each invocation
# n_* is an event count, s_* is a size metric in bytes, t_* is a time measurement in ms
eventMetrics1 = [
        's_htod',
        't_htod',
        's_dtoh',
        't_dtoh',
        't_zero',
        't_kernelLoad',
        't_cudaMM'
        ]

eventMetrics2 = [
        't_kernel'
        ]


class ChainedMults():
    def __init__(self, name, shapes, ffCtx=None, bArrs=None, useCuda=True, stats=None):
        """Represents a series of matmuls where each multiply takes the output
        of the previous as A and uses a constant B. This simulates a basic
        fully-connected neural net.
        Shapes is a series of (N,M,K) for each layer (i.e. ncolB, nrowA, ncolA)
        
        ffCtx will be used to store the final output if it is provided. Name
        will be used to prefix keys."""

        self.shapes = shapes
        self.useCuda = useCuda
        self.name = name
        self.ffCtx = ffCtx
        self.stats = stats

        # devBarrs is cuda pointers and will be allocated on the first invocation
        self.devBarrs = None

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


    # We delay most initialization until the first invocation to emulate
    # cold-start behavior of something like faas or kaas.
    def _checkInitialized(self):
        with ff.timer('t_modelInit', checkLevel(self.stats, 1)): 
            if self.useCuda:
                with ff.timer("t_kernelLoad", checkLevel(self.stats, 1), final=False):
                    global cudaLib, gemmFunc
                    if cudaLib is None:
                        cudaLib = cuda.module_from_file(str(kernsDir / "gemm.cubin"))
                        gemmFunc = cudaLib.get_function("sgemm")
                        gemmFunc.prepare(["P"]*4)

                if self.devBarrs is None:
                    self.devBarrs = []
                    for harr in self.bArrs:
                        with ff.timer("t_cudaMM", checkLevel(self.stats, 1), final=False):
                            darr = cuda.mem_alloc(harr.nbytes)

                        updateProf(self.stats, 's_htod', 1)
                        with ff.timer('t_htod', checkLevel(self.stats, 1), final=False):
                            cuda.memcpy_htod(darr, harr)

                        self.devBarrs.append(darr)


    def _invokeCuda(self, inBuf, outBuf=None):
        self._checkInitialized()

        with ff.timer("t_cudaMM", checkLevel(self.stats, 1), final=False):
            inDbuf = cuda.mem_alloc(inBuf.nbytes)

            cBufs = []
            dimBufs = []
            for layerShape in self.shapes:
                cNBytes = mmShape.nbytes(layerShape.c)
                cBuf = cuda.mem_alloc(cNBytes)

                cBufs.append(cBuf)
                dimBufs.append(cuda.mem_alloc(4*8))

        # this is factored out of the allocation loop to make profiling cleaner
        with ff.timer("t_zero", checkLevel(self.stats, 1), final=False):
            for cBuf, shape in zip(cBufs, self.shapes):
                cuda.memset_d8(cBuf, 0, mmShape.nbytes(shape.c))

        updateProf(self.stats, 's_htod', inBuf.nbytes, 1)
        with ff.timer("t_htod", checkLevel(self.stats, 1), final=False):
            cuda.memcpy_htod(inDbuf, inBuf)

        aBuf = inDbuf
        for i in range(len(self.shapes)):
            dims = np.asarray(list(self.shapes[i].a) + list(self.shapes[i].b))

            updateProf(self.stats, 's_htod', dims.nbytes, 1)
            with ff.timer("t_htod", checkLevel(self.stats, 1), final=False):
                cuda.memcpy_htod(dimBufs[i], dims)

            gridDim = (self.shapes[i].M // tileM, self.shapes[i].N // tileN, 1)
            blockDim = (tileN, tile_tb_height, 1)
            sharedSize = tile_tb_height * tileN * 4

            
            with ff.timer('t_kernel', checkLevel(self.stats, level=2), final=False):
                gemmFunc.prepared_call(gridDim, blockDim,
                        dimBufs[i], aBuf, self.devBarrs[i], cBufs[i],
                        shared_size=sharedSize)

                if profLevel >= 2:
                    # This isn't needed if we aren't profiling and can hurt performance
                    pycuda.driver.Context.synchronize()

            aBuf = cBufs[i]
        
        hostC = np.zeros((self.shapes[-1].M, self.shapes[-1].N), dtype=np.float32)

        updateProf(self.stats, 's_dtoh', hostC.nbytes, 1)
        with ff.timer("t_dtoh", checkLevel(self.stats, 1)):
            cuda.memcpy_dtoh(hostC, cBufs[-1])

        # Free temporaries
        with ff.timer("t_cudaMM", checkLevel(self.stats, 1)):
            inDbuf.free()
            for i in range(len(self.shapes)):
                cBufs[i].free()
                dimBufs[i].free()

        return hostC


    def _invokeNP(self, inBuf, outBuf=None):
        aArr = inBuf
        for bArr in self.bArrs:
            c = np.matmul(aArr, bArr)
            aArr = c

        return aArr


    def invoke(self, inBuf, outBuf=None, stats=None):
        with ff.timer("t_invoke", stats): 
            if self.useCuda:
                res = self._invokeCuda(inBuf, outBuf)

                if profLevel >= 1:
                    for metric in eventMetrics1:
                        self.stats[metric].increment()
                if profLevel >= 2:
                    for metric in eventMetrics2:
                        self.stats[metric].increment()
            else:
                res = self._invokeNP(inBuf, outBuf)
        if self.ffCtx is not None:
            self.ffCtx.kv.put(self.name+"_out", res, profile=stats.mod('kv'))
            return self.name+"_out"
        else:
            return res


    def destroy(self):
        if self.devBarrs is not None:
            for arr in self.devBarrs:
                arr.free()


class benchClient():
    def __init__(self, name, depth, sideLen, preprocessTime=None, ffCtx=None, rng=None, useCuda=True, stats=None):
        """Run a single benchmark client making mm requests. Depth is how many
        matmuls to chain together in a single call. sideLen is the number of
        elements in one side of an array (benchClient works only with square
        matrices). sideLen must be a multiple of mmFunc.tileM. Each matmul in
        the chain will use one static array and one dynamic array.
        
        rng: is used to generate inter-request times for invoke(). The current
        implementation uses synchronous requests, so this is really the delay
        after completion rather than a true inter-arrival period. it should be
        a function with no arguments that returns a wait time in ms. If rng is
        None, invokeDelayed and invoke are identical. You may use
        benchClient.poisson() or benchClient.zipf() to generate an rng.
        
        preprocessTime: If not None, the client will simulate a preprocessing
        phase that takes preprocessTime ms to complete.

        Scale and depth must lead to reasonably sized matrices
        (mmFunc.matSizeA). The limits are:
            scale > (mmFunc.matSizeA * (1 + 2*depth))*4
            scale a multiple of mmFunc.matSizeA*4

        METRICS: benchClient sets the following metrics in stats
            - t_e2e: The total time for one invocation of the prediction,
              including any pre/post processing and the model itself.
        """
        self.rng = rng
        self.name = name
        self.stats = stats
        if preprocessTime is not None:
            self.preprocessSeconds = preprocessTime / 1000
        else:
            self.preprocessSeconds = 0

        # We only use this for the bare-minimum externally-visible stuff.
        # Basically just the input and output arrays. Users would typically use
        # the local memory kv anyway.
        self.ffCtx = ffCtx

        self.nbytes = sizeFromSideLen(depth, sideLen)

        if self.nbytes > DEVICE_MEM_CAP:
            raise RuntimeError("Requested configuration would not fit on the device!")

        # Uniform shape for now
        self.shapes = [ mmShape(sideLen, sideLen, sideLen) ] * depth

        self.func = ChainedMults(name, self.shapes, ffCtx=ffCtx, useCuda=useCuda, stats=self.stats.mod('worker'))


    def invoke(self, inArr):
        """Invoke the client's function once, leaving the output in the kv
        store. If this object was created with an rng, invokeDelayed will wait
        a random amount of time before invoking. You can get the output of the
        last invocation with benchClient.getResult()."""
        if self.rng is not None:
            time.sleep(self.rng() / 1000)

        with ff.timer('t_e2e', self.stats):
            time.sleep(self.preprocessSeconds)

            self.lastRet = self.func.invoke(inArr, stats=self.stats)

        return self.lastRet


    def invokeN(self, n, inArrs=1, fetchResult=False):
        """Invoke the client n times, waiting self.rng() ms inbetween
        invocations. inArrs may be a list of arrays (either a numpy.ndarray or
        kaas.bufferSpec) to use as inputs, if len(inArrs) < n, the inputs will
        be invoked round-robin. If inArrs is a scalar, the benchClient will
        generate random arrays for you. Each array will only be uploaded to the
        kv store once. However, they will not be marked as constant and may not
        be cacheable (until we implement a real consistency protocol). You may
        optionally read the result of each invocation back to the client. This
        is only for benchmarking purposes, the result is immediately
        discarded."""

        if isinstance(inArrs, int):
            inBufs = []
            for i in range(inArrs):
                arr = generateArr(self.shapes[0].a)
                inBufs.append(arr)
        else:
            inBufs = inArrs

        for i in range(n):
            self.lastRet = self.invoke(inBufs[ i % len(inBufs) ])


    def getStats(self):
        return self.stats.report()


    def getResult(self):
        if self.lastRet is None:
            raise RuntimeError("Must invoke benchClient at least once to get a result")

        with ff.timer("t_read_output", self.stats):
            if isinstance(self.lastRet, str):
                res = getData(self.ffCtx, self.lastRet, self.shapes[-1].c)
            else:
                res = self.lastRet
        return res


    def destroy(self):
        pass
