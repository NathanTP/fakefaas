import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule, DynamicModule

import libff as ff

from .util import *

cudaLib = None
gemmFunc = None

class ChainedMults():
    def __init__(self, name, shapes, bArrs=None, useCuda=True):
        """Represents a series of matmuls where each multiply takes the output
        of the previous as A and uses a constant B. This simulates a basic
        fully-connected neural net.
        Shapes is a series of (N,M,K) for each layer (i.e. ncolB, nrowA, ncolA)"""

        self.name = name
        self.shapes = shapes
        self.useCuda = useCuda
        
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


    def destroy(self):
        pass


    # We delay most initialization until the first invocation to emulate
    # cold-start behavior of something like faas or kaas.
    def _checkInitialized(self, times):
        if self.useCuda:
            global cudaLib, gemmFunc
            if cudaLib is None:
                cudaLib = cuda.module_from_file(str(kernsDir / "gemm.cubin"))
                gemmFunc = cudaLib.get_function("sgemm")
                gemmFunc.prepare(["P"]*4)

            if self.devBarrs is None:
                self.devBarrs = []
                for harr in self.bArrs:
                    darr = cuda.mem_alloc(harr.nbytes)
                    cuda.memcpy_htod(darr, harr)
                    self.devBarrs.append(darr)


    def _invokeCuda(self, inBuf, outBuf=None, times=None):
        self._checkInitialized(times)

        inDbuf = cuda.mem_alloc(inBuf.nbytes)

        cBufs = []
        dimBufs = []
        for layerShape in self.shapes:
            cBufs.append(cuda.mem_alloc(mmShape.nbytes(layerShape.c)))
            dimBufs.append(cuda.mem_alloc(4*8))

        cuda.memcpy_htod(inDbuf, inBuf)

        aBuf = inDbuf
        for i in range(len(self.shapes)):
            dims = np.asarray(list(self.shapes[i].a) + list(self.shapes[i].b))
            cuda.memcpy_htod(dimBufs[i], dims)

            gridDim = (self.shapes[i].M // tileM, self.shapes[i].N // tileN, 1)
            blockDim = (tileN, tile_tb_height, 1)
            sharedSize = tile_tb_height * tileN * 4

            gemmFunc.prepared_call(gridDim, blockDim,
                    dimBufs[i], aBuf, self.devBarrs[i], cBufs[i],
                    shared_size=sharedSize)

            aBuf = cBufs[i]
        
        hostC = np.zeros((self.shapes[-1].M, self.shapes[-1].N), dtype=np.float32)
        cuda.memcpy_dtoh(hostC, cBufs[-1])

        # Free temporaries
        inDbuf.free()
        for i in range(len(self.shapes)):
            cBufs[i].free()
            dimBufs[i].free()

        return hostC


    def _invokeNP(self, inBuf, outBuf=None, times=None):
        aArr = inBuf
        for bArr in self.bArrs:
            c = np.matmul(aArr, bArr)
            aArr = c

        return aArr


    def invoke(self, inBuf, outBuf=None, times=None):
        if self.useCuda:
            return self._invokeCuda(inBuf, outBuf, times)
        else:
            return self._invokeNP(inBuf, outBuf, times)


    def destroy(self):
        if self.devBarrs is not None:
            for arr in self.devBarrs:
                cuda.mem_free(arr)


class benchClient():
    def __init__(self, name, depth, sideLen, rng=None, useCuda=True):
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
        
        Scale and depth must lead to reasonably sized matrices
        (mmFunc.matSizeA). The limits are:
            scale > (mmFunc.matSizeA * (1 + 2*depth))*4
            scale a multiple of mmFunc.matSizeA*4
        """
        self.rng = rng
        self.name = name
        self.stats = ff.profCollection()

        self.nbytes = sizeFromSideLen(depth, sideLen)

        if self.nbytes > DEVICE_MEM_CAP:
            raise RuntimeError("Requested configuration would not fit on the device!")

        # Uniform shape for now
        self.shapes = [ mmShape(sideLen, sideLen, sideLen) ] * depth

        self.func = ChainedMults(name, self.shapes, useCuda=useCuda)


    def invoke(self, inArr):
        """Invoke the client's function once, leaving the output in the kv
        store. If this object was created with an rng, invokeDelayed will wait
        a random amount of time before invoking. You can get the output of the
        last invocation with benchClient.getResult()."""
        with ff.timer("invoke", self.stats):
            if self.rng is not None:
                time.sleep(self.rng() / 1000)
            self.lastRet = self.func.invoke(inArr)

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


    def getStats(self, reset=False):
        return {
                "LocalStats" : self.stats.report(),
                "WorkerStats" : {}
        }


    def getResult(self):
        if self.lastRet is None:
            raise RuntimeError("Must invoke benchClient at least once to get a result")

        return self.lastRet        


    def destroy(self):
        pass
