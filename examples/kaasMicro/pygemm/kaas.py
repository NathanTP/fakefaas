import kaasServer

from .util import *

faasWorkerPath = pathlib.Path(__file__).parent.parent.resolve() / "faasWorker.py"

class mmFunc():
    def _bufArgToBuf(self, arg, bname, shape, const=False):
        """Buffer args to various mmFunc methods can be ndarray, or kaasBuf,
        this converts them all to a kaasBuf as appropriate. If arg is not
        already a kaasBuf, a new one will be created with const=const."""
        if isinstance(arg, kaasServer.bufferSpec):
            return arg
        elif isinstance(arg, np.ndarray):
            self.ffCtx.kv.put(self.name + "_" + bname, arg)
        elif arg is not None:
            raise RuntimeError("Unrecognized type (must be either ndarray or kaas.bufferSpec): " + str(type(arg)))

        # None and ndarray need to generate a bufferSpec
        buf = kaasServer.bufferSpec(self.name + "_" + bname, shape[0]*shape[1] * 4, const=const)
        self.generatedBufs.append(buf.name)
        return buf


    def __init__(self, name, shape, libffCtx, kaasHandle, constB=None):
        """Create a matmul function invoker. Shape should be the two matrix
        dimensions [(arows, acols), (brows, bcols)].
        
        constB (a numpy array or kaas.bufferSpec) may be provided to associate
        a permanent B associated with this multiplier rather than a dynamic
        one."""
        self.name = name
        self.ffCtx = libffCtx
        self.kHandle = kaasHandle

        # We remember every array we've allocated in the kv store so that we
        # can destroy them later.
        self.generatedBufs = []

        self.aShape = shape[0]
        self.bShape = shape[1]
        self.cShape = (shape[0][0], shape[1][1])

        if constB is not None:
            self.constB = self._bufArgToBuf(constB, 'b', self.bShape, const=True)
        else:
            self.constB = None

        # dims is a property of a multiplier function, not any particular
        # invocation. We upload it at registration time.
        self.ffCtx.kv.put(self.name+"_dims", np.asarray(list(self.aShape) + list(self.bShape), dtype=np.uint64))
        self.dimBuf = kaasServer.bufferSpec(self.name + "_dims", 4*8)
        self.generatedBufs.append(self.dimBuf.name)

        if mmKern == 'sgemm':
            if self.aShape[0] % tileM != 0 or self.bShape[1] % tileN != 0:
                raise RuntimeError("Arrays must be a multiple of tile size")

            self.gridDim = (self.aShape[0] // tileM, self.bShape[1] // tileN, 1)
            self.blockDim = (tileN, tile_tb_height, 1)
            self.sharedSize = tile_tb_height * tileN * 4
        else:
            threadBlock = 32
            self.gridDim = (math.ceil(self.bShape[0] / threadBlock), math.ceil(self.aShape[0] / threadBlock), 1)
            self.blockDim = (threadBlock, threadBlock, 1)
            self.sharedSize = (2 * blockDim[0] * blockDim[1])*4

       
    def getKernFunc(self, aData=None, bData=None, cData=None):
        """Returns the kernel function object for this multiplier. You can use
        this to compose larger operations. aData, bData, and cData may be:
            None: the buffer will be assumed to exist in the kv store with name 'NAME_BUFNAME' (e.g. 'foo_a')
            np.ndarray: The contents of the array will be uploaded to the kv as 'NAME_BUFNAME'
            kaas.bufferSpec: the buffer spec will be used directly

        Using np.ndarray for cData does nothing as the array will be
        overwritten anyway. If bData is None and the function was created with
        constB, constB will be used.
        """
        aBuf = self._bufArgToBuf(aData, 'a', self.aShape)

        if bData is None and self.constB is not None:
            bBuf = self.constB
        else:
            bBuf = self._bufArgToBuf(aData, 'b', self.aShape)

        cBuf = self._bufArgToBuf(cData, 'c', self.cShape)

        return kaasServer.kernelSpec(kernsDir / 'gemm.cubin',
            mmKern,
            self.gridDim, self.blockDim, sharedSize=self.sharedSize,
            inputs = [self.dimBuf, aBuf, bBuf],
            outputs = [cBuf])


    def invoke(self, aData=None, bData=None, cData=None, times=None):
        """Invoke this multiplier with inputs uploaded with 'name' (using
        uploadArrays()). Invoke returns the name of the result in libff.kv.
        Times may be provided to record invocation statistics. See
        getKernFunc() for details of the data arguments."""
        kern = self.getKernFunc(aData, bData, cData)

        req = kaasServer.kaasReq([ kern ])
        with libff.timer("t_client_invoke", times):
            self.kHandle.Invoke(req.toDict())

        return kern.outputs[0].name 


    def destroy(self):
        """Free any remote resources associated with this function. Any A, B,
        or C buffers (even bConst) are the responsibility of the caller to
        free."""
        for b in self.generatedBufs:
            self.ffCtx.kv.delete(b)


class ChainedMults():
    def __init__(self, name, shapes, libffCtx, kaasHandle, preprocessTime=None, bArrs=None):
        """Represents a series of matmuls where each multiply takes the output
        of the previous as A and uses a constant B. This simulates a basic
        fully-connected neural net.
        Shapes is a series of (N,M,K) for each layer (i.e. ncolB, nrowA, ncolA)"""

        self.name = name
        self.ffCtx = libffCtx
        self.kHandle = kaasHandle
        self.shapes = shapes
        self.preTime = preprocessTime


        # We keep these around so we can run 'invokeBaseline'
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

        self.funcs = []
        for i,b in enumerate(self.bArrs):
            self.funcs.append(mmFunc(name+"_l"+str(i),
                (self.shapes[i].a, self.shapes[i].b),
                self.ffCtx, self.kHandle,
                constB=b))

        if self.preTime is not None:
            if isinstance(kaasHandle, libff.invoke.DirectRemoteFunc):
                self.preFunc = libff.invoke.DirectRemoteFunc(faasWorkerPath, 'preprocess', self.ffCtx)
            else:
                self.preFunc = libff.invoke.ProcessRemoteFunc(faasWorkerPath, 'preprocess', self.ffCtx)

    def destroy(self):
        for f in self.funcs:
            f.destroy()


    def invoke(self, inBuf, outBuf=None, stats=None):
        generatedInput = False
        if isinstance(inBuf, np.ndarray):
            with ff.timer("t_write_input", stats):
                self.ffCtx.kv.put(self.name+"_l0_a", inBuf)
            inBuf = kaasServer.bufferSpec(self.name+"_l0_a", mmShape.nbytes(self.shapes[0].a))
            generatedInput = True

        if outBuf is None:
            outBuf = kaasServer.bufferSpec(self.name+"_out", mmShape.nbytes(self.shapes[-1].c))

        kerns = []
        nextIn = inBuf
        for i,f in enumerate(self.funcs):
            if i == len(self.funcs) - 1:
                cBuf = outBuf
            else:
                cBuf = kaasServer.bufferSpec(self.name+"_l"+str(i)+"_c",  mmShape.nbytes(self.shapes[i].c), ephemeral=True)

            kerns.append(f.getKernFunc(aData = nextIn, cData = cBuf))
            nextIn = cBuf

        if self.preTime is not None:
            with ff.timer("t_client_preprocess", stats):
                self.preFunc.Invoke({"input" : inBuf.name, "output" : inBuf.name, "processTime" : self.preTime})
            
        with ff.timer("t_client_invoke", stats):
            req = kaasServer.kaasReq(kerns)
            self.kHandle.Invoke(req.toDict())

        if generatedInput:
            self.ffCtx.kv.delete(inBuf.name)

        return outBuf.name


    def destroy(self):
        for func in self.funcs:
            func.destroy()


class benchClient():
    def __init__(self, name, depth, sideLen, ffCtx, kaasCtx, preprocessTime=None, rng=None):
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
        self.ff = ffCtx
        self.kaas = kaasCtx
        self.lastRetKey = None
        self.rng = rng
        self.name = name
        self.generatedBufs = []
        self.stats = ff.profCollection()

        # Used to name any generated arrays
        self.nextArrayID = 0

        self.nbytes = sizeFromSideLen(depth, sideLen)

        if self.nbytes > DEVICE_MEM_CAP:
            raise RuntimeError("Requested configuration would not fit on the device!")

        # Uniform shape for now
        self.shapes = [ mmShape(sideLen, sideLen, sideLen) ] * depth

        self.func = ChainedMults(name, self.shapes, self.ff, self.kaas, preprocessTime=preprocessTime)


    def invoke(self, inArr):
        """Invoke the client's function once, leaving the output in the kv
        store. If this object was created with an rng, invokeDelayed will wait
        a random amount of time before invoking. You can get the output of the
        last invocation with benchClient.getResult()."""
        if self.rng is not None:
            time.sleep(self.rng() / 1000)
        self.lastRetKey = self.func.invoke(inArr, stats=self.stats)
        return self.lastRetKey


    def invokeN(self, n, inArrs=1, fetchResult=False, stats=None):
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
                with ff.timer("t_write_input", self.stats):
                    inBufs.append(self._bufArgToBuf(arr))
        else:
            with ff.timer("t_write_input", self.stats):
                inBufs = [ self._bufArgToBuf(arr) for arr in inArrs ]

        for i in range(n):
            self.invoke(inBufs[ i % len(inBufs) ])
            if fetchResult:
                # Read the result and immediately discard to include result
                # reading time in the benchmark
                self.getResult()

            self.ff.kv.delete(self.lastRetKey)


    def getStats(self, reset=False):
        allStats = self.kaas.Stats(reset=reset)
        allStats['LocalStats'] = {**allStats['LocalStats'], **self.stats.report()}
        return allStats 


    def getResult(self):
        if self.lastRetKey is None:
            raise RuntimeError("Must invoke benchClient at least once to get a result")

        with ff.timer("t_read_output", self.stats):
            res = getData(self.ff, self.lastRetKey, self.shapes[-1].c)       
        return res
        

    def destroy(self):
        self.func.destroy()

        for b in self.generatedBufs:
            self.ff.kv.delete(b)


    def _bufArgToBuf(self, arg, const=False):
        """Buffer args to various mmFunc methods can be ndarray or kaasBuf,
        this converts them all to a kaasBuf as appropriate. If arg is not
        already a kaasBuf, a new one will be created with const=const."""
        if isinstance(arg, kaasServer.bufferSpec):
            return arg
        elif isinstance(arg, np.ndarray):
            arrName = self.name + "_array" + str(self.nextArrayID)
            self.nextArrayID += 1

            self.ff.kv.put(arrName, arg)
            b = kaasServer.bufferSpec(arrName, arg.nbytes)
            self.generatedBufs.append(b.name)
            return b
        else:
            raise RuntimeError("Unrecognized type (must be either ndarray or kaas.bufferSpec): " + str(type(arg)))
