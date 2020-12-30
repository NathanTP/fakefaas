import pathlib
import math
from pprint import pprint
import time
import numpy as np
import csv

import libff as ff
import libff.kv
import libff.invoke

import kaasServer as kaas

redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"
testPath = pathlib.Path(__file__).resolve().parent
# colMajor = True
colMajor = False

DEVICE_MEM_CAP = 1024*1024*1024*4

def getCtx(remote=False):
    if remote:
        objStore = ff.kv.Redis(pwd=redisPwd, serialize=True)
    else:
        objStore = ff.kv.Local(copyObjs=False, serialize=False)

    return libff.invoke.RemoteCtx(None, objStore)


class mmShape():
    def __init__(self, M, N, K):
        """Represents the shape of a single matmul operation:
                M: nrowsA and nrowsC
                N: ncolB and ncolC
                K: inner dimension (ncolA and nrowB)
        """
        self.M = M
        self.N = N
        self.K = K

        self.a = (self.M, self.K)
        self.b = (self.K, self.N)
        self.c = (self.M, self.N)


    @staticmethod
    def nbytes(matShape):
        return matShape[0]*matShape[1]*4


rng = np.random.default_rng(0)
def generateArr(shape):
    arr = rng.random(shape, dtype=np.float32)
    if colMajor:
        arr = np.asfortranarray(arr)

    return arr

mmKern = "sgemm"
# mmKern = "matmul"
class mmFunc():
    # tileN = 32
    # tile_tb_height = 16 
    tile_tb_height = 8

    tileN = 16
    tileM = (tileN * tile_tb_height)

    def _bufArgToBuf(self, arg, bname, shape, const=False):
        """Buffer args to various mmFunc methods can be ndarray, or kaasBuf,
        this converts them all to a kaasBuf as appropriate. If arg is not
        already a kaasBuf, a new one will be created with const=const."""
        if isinstance(arg, kaas.bufferSpec):
            return arg
        elif isinstance(arg, np.ndarray):
            self.ffCtx.kv.put(self.name + "_" + bname, arg)
        elif arg is not None:
            raise RuntimeError("Unrecognized type (must be either ndarray or kaas.bufferSpec): " + str(type(arg)))

        # None and ndarray need to generate a bufferSpec
        buf = kaas.bufferSpec(self.name + "_" + bname, shape[0]*shape[1] * 4, const=const)
        self.generatedBufs.append(buf)
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
        self.dimBuf = kaas.bufferSpec(self.name + "_dims", 4*8)
        self.generatedBufs.append(self.dimBuf)

        if mmKern == 'sgemm':
            if self.aShape[0] % self.tileM != 0 or self.bShape[1] % self.tileN != 0:
                raise RuntimeError("Arrays must be a multiple of tile size")

            self.gridDim = (self.aShape[0] // self.tileM, self.bShape[1] // self.tileN, 1)
            self.blockDim = (self.tileN, self.tile_tb_height, 1)
            self.sharedSize = self.tile_tb_height * self.tileN * 4
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

        return kaas.kernelSpec(testPath / 'kerns' / 'gemm.cubin',
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

        req = kaas.kaasReq([ kern ])
        with libff.timer("invoke", times):
            self.kHandle.Invoke(req.toDict())

        return kern.outputs[0].name 


    def destroy(self):
        """Free any remote resources associated with this function. Any A, B,
        or C buffers (even bConst) are the responsibility of the caller to
        free."""
        for b in self.generatedBufs:
            self.ffCtx.kv.delete(b.name)


class ChainedMults():
    def __init__(self, name, shapes, libffCtx, kaasHandle):
        """Represents a series of matmuls where each multiply takes the output
        of the previous as A and uses a constant B. This simulates a basic
        fully-connected neural net.
        Shapes is a series of (N,M,K) for each layer (i.e. ncolB, nrowA, ncolA)"""

        self.name = name
        self.ffCtx = libffCtx
        self.kHandle = kaasHandle
        self.shapes = shapes

        self.funcs = []

        # We keep these around so we can run 'invokeBaseline'
        self.bArrs = []

        # prevShape is used to validate the chain dimensions, it's the C shape
        # of the previous layer. For the first layer, we just set it to A's
        # shape.
        prevShape = self.shapes[0].a
        for i,shape in enumerate(shapes):
            if shape.a != prevShape:
                raise RuntimeError("Invalid input shape for layer " + str(i) + " (" + str(shape.a) + ") previous layer output " + str(prevShape))
            
            constB = generateArr(shape.b)
            self.bArrs.append(constB)
            
            self.funcs.append(mmFunc(name+"_l"+str(i),
                (shape.a, shape.b),
                self.ffCtx, self.kHandle,
                constB=constB))

            prevShape = shape.c


    def destroy(self):
        for f in self.funcs:
            f.destroy()

    def invoke(self, inBuf, outBuf=None, times=None):
        generatedInput = False
        if isinstance(inBuf, np.ndarray):
            self.ffCtx.kv.put(self.name+"_l0_a", inBuf)
            inBuf = kaas.bufferSpec(self.name+"_l0_a", mmShape.nbytes(self.shapes[0].a))
            generatedInput = True

        if outBuf is None:
            outBuf = kaas.bufferSpec(self.name+"_out", mmShape.nbytes(self.shapes[-1].c))

        kerns = []
        nextIn = inBuf
        for i,f in enumerate(self.funcs):
            if i == len(self.funcs) - 1:
                cBuf = outBuf
            else:
                cBuf = kaas.bufferSpec(self.name+"_l"+str(i)+"_c",  mmShape.nbytes(self.shapes[i].c), ephemeral=True)

            kerns.append(f.getKernFunc(aData = nextIn, cData = cBuf))
            nextIn = cBuf

        req = kaas.kaasReq(kerns)
        with libff.timer("invoke", times):
            self.kHandle.Invoke(req.toDict())

        if generatedInput:
            self.ffCtx.kv.delete(inBuf.name)

        return outBuf.name


    def invokeBaseline(self, inArr, times=None):
        aArr = inArr
        for bArr in self.bArrs:
            c = np.matmul(aArr, bArr)
            aArr = c

        return c


    def destroy(self):
        for func in self.funcs:
            func.destroy()


def getData(ffCtx, name, shape):
    raw = ffCtx.kv.get(name)
    arr = np.frombuffer(raw, dtype=np.float32)
    if colMajor:
        arr = arr.reshape(shape[0], shape[1], order="F")
    else:
        arr = arr.reshape(shape[0], shape[1])
    
    return arr


class benchClient():

    @staticmethod
    def poisson(target):
        """Returns a poisson distribution generator suitable for benchClient
        initalization. Target is the desired mean latency in ms."""
        rng = np.random.default_rng()
        return lambda: rng.poisson(target)
    

    @staticmethod
    def zipf(a, scale=1, maximum=None):
        """Returns a zipf generator suitable for benchClient initialization. a
        is the zip factor. Zipf returns values starting at 1ms with an
        unbounded upper limit. You can change this by applying a scaling factor
        and maximum value."""
        rng = np.random.default_rng()

        def sampler():
            z = rng.zipf(a)*scale
            if maximum is not None and z > maximum:
                z = maximum
            return z

        return sampler
            

    @staticmethod
    def sideLenFromSize(depth, nbyte):
        """Returns the closest legal sideLen that would use at most nbyte bytes of
        arrays for depth."""
        # XXX nbyte = (nmat * (sideLen**2)) * 4
        # (nbyte / 4) / nmat = sideLen**2
        # sqrt((nbyte / 4) / nmat) = sideLen

         # 1 mtx for the input, depth matrices for the static, depth matrices for
        # the outputs
        nMatrix = 1 + (2*depth)

        # Biggest sidelen that would fit in nbyte
        sideLen = math.isqrt((nbyte // 4) // nMatrix)

        if sideLen < mmFunc.tileM:
            # it's impossible
            return None
       
        # Round down to the nearest multiple of tileM
        sideLen -= sideLen % mmFunc.tileM

        return sideLen


    @staticmethod
    def sizeFromSideLen(depth, sideLen):
        """Returns the number of bytes of device memory that will be needed for
        the given depth and matrix side length (in elements)"""
        nMatrix = 1 + (2*depth)
        return (nMatrix * (sideLen**2)) * 4


    def __init__(self, name, depth, sideLen, ffCtx, kaasCtx, rng=None):
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

        # Used to name any generated arrays
        self.nextArrayID = 0

        self.nbytes = self.sizeFromSideLen(depth, sideLen)

        if self.nbytes > DEVICE_MEM_CAP:
            raise RuntimeError("Requested configuration would not fit on the device!")

        # Uniform shape for now
        self.shapes = [ mmShape(sideLen, sideLen, sideLen) ] * depth

        self.func = ChainedMults(name, self.shapes, self.ff, self.kaas)


    def invoke(self, inArr):
        """Invoke the client's function once, leaving the output in the kv
        store. If this object was created with an rng, invokeDelayed will wait
        a random amount of time before invoking. You can get the output of the
        last invocation with benchClient.getResult()."""
        if self.rng is not None:
            time.sleep(self.rng() / 1000)
        self.lastRetKey = self.func.invoke(inArr)
        return self.lastRetKey


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
                inBufs.append(self._bufArgToBuf(arr))
        else:
            inBufs = [ self._bufArgToBuf(arr) for arr in inArrs ]

        for i in range(n):
            self.invoke(inBufs[ i % len(inBufs) ])
            if fetchResult:
                # Read the result and immediately discard to include result
                # reading time in the benchmark
                self.getResult()


    def getResult(self):
        if self.lastRetKey is None:
            raise RuntimeError("Must invoke benchClient at least once to get a result")

        return getData(self.ff, self.lastRetKey, self.shapes[-1].c)       
        

    def destroy(self):
        self.func.destroy()

        for b in self.generatedBufs:
            self.ff.kv.delete(b.name)


    def _bufArgToBuf(self, arg, const=False):
        """Buffer args to various mmFunc methods can be ndarray or kaasBuf,
        this converts them all to a kaasBuf as appropriate. If arg is not
        already a kaasBuf, a new one will be created with const=const."""
        if isinstance(arg, kaas.bufferSpec):
            return arg
        elif isinstance(arg, np.ndarray):
            arrName = self.name + "_array" + str(self.nextArrayID)
            self.nextArrayID += 1

            self.ff.kv.put(arrName, arg)
            b = kaas.bufferSpec(arrName, arg.nbytes)
            self.generatedBufs.append(b)
            return b
        else:
            raise RuntimeError("Unrecognized type (must be either ndarray or kaas.bufferSpec): " + str(type(arg)))


def testMMChained(mode='direct'):
    libffCtx = getCtx(remote=(mode == 'process'))
    kaasHandle = kaas.getHandle(mode, libffCtx)

    shapes = [
            mmShape(128,128,128),
            mmShape(128,256,128),
            mmShape(128,512,256) ]

    inArr = generateArr(shapes[0].a)

    func = ChainedMults("testchain", shapes, libffCtx, kaasHandle)

    retKey = func.invoke(inArr)

    testOut = getData(libffCtx, retKey, shapes[-1].c)

    baseArr = func.invokeBaseline(inArr)

    func.destroy()

    diff = testOut - baseArr
    dist = np.linalg.norm(diff)

    if dist > 10:
        print("FAIL")
        print("Distance: " + str(dist))
    else:
        print("PASS")


def testMMOne(mode='direct'):
    libffCtx = getCtx(remote=(mode == 'process'))
    kaasHandle = kaas.getHandle(mode, libffCtx)

    # arrA = generateArr((32,32))
    # arrB = generateArr((32,32))
    arrA = generateArr((1024,256))
    arrB = generateArr((256,512))
    libffCtx.kv.put("test_a", arrA)
    libffCtx.kv.put("test_b", arrB)

    shape = [arrA.shape, arrB.shape]
    func = mmFunc('test', shape, libffCtx, kaasHandle)

    rKey = func.invoke()
    
    arrC = getData(libffCtx, rKey, func.cShape)

    libffCtx.kv.delete('test_a')
    libffCtx.kv.delete('test_b')
    libffCtx.kv.delete('test_c')
    func.destroy()

    arrC = arrC.round(4)
    npC = np.matmul(arrA, arrB).round(4)

    # Single precision accumulates errors really fast so the differences can be
    # big. The euclidean distance seems to get us close enough for a pretty
    # good guess at correctness
    diff = arrC - npC
    dist = np.linalg.norm(diff)
    if dist > 1:
        print("FAIL:")
        print("Euclidean Dist: " + str(dist))
    else:
        print("PASS")


def testClient(mode='direct'):
    libffCtx = getCtx(remote=(mode == 'process'))
    kaasHandle = kaas.getHandle(mode, libffCtx)
    # clientPlain = benchClient("benchClientTest", 3, 128, libffCtx, kaasHandle)
    clientPlain = benchClient("benchClientTest", 4, 1024*4, libffCtx, kaasHandle)
    clientPlain.invokeN(1)
    clientPlain.destroy()
    
    print("Stats: ")
    s = kaasHandle.Stats()

    timeMetricsTotal = 0
    for k,v in s['WorkerStats'].items():
        if k[:2] == 't_':
            timeMetricsTotal += v
    print("Total Time: ", s['LocalStats']['invoke'])
    print("Time Metrics Total: ", timeMetricsTotal)
    print("Missing Time: ", s['LocalStats']['invoke'] - timeMetricsTotal)
    pprint(s)

    # clientPoisson = benchClient("benchClientTest", 3, 128, libffCtx, kaasHandle, rng=benchClient.poisson(5))
    # clientPoisson.invokeN(5, inArrs=5)
    # clientPoisson.destroy()
    #
    # clientZipf = benchClient("benchClientTest", 3, 128, libffCtx, kaasHandle, rng=benchClient.zipf(2, maximum=100))
    # clientZipf.invokeN(5, inArrs = [ generateArr(clientZipf.shapes[0].a) for i in range(5) ])
    # clientZipf.destroy()


def startKaas(mode='direct'):
    """Start the kaas server and run some trivial computation to ensure the kaas server is warm."""
    libffCtx = getCtx(remote=(mode == 'process'))
    kaasHandle = kaas.getHandle(mode, libffCtx)

    kern = kaas.kernelSpec(testPath / 'kerns' / 'noop.cubin', 'noop', (1,1,1), (1,1,1))
    kaasHandle.Invoke(kaas.kaasReq([kern]).toDict())
    kaasHandle.Stats(reset=True)

    return (libffCtx, kaasHandle)


def benchmark(name, depth, size, mode, nrepeat, outPath=None):
    """Run a benchmark, outputing a CSV named ${name}.csv with the name column
    set to name.  The benchmark will be run with the depth,size,mode and
    repeated nrpeat times + 1 (cold start + nrepeat warm starts).

    If outPath is set, the results will be appended to that CSV file instead
    of creating a new one."""

    ffCtx, kaasCtx = startKaas(mode)
    client = benchClient('benchmark-'+ mode, depth, size, ffCtx, kaasCtx)

    configDict = { 'name' : name, 'mode' : mode, 'nrepeat' : nrepeat,
            'matDim' : size, 'depth' : depth, 'nbyte' :  benchClient.sizeFromSideLen(depth, size)}

    # Cold Start
    client.invokeN(1)
    coldStats = kaasCtx.Stats(reset=True)['WorkerStats']
    coldStats['warm'] = False 
    coldStats = {**coldStats, **configDict}

    # Warm Start
    client.invokeN(nrepeat)
    warmStats = kaasCtx.Stats(reset=True)['WorkerStats']
    warmStats['warm'] = True
    warmStats = {**warmStats, **configDict}

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

if __name__ == "__main__":
    # print(benchClient.sizeFromSideLen(3, 1024*8) / (1024*1024*1024))
    # testMMOne('direct')
    # testMMChained('direct')
    # testClient('direct')
    # benchmark('testingBench', 1, 128, 'direct', 2, outPath='test.csv')

    benchmark('smallDirect', 4, 1024,   'direct', 5, outPath='matmul.csv')
    # benchmark('largeDirect', 4, 1024*8, 'direct', 5, outPath='matmul.csv')

    # benchmark('smallRemote', 4, 1024,   'process', 5, outPath='matmul.csv')
    # benchmark('largeRemote', 4, 1024*8, 'process', 5, outPath='matmul.csv')
