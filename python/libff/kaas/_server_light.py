import pycuda.driver as cuda
import pycuda.tools
import numpy as np
import collections
import atexit
import os

from . import kaas

from . import cutlass

from . import complexCutlass

# logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)
# logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# Profiling level sets how aggressive we are in profiling.
#   - 0 no profiling
#   - 1 metrics that will have little effect on performance
#   - 2 synchronize cuda aggresively to get accurate measurements (hurts e2e
#       perf)
profLevel = 1

# This is a global reference to the current ff.profCollection (reset for each call)
profs = None

kCache = None
bCache = None

dir_path = os.path.dirname(os.path.realpath(__file__))


def profSync():
    """If required by the profiling level, synchronize the cuda context"""
    if profLevel > 1:
        cuda.Context.synchronize()


def updateProf(name, val, level=1):
    if level <= profLevel:
        profs[name].update(val)


def getProf(level=1, mod=None):
    """Returns the profs object (ff.profCollection) for this level. If level is
    above the current profLevel, None is returned. This is suitable for use in a
    ff.timer context."""
    if level <= profLevel:
        if mod is None:
            return profs
        else:
            return profs.mod(mod)
    else:
        return None


# A list of all metrics that must be finalized after each invocation
# n_* is an event count, s_* is a size metric in bytes, t_* is a time measurement in ms
eventMetrics = [
    'n_hostDMiss',
    'n_hostDHit',
    's_hostDLoad',
    't_hostDLoad',
    'n_hostDWriteBack',
    't_hostDWriteBack',
    'n_devDHit',
    'n_devDMiss',
    'n_devDEvict',
    't_devDEvict',
    's_devDWriteBack',
    's_htod',
    's_dtoh',
    't_htod',
    't_dtoh',
    't_zero',
    'n_KMiss',
    'n_KHit',
    't_kernelLoad',
    't_cudaMM',
    't_hostMM']

eventMetrics += ['t_makeRoom', 't_invokeExternal', 't_getKern', 't_setupArgs', 't_bCacheLoad']

# These metrics need to be handled with special cases
metricSpecial = ['t_invoke']


class kaasBuf():
    @classmethod
    def fromSpec(cls, spec, src=None):
        return cls(spec[0], spec[2], src,
                   size=spec[1], ephemeral=spec[3])

    def __init__(self, name, key, src, size=None, ephemeral=False):
        """A kaasBuffer represnts a binary data buffer managed by the kaas
        system, it can be either on the device or on the host. If src is
        provided, it will be used as the buffer, otherwise size will be used to
        represent a zeroed buffer. kaasBuffers do not automatically manage
        consistency, it is up to the user to decide when and how to synchronize
        host and device memory.
        """
        self.name = name
        self.key = key
        self.dirty = False
        self.ephemeral = ephemeral
        self.useCount = 0

        if src is not None:
            self.dbuf = None
            self.hbuf = memoryview(src)
            self.size = self.hbuf.nbytes
            self.onDevice = False
        else:
            self.dbuf = None
            self.hbuf = None
            self.size = size
            self.onDevice = False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "KaaS Buffer (name={}, key={}, dirty={}, ephemeral={}, onDevice={}, size={})".format(
                self.name, self.key, self.dirty, self.ephemeral, self.onDevice, self.size)

    def setHostBuffer(self, newBuf):
        """Allows changing the host-side memory allocated to this buffer. This
        is useful for zero copy data transfers. The newBuf must be bigger or
        equal to the kaasBuf size. If newBuf is None, the kaasBuf will drop its
        reference to any host buffer, allowing host memory to be reclaimed."""
        if newBuf is None:
            self.hbuf = None
        else:
            newBuf = memoryview(newBuf)
            if newBuf.nbytes < self.size:
                raise kaas.KaasError("New buffer is not big enough")

            self.hbuf = newBuf

    def toDevice(self):
        """Place the buffer onto the device if it is not already there.  If no
        host buffer is set, zeroed device memory will be allocated.

        Returns: amount of additional memory used on the device"""
        if self.onDevice:
            return 0
        else:
            self.dbuf = cuda.mem_alloc(self.size)

            if self.hbuf is not None:
                cuda.memcpy_htod(self.dbuf, self.hbuf)

            # if self.hbuf is None:
            #     cuda.memset_d8(self.dbuf, 0, self.size)
            # else:
            #     cuda.memcpy_htod(self.dbuf, self.hbuf)

            self.onDevice = True
            return self.size

    def toHost(self):
        """Copy data from the device (if present). If the kaasBuf does not have
        a host buffer set, one will be allocated, otherwise the currently
        configured host buffer will be overwritten with the device buffer
        data."""
        if not self.onDevice:
            return
        else:
            if self.hbuf is None:
                self.hbuf = memoryview(bytearray(self.size))

            cuda.memcpy_dtoh(self.hbuf, self.dbuf)

    def freeDevice(self):
        """Free any memory that is allocated on the device."""
        if not self.onDevice:
            return
        else:
            self.dbuf.free()
            self.dbuf = None
            self.onDevice = False

    def clear(self):
        self.hbuf = None
        if self.onDevice:
            cuda.memset_d8(self.dbuf, 0, self.size)


class kaasFunc():
    def __init__(self, library, fName, literalTypes, nBuf):
        self.func = library.get_function(fName)
        self.func.prepare(literalTypes + ["P"]*nBuf)

        self.fName = fName
        self.lib = library

    def __str__(self):
        return self.func

    def Invoke(self, literals, bufs, gridDim, blockDim, sharedSize):
        """Invoke the func with the provided diminsions:
            - bufs is a list of kaasBuf objects
            - outs is a list of indexes of bufs to copy back to host memory
              after invocation
        """

        literalVals = [lit[1] for lit in literals]

        dAddrs = []
        for b in bufs:
            if not b.onDevice:
                raise RuntimeError(f"Provided buffer was not resident on the device (insufficient memory?): {b.name} ({b.key})")

            dAddrs.append(b.dbuf)

        args = literalVals + dAddrs

        if self.fName == "_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock12MmaPipelinedINS1_9GemmShapeILi128ELi128ELi8EEENS_9transform11threadblock22PredicatedTileIteratorINS_11MatrixShapeILi128ELi8EEEfNS_6layout8RowMajorELi1ENS8_30PitchLinearStripminedThreadMapINSD_16PitchLinearShapeILi8ELi128EEELi256ELi1EEELi1EEENS9_19RegularTileIteratorISC_fNSD_11ColumnMajorELi1ENS8_33TransposePitchLinearThreadMapSimtISI_EELi4EEENSA_INSB_ILi8ELi128EEEfSE_Li0ENSF_INSG_ILi128ELi8EEELi256ELi1EEELi1EEENSK_ISP_fSE_Li0ESR_Li4EEEfSE_NS4_9MmaPolicyINS1_4warp7MmaSimtINS6_ILi32ELi64ELi8EEEfSL_fSE_fSE_NSV_13MmaSimtPolicyINSB_ILi4ELi8EEENSD_19RowMajorInterleavedILi2EEENS6_ILi4ELi4ELi1EEEEELi1ELNS_16ComplexTransformE0ELS14_0EbEENSB_ILi4ELi0EEENSB_ILi0ELi0EEELi1EEENS_21NumericArrayConverterIffLi4ELNS_15FloatRoundStyleE2EEES1B_bEENS_8epilogue11threadblock8EpilogueIS7_S15_Li1ENS1E_22PredicatedTileIteratorINS1E_26OutputTileOptimalThreadMapINS1E_15OutputTileShapeILi128ELi1ELi4ELi4ELi1EEENS1I_ILi1ELi4ELi2ELi1ELi8EEELi256ELi1ELi32EEEfEENS1D_4warp20FragmentIteratorSimtISX_NS1_6thread3MmaINS6_ILi8ELi8ELi1EEEfSL_fSE_fSE_NS_4arch13OpMultiplyAddEbEESE_S13_EENS1N_16TileIteratorSimtISX_S1U_fSE_S13_EENS1E_18SharedLoadIteratorINS1L_18CompactedThreadMapEfLi4EEENS1D_6thread17LinearCombinationIfLi1EffLNS21_9ScaleType4KindE0ELS1A_2EEENSB_ILi0ELi17EEELi1EEENS4_30GemmIdentityThreadblockSwizzleILi1EEELb0EEEEEvNT_6ParamsE":
            cuda.memset_d8(bufs[2].dbuf, 0, bufs[2].size)
            params = cutlass.parseSgemmArgs(literalVals, dAddrs, kCache.cutlassAdapter)
            self.func.prepare("320s")
            return self.func.prepared_call(gridDim, blockDim, params.contents, shared_size=sharedSize)
        elif self.fName == "_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock12MmaPipelinedINS1_9GemmShapeILi128ELi128ELi8EEENS_9transform11threadblock22PredicatedTileIteratorINS_11MatrixShapeILi128ELi8EEENS_7complexIfEENS_6layout8RowMajorELi1ENS8_30PitchLinearStripminedThreadMapINSF_16PitchLinearShapeILi8ELi128EEELi256ELi1EEELi1EEENS9_19RegularTileIteratorISC_SE_NSF_11ColumnMajorELi1ENS8_33TransposePitchLinearThreadMapSimtISK_EELi8EEENSA_INSB_ILi8ELi128EEESE_SG_Li0ENSH_INSI_ILi128ELi8EEELi256ELi1EEELi1EEENSM_ISR_SE_SG_Li0EST_Li8EEESE_SG_NS4_9MmaPolicyINS1_4warp7MmaSimtINS6_ILi32ELi64ELi8EEESE_SN_SE_SG_SE_SG_NSX_13MmaSimtPolicyINSB_ILi4ELi8EEENSF_19RowMajorInterleavedILi2EEENS6_ILi2ELi2ELi1EEEEELi1ELNS_16ComplexTransformE0ELS16_0EbEENSB_ILi2ELi0EEENSB_ILi0ELi0EEELi1EEENS_21NumericArrayConverterISE_SE_Li4ELNS_15FloatRoundStyleE2EEES1D_bEENS_8epilogue11threadblock8EpilogueIS7_S17_Li1ENS1G_22PredicatedTileIteratorINS1G_26OutputTileOptimalThreadMapINS1G_15OutputTileShapeILi128ELi1ELi4ELi4ELi1EEENS1K_ILi1ELi2ELi4ELi1ELi8EEELi256ELi1ELi64EEESE_EENS1F_4warp20FragmentIteratorSimtISZ_NS1_6thread3MmaINS6_ILi8ELi8ELi1EEESE_SN_SE_SG_SE_SG_NS_4arch13OpMultiplyAddEbEESG_S15_EENS1P_16TileIteratorSimtISZ_S1W_SE_SG_S15_EENS1G_18SharedLoadIteratorINS1N_18CompactedThreadMapESE_Li8EEENS1F_6thread17LinearCombinationISE_Li1ESE_SE_LNS23_9ScaleType4KindE0ELS1C_2EEENSB_ILi0ELi9EEELi1EEENS4_30GemmIdentityThreadblockSwizzleILi1EEELb0EEEEEvNT_6ParamsE":
            cuda.memset_d8(bufs[2].dbuf, 0, bufs[2].size)
            params = complexCutlass.parseSgemmArgs(literalVals, dAddrs, kCache.complexAdapter)
            self.func.prepare("328s")
            return self.func.prepared_call(gridDim, blockDim, params.contents, shared_size=sharedSize)

        else:
            return self.func.prepared_call(gridDim, blockDim, *args, shared_size=sharedSize)


class kernelCache():
    def __init__(self):
        self.libs = {}
        self.kerns = {}

        pycuda.driver.init()
        self.cudaCtx = pycuda.tools.make_default_context()

        self.cutlassAdapter = cutlass.loadSgemmAdapter()

        self.complexAdapter = complexCutlass.loadAdapter()

    def get(self, spec):
        name = spec[0]
        if name not in self.kerns:
            libPath = spec[1]
            args = spec[7]
            literals = spec[6]
            kernelFunc = spec[2]

            if libPath not in self.libs:
                self.libs[libPath] = cuda.module_from_file(libPath)

            litTypes = [lit[0] for lit in literals]
            self.kerns[name] = kaasFunc(self.libs[libPath], kernelFunc, litTypes, len(args))

        return self.kerns[name]


class lruPolicy():
    def __init__(self):
        # Only contains buffers that are currently on the device. Constants are
        # given higher priority than other types of buffers.
        self.lruConst = collections.deque()
        self.lruOther = collections.deque()

    def remove(self, buf):
        """Remove buffer from consideration"""
        if buf.useCount > 2:
            self.lruConst.remove(buf)
        else:
            self.lruOther.remove(buf)

    def push(self, buf):
        """Place buffer in the most-recently-used slot"""
        if buf.useCount > 2:
            self.lruConst.appendleft(buf)
        else:
            self.lruOther.appendleft(buf)

    def pop(self):
        """Remove and return the least recently used buffer"""
        # pull from lruOther if we can, otherwise start evicting consts
        if self.lruOther:
            b = self.lruOther.pop()
        else:
            b = self.lruConst.pop()

        return b


class bufferCache():
    """The buffer cache tracks buffers both on and off the device. Any sort of
    distributed storage properties are provided by an external KV store.
    Buffers are explicitly loaded onto the device by clients and automatically
    migrated off the device as needed. The cache is inclusive (anything on the
    device also exists on the host). Buffers are only committed back to the KV
    store explicitly by the client (dirty() and flush()). Eventually, this
    should really be integrated with the normal cache but for now it's a
    separate thing.

    WARNING: This cache doesn't really do everything it should correctly, I'm
    just trying to get something to work. It works with the server as written
    but it's really bad at keeping things consistent. Writing a real cache is a
    task for another day."""

    def __init__(self):
        """Initialize a device buffer cache with byte capacity cap. If this cap
        is exceeded, things will be evicted"""
        # Contains all bufs, on device or not
        self.bufs = {}
        self.dirtyBufs = {}
        self.ephemerals = {}

        self.policy = lruPolicy()

        memFree, memAvail = cuda.mem_get_info()

        # Maximum number of bytes on the device
        # We build in a bit of a margin because we can't track size very
        # accurately. We assume we won't be off by more than this margin within
        # a single request (this is just a guess).
        self.cap = memAvail - 10*1024*1024

        # Size represents the amount of memory used on the device, it's more
        # complicated than just the sum of buffer sizes because of device
        # memory overheads (e.g. cudaMalloc() uses a lot of device space)
        self.size = memAvail - memFree

    def setKV(self, kv):
        self.kv = kv

    def updateSize(self):
        memFree, memAvail = cuda.mem_get_info()
        self.size = memAvail - memFree

    def makeRoom(self, sz):
        while self.cap - self.size < sz:
            b = self.policy.pop()

            if b.dirty:
                b.toHost()

            b.freeDevice()
            self.size -= b.size

    def makeRoomForBufs(self, bSpecs):
        """Ensure that we have enough room to load all the buffers in bSpecs.
        This accounts for if any of the buffers are already on the device. This
        function helps performance and possibly memory fragmentation by
        batching frees()."""
        total = 0
        for bSpec in bSpecs:
            key = bSpec[2]
            size = bSpec[1]
            cacheBuf = self.bufs.get(key, None)
            if cacheBuf is None:
                total += size
            else:
                if not cacheBuf.onDevice:
                    total += size

        self.makeRoom(total)

    def load(self, bSpec, overwrite=False):
        """Load a buffer onto the device, fetching from the KV store if
        necessary. If overwrite=True, a new buffer will be created. If the
        buffer is already in the cache and dirty, it will be written back and
        re-read (you should probably make sure this doesn't happen by flushing
        when needed)."""

        key = bSpec[2]
        ephemeral = bSpec[3]

        buf = self.bufs.get(key, None)
        if buf is not None:
            # if overwrite:
            #     buf.clear()

            # Reset LRU
            if buf.onDevice:
                self.policy.remove(buf)
        else:
            if ephemeral or overwrite:
                buf = kaasBuf.fromSpec(bSpec)

                if buf.ephemeral:
                    self.ephemerals[key] = buf
            else:
                raw = self.kv.get(key)
                if raw is None:
                    buf = kaasBuf.fromSpec(bSpec)
                else:
                    buf = kaasBuf.fromSpec(bSpec, raw)

        self.makeRoom(buf.size)

        self.size += buf.toDevice()

        self.bufs[key] = buf

        return buf

    def release(self, buf):
        buf.useCount += 1
        self.policy.push(buf)

    def dirty(self, key):
        buf = self.bufs[key]
        buf.dirty = True
        self.dirtyBufs[key] = buf

    def _flushOne(self, buf):
        if buf.dirty:
            if buf.onDevice:
                buf.toHost()
            if not buf.ephemeral:
                # Data are stored as numpy arrays because memoryviews can't be
                # pickled. This should still be zero copy.
                self.kv.put(buf.key, np.asarray(buf.hbuf))
            buf.dirty = False

    def flush(self):
        for b in self.dirtyBufs.values():
            self._flushOne(b)

        self.dirtyBufs = {}

    def drop(self, key):
        """Remove a buffer from the cache (writing back if dirty). This frees
        any device memory and drops references to the host buffer (Python's GC
        will pick it up eventually)."""
        buf = self.bufs[key]
        self._flushOne(buf)

        if buf.onDevice:
            self.policy.remove(buf)
            buf.freeDevice()
            self.size -= buf.size

        if buf.dirty:
            self.dirtyBufs.pop(buf.key, None)
        if buf.ephemeral:
            self.ephemerals.pop(buf.key, None)

        del self.bufs[buf.key]


def kaasServeInternal(req, ctx):
    """Internal implementation of kaas execution. Req is a kaas.kaasReq, not a
    dictionary"""

    # These should only get initialized upon invocation, not at import. This is
    # most important for the kCache which has to manage a cuda device context
    global kCache, bCache
    if kCache is None:
        kCache = kernelCache()

    if bCache is None:
        bCache = bufferCache()

    # This gets reset every call because libff only gives us the kv handle per
    # call and it could (in theory) change in some way between calls.
    bCache.setKV(ctx.kv)

    # We can't estimate memory utilization perfectly so we update our estimate
    # on every request
    bCache.updateSize()

    global profs
    profs = ctx.stats

    # We try to help the cuda memory allocator out by freeing all the buffers
    # at once instead of mixing frees and mallocs at a fine-grain. This loop
    # finds all the unique buffers in the request
    bCache.makeRoomForBufs(req.bufferMap.values())

    # when nIter > 1, we may see the same output multiple times, but we only
    # want to report it once.
    visibleOutputs = set()
    for i in range(req.nIter):
        for kSpec in req.kernels:
            kern = kCache.get(kSpec)

            specArgs = kSpec[7]
            ioTypes = kSpec[8]




            arguments = []
            for argName, ioType in zip(specArgs, ioTypes):
                arg = req.bufferMap[argName]
                if ioType == 'o':
                    argBuf = bCache.load(arg, overwrite=True)
                else:
                    argBuf = bCache.load(arg)

                if (ioType == 'o' or ioType == 'io') and not argBuf.ephemeral:
                    bCache.dirty(argBuf.key)
                    visibleOutputs.add(argBuf.key)

                arguments.append(argBuf)



            kern.Invoke(kSpec[6], arguments, kSpec[3], kSpec[4], kSpec[5])

            for buf in arguments:
                bCache.release(buf)

            # ***********************
            # It turns out on the big models, cudaMM is dominant. We should measure
            # it, but the overhead of extra evictions and stuff is unlikely to
            # outweight the opportunity for buffer re-use. We just bzero stuff
            # instead.
            # ***********************
            # # Don't bother waiting for the caching policy on temps, they will for
            # # sure never be needed again. In the future we may avoid this to save
            # # on cudaMalloc.
            # for t in kSpec.temps:
            #     bCache.drop(t.key)
            #
            # # For now, we assume we'll never re-use non-constant inputs. It's true
            # # for current workloads but not in general. This saves us a bunch of
            # # time spent evicting stuff.
            # for bSpec in kSpec.inputs:
            #     if not bSpec.const:
            #         bCache.drop(bSpec.key)

    # Make sure all outputs are visible externally (basically this merges our
    # private state into whatever consistency properties the KV gives us.
    bCache.flush()

    return visibleOutputs


@atexit.register
def kaasCleanup():
    global kCache
    global bCache

    if bCache is not None:
        bCache.flush()
        bCache = None

    if kCache is not None:
        kCache.cudaCtx.detach()
        kCache = None
