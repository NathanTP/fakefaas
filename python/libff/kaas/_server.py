import pycuda.driver as cuda
import pycuda.tools
# import pycuda.autoinit  # NOQA (this import forces pycuda to initialize but the linter doesn't like it)
import ctypes
import ctypes.util
import numpy as np
import collections
import logging
import atexit

import libff as ff
import libff.kv

from . import kaas

# logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

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


def profSync():
    """If required by the profiling level, synchronize the cuda context"""
    if profLevel > 1:
        cuda.Context.synchronize()


def updateProf(name, val, level=1):
    if level <= profLevel:
        profs[name].update(val)


def getProf(level=1, mod=None):
    """Returns the profs object (ff.profCollection) for this level. If level is
    above the current profLevel, Non is returned. This is suitable for use in a
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

eventMetrics += ['t_makeRoom', 't_invokeExternal']

# These metrics need to be handled with special cases
metricSpecial = ['t_invoke']


class kaasBuf():
    @classmethod
    def fromSpec(cls, spec, src=None):
        return cls(spec.name, spec.key, src,
                   size=spec.size, const=spec.const, ephemeral=spec.ephemeral)

    def __init__(self, name, key, src, size=None, const=False, ephemeral=False):
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
        self.const = const
        self.ephemeral = ephemeral

        # Pinned buffers cannot be evicted, this is mostly a safety check since
        # the eviction policy would only evict important buffers if a single
        # request doesn't fit in memory.
        self.pin = False

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
        return "KaaS Buffer (name={}, key={}, dirty={}, const={}, ephemeral={}, onDevice={}, size={})".format(
                self.name, self.key, self.dirty, self.const, self.ephemeral, self.onDevice, self.size)

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
            logging.debug("Moving {} to device".format(self.name))
            updateProf('s_htod', self.size)

            with ff.timer('t_cudaMM', getProf(), final=False):
                self.dbuf = cuda.mem_alloc(self.size)
                profSync()

            if self.hbuf is None:
                logging.debug("Zeroing {} ({})".format(self.name, hex(int(self.dbuf))))
                with ff.timer('t_zero', getProf(), final=False):
                    cuda.memset_d8(self.dbuf, 0, self.size)
                    profSync()

            else:
                with ff.timer('t_htod', getProf(), final=False):
                    cuda.memcpy_htod(self.dbuf, self.hbuf)
                    profSync()

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
            logging.debug("Moving {}b from device (0x{}) {} to host".format(
                self.size, hex(int(self.dbuf)), self.name))

            if self.hbuf is None:
                with ff.timer('t_hostMM', getProf(), final=False):
                    self.hbuf = memoryview(bytearray(self.size))

            updateProf('s_dtoh', self.size)
            with ff.timer('t_dtoh', getProf(), final=False):
                cuda.memcpy_dtoh(self.hbuf, self.dbuf)
                profSync()

    def freeDevice(self):
        """Free any memory that is allocated on the device."""
        if not self.onDevice:
            return
        else:
            with ff.timer('t_cudaMM', getProf(), final=False):
                self.dbuf.free()
                profSync()
            self.dbuf = None
            self.onDevice = False

    def clear(self):
        logging.debug("Clearing existing buffer " + self.name)
        with ff.timer('t_zero', getProf(), final=False):
            if self.onDevice:
                cuda.memset_d8(self.dbuf, 0, self.size)
                profSync()
                self.hbuf = None
            else:
                if self.hbuf is not None:
                    ctypes.memset(self.hbuf, 0, self.size)


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

        literalVals = [lit.val for lit in literals]

        dAddrs = []
        for b in bufs:
            if not b.onDevice:
                raise RuntimeError("Provided buffer was not resident on the device (insufficient memory?): " + b.name)

            dAddrs.append(b.dbuf)

        args = literalVals + dAddrs

        logging.debug("Invoking <<<{}, {}, {}>>>{}({})".format(gridDim, blockDim, sharedSize, self.fName,
                      ", ".join([str(lit) for lit in literalVals] + [str(b.name) for b in bufs])))

        return self.func.prepared_timed_call(gridDim, blockDim, *args, shared_size=sharedSize)


class kernelCache():
    def __init__(self):
        self.libs = {}
        self.kerns = {}

        pycuda.driver.init()
        self.cudaCtx = pycuda.tools.make_default_context()

    def get(self, spec):
        if spec.name not in self.kerns:
            updateProf('n_KMiss', 1)
            with ff.timer('t_kernelLoad', getProf(), final=False):
                if spec.libPath not in self.libs:
                    self.libs[spec.libPath] = cuda.module_from_file(str(spec.libPath))

                nBuf = len(spec.inputs) + len(spec.temps) + len(spec.uniqueOutputs)
                litTypes = [lit.t for lit in spec.literals]
                self.kerns[spec.name] = kaasFunc(self.libs[spec.libPath], spec.kernel, litTypes, nBuf)
        else:
            updateProf('n_KHit', 1)

        return self.kerns[spec.name]


class lruPolicy():
    def __init__(self):
        # Only contains buffers that are currently on the device. Constants are
        # given higher priority than other types of buffers.
        self.lruConst = collections.deque()
        self.lruOther = collections.deque()

    def remove(self, buf):
        """Remove buffer from consideration"""
        if buf.const:
            self.lruConst.remove(buf)
        else:
            self.lruOther.remove(buf)

    def push(self, buf):
        """Place buffer in the most-recently-used slot"""
        if buf.const:
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

        if b.pin:
            raise RuntimeError(f"Attempting to evict pinned buffer: name {b.name}, key {b.key}, eph {b.ephemeral}")

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
            updateProf('n_devDEvict', 1)
            # Pull from general pool first, only touch const if you have to
            b = self.policy.pop()
            logging.debug("Evicting " + b.name)

            if b.dirty:
                updateProf('s_devDWriteBack', b.size())
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
            if bSpec.key in self.bufs:
                if not self.bufs[bSpec.key].onDevice:
                    updateProf('n_devDMiss', 1)
                    total += bSpec.size
                else:
                    updateProf('n_devDHit', 1)
            else:
                updateProf('n_devDMiss', 1)
                total += bSpec.size

        with ff.timer('t_devDEvict', getProf(), final=False):
            self.makeRoom(total)

    def load(self, bSpec, overwrite=False):
        """Load a buffer onto the device, fetching from the KV store if
        necessary. If overwrite=True, a new buffer will be created and any
        existing value in the KV store will be overwritten. If the buffer is
        already in the cache and dirty, it will be written back and re-read
        (you should probably make sure this doesn't happen by flushing when
        needed)."""

        if bSpec.key in self.bufs:
            logging.debug("Loading from Cache: {}".format(bSpec.name))
            updateProf('n_hostDHit', 1)
            buf = self.bufs[bSpec.key]

            if overwrite:
                buf.clear()

            # Reset LRU
            if buf.onDevice:
                self.policy.remove(buf)
        else:
            updateProf('n_hostDMiss', 1)
            if bSpec.ephemeral or overwrite:
                logging.debug("Loading (new buffer): {}".format(bSpec.name))
                buf = kaasBuf.fromSpec(bSpec)

                if buf.ephemeral:
                    self.ephemerals[bSpec.key] = buf
            else:
                with ff.timer('t_hostDLoad', getProf(), final=False):
                    raw = self.kv.get(bSpec.key, profile=getProf(mod='kv'), profFinal=False)
                if raw is None:
                    logging.debug("Loading (new buffer): {}".format(bSpec.name))
                    buf = kaasBuf.fromSpec(bSpec)
                else:
                    logging.debug("Loading from KV: {} (key: {})".format(bSpec.name, bSpec.key))
                    updateProf('s_hostDLoad', bSpec.size)
                    buf = kaasBuf.fromSpec(bSpec, raw)

        # This is mostly a safety check, we should have already made enough
        # room
        self.makeRoomForBufs([buf])

        self.size += buf.toDevice()

        self.bufs[bSpec.key] = buf
        self.policy.push(buf)

        return buf

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
                logging.debug("Writing back to kv: {} (key: {})".format(buf.name, buf.key))
                updateProf('n_hostDWriteBack', 1)
                with ff.timer('t_hostDWriteBack', getProf(), final=False):
                    self.kv.put(buf.key, np.asarray(buf.hbuf), profile=getProf(mod='kv'), profFinal=False)
            buf.dirty = False

    def flush(self):
        for b in self.dirtyBufs.values():
            self._flushOne(b)

        self.dirtyBufs = {}

    def drop(self, key):
        """Remove a buffer from the cache (writing back if dirty). This frees
        any device memory and drops references to the host buffer (Python's GC
        will pick it up eventually)."""
        logging.debug("Dropping " + str(key))
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
    allBSpecs = {}
    for kSpec in req.kernels:
        for bSpec in kSpec.arguments:
            allBSpecs[bSpec.key] = bSpec
    with ff.timer("t_makeRoom", profs, final=False):
        bCache.makeRoomForBufs(allBSpecs.values())

    invokeTimes = []
    for kSpec in req.kernels:
        kern = kCache.get(kSpec)

        # The user should ensure that all buffers will fit on the device.
        # Invoke() will catch the mistake if they don't.
        arguments = []
        for i in range(len(kSpec.arguments)):
            arg = kSpec.arguments[i]

            if kSpec.type_list[i] == 'o':
                argBuf = bCache.load(arg, overwrite=True)
            else:
                argBuf = bCache.load(arg)

            # Prevent buffers needed by the current request from being evicted
            # This shouldn't happen anyway if LRU is working correctly.
            argBuf.pin = True

            arguments.append(argBuf)

        with ff.timer("t_invokeExternal", profs, final=False):
            timer = kern.Invoke(kSpec.literals, arguments, kSpec.gridDim, kSpec.blockDim, kSpec.sharedSize)
            profSync()
        invokeTimes.append(timer)

        # Enable the bCache to evict arguments if needed
        for arg in arguments:
            arg.pin = False

        # Inform the bCache that the output buffers are dirty and need to be
        # committed on eviction.
        for o in kSpec.outputs:
            if not o.ephemeral:
                bCache.dirty(o.key)

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

    if profLevel >= 1:
        for metric in eventMetrics:
            profs[metric].increment()
        for p in profs.mod('kv').values():
            p.increment()

        t_invoke = 0
        for t in invokeTimes:
            t_invoke += t()
        profs['t_invoke'].increment(t_invoke*1000)

    return {}


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
