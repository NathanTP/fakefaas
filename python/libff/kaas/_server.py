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
#   - 2 All metrics. This may have a performance impact, particularly for small kernels and/or data sizes.
profLevel = 2

# This is a global reference to the current ff.profCollection (reset for each call)
profs = None

kCache = None
bCache = None


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
eventMetrics1 = [
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
        't_hostMM'
        ]

eventMetrics2 = [
        't_kernel'
        ]


class kaasBuf():
    @classmethod
    def fromSpec(cls, spec, src=None):
        return cls(spec.name, src, size=spec.size, const=spec.const, ephemeral=spec.ephemeral)

    def __init__(self, name, src, size=None, const=False, ephemeral=False):
        """A kaasBuffer represnts a binary data buffer managed by the kaas
        system, it can be either on the device or on the host. If src is
        provided, it will be used as the buffer, otherwise size will be used to
        represent a zeroed buffer. kaasBuffers do not automatically manage
        consistency, it is up to the user to decide when and how to synchronize
        host and device memory.
        """
        self.name = name
        self.dirty = False
        self.const = const
        self.ephemeral = ephemeral

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
        return "KaaS Buffer (name={}, dirty={}, const={}, ephemeral={}, onDevice={}, size={})".format(
                self.name, self.dirty, self.const, self.ephemeral, self.onDevice, self.size)

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
        host buffer is set, zeroed device memory will be allocated."""
        if self.onDevice:
            return
        else:
            logging.debug("Moving {} to device".format(self.name))
            updateProf('s_htod', self.size)

            with ff.timer('t_cudaMM', getProf(), final=False):
                self.dbuf = cuda.mem_alloc(self.size)

            if self.hbuf is None:
                logging.debug("Zeroing {} ({})".format(self.name, hex(int(self.dbuf))))
                with ff.timer('t_zero', getProf(), final=False):
                    cuda.memset_d8(self.dbuf, 0, self.size)

            else:
                with ff.timer('t_htod', getProf(), final=False):
                    cuda.memcpy_htod(self.dbuf, self.hbuf)

            self.onDevice = True

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

    def freeDevice(self):
        """Free any memory that is allocated on the device."""
        if not self.onDevice:
            return
        else:
            with ff.timer('t_cudaMM', getProf(), final=False):
                self.dbuf.free()
            self.dbuf = None
            self.onDevice = False

    def clear(self):
        logging.debug("Clearing existing buffer " + self.name)
        with ff.timer('t_zero', getProf(), final=False):
            if self.onDevice:
                cuda.memset_d8(self.dbuf, 0, self.size)
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
        if buf.const:
            self.lruConst.appendleft(buf)
        else:
            self.lruOther.appendleft(buf)

    def pop(self):
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

    def __init__(self, cap):
        """Initialize a device buffer cache with byte capacity cap. If this cap
        is exceeded, things will be evicted"""
        # Contains all bufs, on device or not
        self.bufs = {}
        self.dirtyBufs = {}
        self.ephemerals = {}

        self.policy = lruPolicy()

        # Maximum number of bytes on the device
        self.cap = cap

        # Current number of bytes on the device
        self.size = 0

    def setKV(self, kv):
        self.kv = kv

    def _makeRoom(self, buf):
        if buf.onDevice:
            updateProf('n_devDHit', 1)
        else:
            updateProf('n_devDMiss', 1)
            with ff.timer('t_devDEvict', getProf(), final=False):
                while self.cap - self.size < buf.size:
                    updateProf('n_devDEvict', 1)
                    # Pull from general pool first, only touch const if you have to
                    b = self.policy.pop()
                    logging.debug("Evicting " + b.name)

                    if b.dirty:
                        updateProf('s_devDWriteBack', b.size())
                        b.toHost()

                    b.freeDevice()
                    self.size -= b.size()

    def load(self, bSpec, overwrite=False):
        """Load a buffer onto the device, fetching from the KV store if
        necessary. If overwrite=True, a new buffer will be created and any
        existing value in the KV store will be overwritten. If the buffer is
        already in the cache and dirty, it will be written back and re-read
        (you should probably make sure this doesn't happen by flushing when
        needed)."""

        if not bSpec.const and not bSpec.ephemeral:
            logging.debug("Invalidating: {}".format(bSpec.name))
            # Refetch even if we already have it
            if bSpec.name in self.bufs:
                self.drop(bSpec.name)

        if bSpec.name in self.bufs:
            logging.debug("Loading from Cache: {}".format(bSpec.name))
            updateProf('n_hostDHit', 1)
            buf = self.bufs[bSpec.name]

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
                    self.ephemerals[bSpec.name] = buf
            else:
                with ff.timer('t_hostDLoad', getProf(), final=False):
                    raw = self.kv.get(bSpec.name, profile=getProf(mod='kv'), profFinal=False)
                if raw is None:
                    logging.debug("Loading (new buffer): {}".format(bSpec.name))
                    buf = kaasBuf.fromSpec(bSpec)
                else:
                    logging.debug("Loading from KV: {}".format(bSpec.name))
                    updateProf('s_hostDLoad', bSpec.size)
                    buf = kaasBuf.fromSpec(bSpec, raw)

        self._makeRoom(buf)
        buf.toDevice()

        self.bufs[bSpec.name] = buf
        self.policy.push(buf)

        return buf

    def dirty(self, name):
        buf = self.bufs[name]
        buf.dirty = True
        self.dirtyBufs[name] = buf

    def _flushOne(self, buf):
        if buf.dirty:
            if buf.onDevice:
                buf.toHost()
            if not buf.ephemeral:
                # Data are stored as numpy arrays because memoryviews can't be
                # pickled. This should still be zero copy.
                logging.debug("Writing back to kv: {}".format(buf.name))
                updateProf('n_hostDWriteBack', 1)
                with ff.timer('t_hostDWriteBack', getProf(), final=False):
                    self.kv.put(buf.name, np.asarray(buf.hbuf), profile=getProf(mod='kv'), profFinal=False)
            buf.dirty = False

    def flush(self):
        for b in self.dirtyBufs.values():
            self._flushOne(b)

        self.dirtyBufs = {}

    def clearEphemerals(self):
        for b in list(self.ephemerals.values()):
            self.drop(b.name)

        self.ephemerals = {}

    def drop(self, name):
        """Remove a buffer from the cache (writing back if dirty). This frees
        any device memory and drops references to the host buffer (Python's GC
        will pick it up eventually)."""
        logging.debug("Dropping " + name)
        buf = self.bufs[name]
        self._flushOne(buf)
        buf.freeDevice()

        if buf.dirty:
            self.dirtyBufs.pop(name, None)
        if buf.ephemeral:
            self.ephemerals.pop(name, None)

        self.policy.remove(buf)

        del self.bufs[name]


def kaasServeInternal(req, ctx):
    """Internal implementation of kaas execution. Req is a kaas.kaasReq, not a
    dictionary"""

    # These should only get initialized upon invocation, not at import. This is
    # most important for the kCache which has to manage a cuda device context
    global kCache, bCache
    if kCache is None:
        kCache = kernelCache()

    if bCache is None:
        # I think the device has 4GB
        bCache = bufferCache(1024*1024*1024*4)

    # This gets reset every call because libff only gives us the kv handle per
    # call and it could (in theory) change in some way between calls.
    bCache.setKV(ctx.kv)

    global profs
    profs = ctx.stats

    invokeTimes = []
    for kSpec in req.kernels:
        kern = kCache.get(kSpec)

        # The user should ensure that all buffers will fit on the device.
        # Invoke() will catch the mistake if they don't.
        inputs = [bCache.load(b) for b in kSpec.inputs]
        temps = [bCache.load(b) for b in kSpec.temps]
        outputs = [bCache.load(b, overwrite=True) for b in kSpec.uniqueOutputs]

        timer = kern.Invoke(kSpec.literals,
                            inputs + temps + outputs,
                            kSpec.gridDim, kSpec.blockDim, kSpec.sharedSize)
        invokeTimes.append(timer)

        # Inform the bCache that the output buffers are dirty and need to be
        # committed on eviction.
        for o in kSpec.outputs:
            if not o.ephemeral:
                bCache.dirty(o.name)

        # Don't bother waiting for the caching policy on temps, they will for
        # sure never be needed again. In the future we may avoid this to save
        # on cudaMalloc.
        for t in kSpec.temps:
            bCache.drop(t.name)

    # Make sure all outputs are visible externally (basically this merges our
    # private state into whatever consistency properties the KV gives us.
    bCache.flush()
    bCache.clearEphemerals()

    if profLevel >= 1:
        for metric in eventMetrics1:
            profs[metric].increment()
        for p in profs.mod('kv').values():
            p.increment()

        t_invoke = 0
        for t in invokeTimes:
            t_invoke += t()
        profs['t_invoke'].increment(t_invoke*1000)

    if profLevel >= 2:
        for metric in eventMetrics2:
            profs[metric].increment()

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
