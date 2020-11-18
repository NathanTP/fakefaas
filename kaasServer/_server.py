import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule, DynamicModule
import ctypes
import ctypes.util
import sys
import numpy as np
import pathlib
import collections

import libff as ff
import libff.kv

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
                raise KaasError("New buffer is not big enough")

            self.hbuf = newBuf


    def toDevice(self):
        """Place the buffer onto the device if it is not already there.  If no
        host buffer is set, zeroed device memory will be allocated."""
        if self.onDevice:
            return
        else:
            self.dbuf = cuda.mem_alloc(self.size)

            if self.hbuf is None:
                cuda.memset_d8(self.dbuf, 0, self.size)
            else:
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


class kaasFunc():
    def __init__(self, library, fName, nBuf):
        self.argType = ctypes.c_void_p * nBuf 

        self.lib = library 
        self.func = getattr(self.lib, fName)
        self.func.argtypes = [ ctypes.c_int, ctypes.c_int, self.argType ]

    
    def __str__(self):
        return self.func

    def Invoke(self, gridDim, blkDim, bufs):
        """Invoke the func with the provided diminsions:
            - bufs is a list of kaasBuf objects
            - outs is a list of indexes of bufs to copy back to host memory
              after invocation
        """
        dAddrs = []
        for b in bufs:
            if not b.onDevice:
                raise RuntimeError("Provided buffer was not resident on the device")

            dAddrs.append(ctypes.cast(int(b.dbuf), ctypes.c_void_p))

        self.func(gridDim, blkDim, self.argType(*dAddrs))


class kernelCache():
    def __init__(self):
        self.libs = {}
        self.kerns = {}

    def get(self, spec):
        if spec.name not in self.kerns:
            if spec.libPath not in self.libs:
                self.libs[spec.libPath] = ctypes.cdll.LoadLibrary(spec.libPath)

            nBuf = len(spec.inputs) + len(spec.temps) + len(spec.uniqueOutputs)

            self.kerns[spec.name] = kaasFunc(self.libs[spec.libPath], spec.kernel, nBuf)

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

        self.policy = lruPolicy()

        # Maximum number of bytes on the device
        self.cap = cap

        # Current number of bytes on the device
        self.size = 0


    def setKV(self, kv):
        self.kv = kv


    def _makeRoom(self, buf):
        if buf.onDevice:
            return

        while self.cap - self.size < buf.size:
            # Pull from general pool first, only touch const if you have to
            b = self.policy.pop()

            if b.dirty:
                b.toHost()

            b.freeDevice()
            self.size -= b.size()


    def load(self, bSpec):
        """Load a buffer onto the device, fetching from the KV store if
        necessary. If the buffer is already in the cache and dirty, it will be
        written back and re-read (you should probably make sure this doesn't
        happen by flushing when needed)."""
        if not bSpec.const and not bSpec.ephemeral:
            # Refetch even if we already have it
            if bSpec.name in self.bufs:
                self.drop(bSpec.name)

        if bSpec.name in self.bufs:
            buf = self.bufs[bSpec.name]

            # Reset LRU
            if buf.onDevice:
                self.policy.remove(buf)
        else:
            if bSpec.ephemeral:
                buf = kaasBuf.fromSpec(bSpec)
            else:
                raw = self.kv.get(bSpec.name)
                if raw is None:
                    buf = kaasBuf.fromSpec(bSpec)
                else:
                    buf = kaasBuf.fromSpec(bSpec, raw)

        self._makeRoom(buf)
        buf.toDevice()

        self.bufs[bSpec.name] = buf
        return buf


    def dirty(self, name):
        self.bufs[name].dirty = True


    def _flushOne(self, buf):
        if buf.dirty:
            if buf.onDevice:
                buf.toHost()
            if not buf.ephemeral:
                self.kv.put(buf.name, buf.hbuf)
            buf.dirty = False


    def flush(self):
        for b in self.bufs.values():
            self._flushOne(b)


    def drop(self, name):
        """Remove a buffer from the cache. This frees any device memory and
        drops references to the host buffer (Python's GC will pick it up
        eventually)."""
        buf = self.bufs[name]
        self._flushOne(buf)
        buf.freeDevice()

        self.policy.remove(buf)

        del self.bufs[name]


kCache = kernelCache()

# I think the device has 4GB, I'll keep it conservative for now
bCache = bufferCache(1024*1024*2)

def kaasServeInternal(req, ctx):
    """Internal implementation of kaas execution. Req is a kaas.kaasReq, not a
    dictionary"""

    # This gets reset every call because libff only gives us the kv handle per
    # call and it could (in theory) change in some way between calls.
    bCache.setKV(ctx.kv)

    for kSpec in req.kernels:
        kern = kCache.get(kSpec)

        # XXX Too lazy to check this for now, we assume all the buffers will fit.
        inputs = [ bCache.load(b) for b in kSpec.inputs ]
        temps = [ bCache.load(b) for b in kSpec.temps ]
        outputs = [ bCache.load(b) for b in kSpec.uniqueOutputs ]

        kern.Invoke(kSpec.nGrid, kSpec.nBlock, inputs + temps + outputs)

        # Inform the bCache that the output buffers are dirty and need to be
        # committed on eviction.
        for o in kSpec.outputs:
            bCache.dirty(o.name)

        # Don't bother waiting for the caching policy on temps, they will for
        # sure never be needed again.
        for t in kSpec.temps:
            bCache.drop(t.name)

    # Make sure all outputs are visible externally (basically this merges our
    # private state into whatever consistency properties the KV gives us.
    bCache.flush()

    return {}
