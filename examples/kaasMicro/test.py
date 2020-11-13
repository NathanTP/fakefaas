import sys
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule, DynamicModule
import numpy as np
import ctypes
import ctypes.util
import pathlib

class KaasError(Exception):
    def __init__(self, cause):
        self.cause = cause


    def __str__(self):
        return self.cause


# Example of using pyCuda to run a kernel. I can't figure out how to get pyCuda
# to load from a shared library though, it seems to require jitting the
# kernels...
def pyCudaNative():
    a = np.random.randn(4,4)
    a = a.astype(np.float32)
    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)

    mod = SourceModule("""
      __global__ void doublify(float *a)
      {
	int idx = threadIdx.x + threadIdx.y*4;
	a[idx] *= 2;
      }
      """)
    func = mod.get_function("doublify")
    func(a_gpu, block=(4,4,1))

    a_doubled = np.empty_like(a)
    cuda.memcpy_dtoh(a_doubled, a_gpu)
    print(a_doubled)
    print(a)


class kaasBuf():
    def __init__(self, src, size=None):
        """A kaasBuffer represnts a binary data buffer managed by the kaas
        system, it can be either on the device or on the host. If src is
        provided, it will be used as the buffer, otherwise size will be used to
        represent a zeroed buffer. kaasBuffers do not automatically manage
        consistency, it is up to the user to decide when and how to synchronize
        host and device memory.
        """
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

    def fromDevice(self):
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
    def __init__(self, libPath, fName, nBuf):
        self.argType = ctypes.c_void_p * nBuf 

        self.lib = ctypes.cdll.LoadLibrary(libPath)
        self.func = getattr(self.lib, fName)
        self.func.argtypes = [ ctypes.c_int, ctypes.c_int, self.argType ]

    def Invoke(self, gridDim, blkDim, bufs, outs):
        """Invoke the func with the provided diminsions:
            - bufs is a list of kaasBuf objects
            - outs is a list of indexes of bufs to copy back to host memory
              after invocation
        """
        dAddrs = []
        for b in bufs:
            b.toDevice()
            dAddrs.append(ctypes.cast(int(b.dbuf), ctypes.c_void_p))

        self.func(gridDim, blkDim, self.argType(*dAddrs))

        for outX in outs:
            bufs[outX].fromDevice()


# Using ctypes to call a wrapper. In theory I guess I could generate the
# wrapper in ctypes and JIT it but meh.
def kaasInvoke():
    packagePath = pathlib.Path(__file__).parent
    libPath = packagePath / "kerns" / "libkaasMicro.so"

    func = kaasFunc(libPath, 'doublify', 1)

    a = np.random.randn(4,4)
    a = a.astype(np.float32)
    print("Orig:")
    print(a)

    inputs = [ kaasBuf(a) ]
    func.Invoke(1, 4, inputs, [0])

    print("Doubled:")
    print(a)

    # Cleanup (free any memory)
    for i in inputs:
        i.freeDevice()
        i.setHostBuffer(None)

kaasInvoke()
