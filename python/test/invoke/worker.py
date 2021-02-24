import sys
import time
import random
import numpy as np

import libff

def echo(req, ctx):
    return req


def state(req, ctx):
    if 'scratchData' in req:
        ctx.scratch = req['scratchData']

    return {"cachedData" : ctx.scratch}


random.seed(time.time())
def perfSim(req, ctx):
    """Sleeps for req['runtime'] ms and returns some stats"""
    # You can time stuff
    with libff.timer('runtime', ctx.profs):
        time.sleep(req['runtime'] / 1000)

    # You can also record non-time stuff using libff.prof() objects
    if 'randSample' not in ctx.profs:
        ctx.profs['randSample'] = libff.prof()
    randomMetric = random.randint(0,1000)
    ctx.profs['randSample'].increment(randomMetric)

    # returns the random metric to aid in testing
    return {"validateMetric" : randomMetric}


def cuda(req, ctx):
    import pycuda.driver as cuda
    import pycuda.tools
    from pycuda.compiler import SourceModule

    cuda.init()
    if ctx.cudaDev is not None:
        dev = cuda.Device(ctx.cudaDev)
        cudaCtx = dev.make_context()
    else:
        raise RuntimeError("No CUDA device specified")

    a = np.random.randn(4,4)
    a = a.astype(np.float32)

    ad = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(ad, a)

    mod = SourceModule("""
      __global__ void doublify(float *a)
      {
	int idx = threadIdx.x + threadIdx.y*4;
	a[idx] *= 2;
      }
      """) 

    func = mod.get_function("doublify")
    func(ad, block=(4,4,1))

    a_doubled = np.empty_like(a)
    cuda.memcpy_dtoh(a_doubled, ad)
    ad.free()

    cudaCtx.detach()

    if not np.allclose(a_doubled, a*2):
        raise RuntimeError("CUDA test gave wrong answer")

    return {"status" : "SUCESS", "deviceID" : dev.pci_bus_id()}


funcMap = {"echo" : echo, "perfSim" : perfSim, 'state' : state, 'cuda': cuda}

def LibffInvokeRegister():
    return funcMap 

if __name__ == "__main__":
    import libff.invoke
    libff.invoke.RemoteProcessServer(funcMap, sys.argv[1:])
