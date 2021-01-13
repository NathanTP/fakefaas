import pathlib
import numpy as np

import libff as ff
import libff.kv
import libff.invoke


redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"
testPath = pathlib.Path(__file__).resolve().parent

kernsDir = testPath.parent / 'kerns'
mmKern = "sgemm"
colMajor = False

DEVICE_MEM_CAP = 1024*1024*1024*4
# tileN = 32
# tile_tb_height = 16 
tile_tb_height = 8
tileN = 16
tileM = (tileN * tile_tb_height)

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


    def __eq__(self, other):
        if not isinstance(other, mmShape):
            return False

        if self.M != other.M or self.N != other.N or self.K != other.K:
            return False

        return True


rng = np.random.default_rng(0)
def generateArr(shape):
    arr = rng.random(shape, dtype=np.float32)
    if colMajor:
        arr = np.asfortranarray(arr)

    return arr


def getData(ffCtx, name, shape):
    raw = ffCtx.kv.get(name)
    arr = np.frombuffer(raw, dtype=np.float32)
    if colMajor:
        arr = arr.reshape(shape[0], shape[1], order="F")
    else:
        arr = arr.reshape(shape[0], shape[1])
    
    return arr

def poissonDelay(target):
    """Returns a poisson distribution generator suitable for benchClient
    initalization. Target is the desired mean latency in ms."""
    rng = np.random.default_rng()
    return lambda: rng.poisson(target)


def zipfDelay(a, scale=1, maximum=None):
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
        

def sideLenFromSize(depth, nbyte):
    """Returns the closest legal sideLen that would use at most nbyte bytes of
    arrays for depth."""
    # 1 mtx for the input, depth matrices for the static, depth matrices for
    # the outputs
    nMatrix = 1 + (2*depth)

    # Biggest sidelen that would fit in nbyte
    sideLen = math.isqrt((nbyte // 4) // nMatrix)

    if sideLen < tileM:
        # it's impossible
        return None
   
    # Round down to the nearest multiple of tileM
    sideLen -= sideLen % tileM

    return sideLen


def sizeFromSideLen(depth, sideLen):
    """Returns the number of bytes of device memory that will be needed for
    the given depth and matrix side length (in elements)"""
    nMatrix = 1 + (2*depth)
    return (nMatrix * (sideLen**2)) * 4
