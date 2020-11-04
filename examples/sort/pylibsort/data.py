import pathlib
import os
import ctypes
import shutil
import itertools
import numpy as np
import pickle
import copy
from libff import array as farr

from . import __state

class DistribArrayError(Exception):
    def __init__(self, cause):
        self.cause = cause


    def __str__(self):
        return self.cause


# Ideally we'd make pylibsort more OOP as well, but for now we'll just stick
# with the global
_libffCtx = None
def ConfigureBackend(storageType, ctx):
    """Setup the backing store for pylibsort distributed arrays. ctx is a
    libff.invoke.RemoteCtx. It is safe to call this multiple times, but only
    newly created DistribArrays will be affected."""
    if storageType != "file":
        raise DistribArrayError("storage type: " + storageType + " not currently supported")

    global _libffCtx
    _libffCtx = ctx


class ArrayShape():
    def __init__(self, caps, lens):
        """You probably don't want to call this directly, use the from* methods instead"""
        self.caps = caps.copy()
        self.lens = lens.copy()
        self.npart = len(caps)
        
        # Starting location of each partition, includes a vitual partition n+1
        # to make iteration easier
        self.starts = [0]*(self.npart+1)

        total = sum(self.caps)
        self.starts[self.npart] = total
        for i in reversed(range(self.npart)):
            total -= self.caps[i]
            self.starts[i] = total
            

    @classmethod
    def fromUniform(cls, cap, npart):
        """Shape will have npart partitions, all with the same capacity"""
        return cls([cap]*npart, [0]*npart)


    @classmethod
    def fromCaps(cls, caps):
        """Explicitly provide a list of capacities"""
        return cls(caps, [0]*len(caps))

    
class DistribArray():
    # An ArrayShape describing the lengths and capacities of the partitions in this array
    shape = None


    def __init__(self, array: farr.Array, meta: farr.Array, shape: ArrayShape):
        self.array = array
        self.meta = meta

        # While open, shape is the source of truth and meta may be out of sync
        self.shape = shape

    @classmethod
    def Create(cls, baseName, shape: ArrayShape, atype='file'):
        dataArr = _libffCtx.array.Create(baseName+".dat")
        metaArr = _libffCtx.array.Create(baseName+".meta")

        metaArr.Write(pickle.dumps(shape))
        distArr = cls(dataArr, metaArr, copy.deepcopy(shape))
        
        return distArr


    @classmethod
    def Open(cls, baseName, atype='file'):
        dataArr = _libffCtx.array.Open(baseName+".dat")
        metaArr = _libffCtx.array.Open(baseName+".meta")

        shape = pickle.loads(metaArr.Read())
        distArr = cls(dataArr, metaArr, shape)

        return distArr


    def Close(self):
        """Commit any outstanding changes to the backing store. The current python
        object is no longer valid but the persistent backing object may be re-opened"""
        self.meta.Write(pickle.dumps(self.shape))

        self.array.Close()
        self.meta.Close()
    

    def Destroy(self):
        """Completely remove the array from the backing store"""
        self.array.Destroy()
        self.meta.Destroy()


    def ReadPart(self, partID, start=0, nbyte=-1, dest=None):
        """Read nbyte bytes from the partition starting from offset 'start'. If
        nbyte is -1, read the entire remaining partition after start."""

        if nbyte == -1:
            nbyte = self.shape.lens[partID] - start

        if start > self.shape.lens[partID] or start+nbyte > self.shape.lens[partID]:
            raise DistribArrayError("Read beyond end of partition {} (asked for {}+{}, limit {}".format(partID, start, nbyte, self.shape.lens[partID])) 

        partStart = self.shape.starts[partID]
        datStart = partStart + start 

        return self.array.Read(datStart, nbyte, dest)


    def WritePart(self, partID, buf):
        """Append the contents of buf to partID. The partition must have
        sufficient remaining capacity to store buf."""
        if self.shape.lens[partID] + len(buf) > self.shape.caps[partID]:
            raise DistribArrayError("Wrote beyond end of partition (asked for {}b, limit {}b)".format(len(buf),
                self.shape.caps[partID] - self.shape.lens[partID]))

        partStart = self.shape.starts[partID]
        datStart = partStart + self.shape.lens[partID] 
        self.shape.lens[partID] += len(buf)

        return self.array.Write(buf, datStart)


    def ReadAll(self):
        """Returns the entire contents of the array in a single buffer. The
        returned buffer will match the total reserved capacity of the array,
        use the shape attribute to determine partition boundaries and the valid
        portions of each partition."""
        return self.array.Read()


    def WriteAll(self, buf):
        """Completely overwrite an array with the contents of buf. Each
        partition is assumed to use its entire capacity."""
        totalCap = sum(self.shape.caps)
        if len(buf) != totalCap:
            raise DistribArrayError("Buffer length {}b does not match array capacity {}b".format(len(buf), totalCap))

        self.shape.lens = self.shape.caps.copy()
        return self.array.Write(buf)


class partRef():
    """Reference to a segment of a partition to read."""
    def __init__(self, arr: DistribArray, partID=0, start=0, nbyte=-1):
        self.arr = arr
        self.partID = partID
        self.start = start
        self.nbyte = nbyte 

    def read(self, dest=None):
        if dest is None:
            return self.arr.ReadPart(self.partID, start=self.start, nbyte=self.nbyte)
        else:
            return self.arr.ReadPart(self.partID, start=self.start, nbyte=self.nbyte, dest=dest)

# {arrayName : fileDistribArray}, minimize the number of re-opened files. This
# makes the partRefs list non-threadsafe and makes it so you have to close all
# or none, closing one may close many others.
openArrs = {}

def readPartRefs(refs):
    nTotal = 0
    for r in refs:
        nTotal += r.nbyte

    out = memoryview(bytearray(nTotal))

    loc = 0
    for r in refs:
        r.read(dest=out[loc:])
        loc += r.nbyte

    return out


def __getPartRef(req, atype):
    if req['arrayName'] in openArrs:
        arr = openArrs[req['arrayName']]
    else:
        arr = DistribArray.Open(req['arrayName'], atype=atype)

    # Internally, it's easier to work with absolute numbers, so we convert -1
    # nbyte's from the request into their real value
    nbyte = req['nbyte']
    if nbyte == -1:
        nbyte = arr.shape.lens[req['partID']]

    return partRef(arr, partID=req['partID'], start=req['start'], nbyte=nbyte)


def getPartRefs(req: dict):
    """Returns a list of partRefs from a sort request dictionary."""

    return [__getPartRef(r, req['arrType']) for r in req['input']]


def writeOutput(req: dict, rawBytes, boundaries):
    # int boundaries to byte capacities
    caps = np.diff(boundaries*4, append=len(rawBytes))
    
    shape = ArrayShape.fromCaps(caps.tolist())
    outArr = DistribArray.Create(req['output'], shape)
    outArr.WriteAll(rawBytes)
    outArr.Close()


# Generates n random integers and returns a bytearray of them
def generateInputs(n):
    b = bytearray(n*4)
    cInts = (ctypes.c_uint32*n).from_buffer(b)
    __state.sortLib.populateInput(cInts, n)
    return b
