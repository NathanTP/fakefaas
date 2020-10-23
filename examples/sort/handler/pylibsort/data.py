import pathlib
import os
import re
import abc
import ctypes
import shutil
import itertools
import operator
import json
import numpy as np

# ol-install: numpy

from . import __state

FileDistribArrayMount = pathlib.Path("/shared")

class DistribArrayError(Exception):
    def __init__(self, cause):
        self.cause = cause


    def __str__(self):
        return self.cause


def SetDistribMount(newRoot: pathlib.Path):
    """Change the default mount point for File distributed arrays to newRoot.
    It is not necessary to call this, the default is '/shared'"""
    global FileDistribArrayMount
    FileDistribArrayMount = newRoot


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

    
class DistribArray(abc.ABC):
    # An ArrayShape describing the lengths and capacities of the partitions in this array
    shape = None

    @abc.abstractmethod
    def Close(self):
        """Commit any outstanding changes to the backing store. The current python
        object is no longer valid but the persistent backing object may be re-opened"""
        pass
    

    @abc.abstractmethod
    def Destroy(self):
        """Completely remove the array from the backing store"""
        pass


    @abc.abstractmethod
    def ReadPart(self, partID, start=0, nbyte=-1):
        """Read nbyte bytes from the partition starting from offset 'start'. If
        nbyte is -1, read the entire remaining partition after start."""
        pass


    @abc.abstractmethod
    def WritePart(self, partID, buf):
        """Append the contents of buf to partID. The partition must have
        sufficient remaining capacity to store buf."""
        pass


    @abc.abstractmethod
    def ReadAll(self):
        """Returns the entire contents of the array in a single buffer. The
        returned buffer will match the total reserved capacity of the array,
        use the shape attribute to determine partition boundaries and the valid
        portions of each partition."""
        pass


    @abc.abstractmethod
    def WriteAll(self, buf):
        """Completely overwrite an array with the contents of buf. Each
        partition is assumed to use its entire capacity."""
        pass


class fileDistribArray(DistribArray):
    """A distributed array that stores its data in the filesystem. If the
    provided path already exists, it is used directly, otherwise a directory is
    created for the new array. If the array already exists, the npart argument
    is ignored."""
    shape = None

    def __init__(self, rootPath):
        """Prepare the array for opening or creation. You almost certainly
        don't want to call this directly, use Open or Create instead"""
        self.rootPath = pathlib.Path(rootPath)
        self.datPath = self.rootPath / 'data.dat'
        self.metaPath = self.rootPath / 'meta.json'
        self.closed = False


    def __commitMeta(self):
        with open(self.metaPath, 'w') as metaF:
            jsonShape = {"Lens" : self.shape.lens, "Caps" : self.shape.caps}
            json.dump(jsonShape, metaF)


    @classmethod
    def Create(cls, rootPath, shape: ArrayShape):
        arr = cls(rootPath)

        # These need open permissions because of docker user mismatches (docker
        # will use root so the host can't re-open the file).
        arr.rootPath.mkdir(0o777)
        arr.datPath.touch(0o666)
        arr.metaPath.touch(0o666)

        arr.shape = ArrayShape(lens=shape.lens.copy(), caps=shape.caps.copy())
        
        arr.dataF = open(arr.datPath, 'r+b')

        return arr

    @classmethod
    def Open(cls, rootPath):
        arr = cls(rootPath)

        if not arr.rootPath.exists():
            raise DistribArrayError("Array {} does not exist".format(rootPath))

        with open(arr.metaPath, 'r') as metaF:
            jsonShape = json.load(metaF)
            arr.shape = ArrayShape(lens = jsonShape['Lens'], caps = jsonShape['Caps'])
        
        arr.dataF = open(arr.datPath, 'r+b')

        return arr


    def Close(self):
        # Being idempotent just makes things easier
        if not self.closed:
            self.dataF.close()
            self.__commitMeta()
            self.closed = True


    def Destroy(self):
        shutil.rmtree(self.rootPath)


    def ReadPart(self, partID, start=0, nbyte=-1, dest=None):
        """Read partition 'partID' of this array. If 'dest' is specified, the
        data is read into that bytearray or memoryview, otherwise a new
        bytearray is returned"""
        if nbyte == -1:
            nbyte = self.shape.lens[partID] - start

        if start > self.shape.lens[partID] or start+nbyte > self.shape.lens[partID]:
            raise DistribArrayError("Read beyond end of partition {} (asked for {}+{}, limit {}".format(partID, start, nbyte, self.shape.lens[partID])) 

        self.dataF.seek(self.shape.starts[partID] + start)

        if dest is None:
            return bytearray(self.dataF.read(nbyte))
        else:
            self.dataF.readinto(dest[:nbyte])


    def WritePart(self, partId, buf):
        if self.shape.lens[partId] + len(buf) > self.shape.caps[partId]:
            raise DistribArrayError("Wrote beyond end of partition (asked for {}b, limit {}b)".format(len(buf),
                self.shape.caps[partId] - self.shape.lens[partId]))

        self.dataF.seek(self.shape.starts[partId] + self.shape.lens[partId])
        self.dataF.write(buf)
        self.shape.lens[partId] += len(buf)


    def ReadAll(self):
        """Returns the entire contents of the array in a single buffer. The
        returned buffer will match the total reserved capacity of the array,
        use the shape attribute to determine partition boundaries and the valid
        portions of each partition."""
        self.dataF.seek(0)
        # return bytearray(self.dataF.read())
        return memoryview(self.dataF.read())


    def WriteAll(self, buf):
        """Completely overwrite an array with the contents of buf. Each
        partition is assumed to use its entire capacity."""
        totalCap = sum(self.shape.caps)
        if len(buf) != totalCap:
            raise DistribArrayError("Buffer length {}b does not match array capacity {}b".format(len(buf), totalCap))

        self.dataF.seek(0)
        self.dataF.write(buf)

        self.shape.lens = self.shape.caps.copy()
        

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

def __fileGetRef(req) -> partRef:
    """Return a partRef from an entry in the 'input' field of a req"""
    arr = None
    if req['arrayName'] in openArrs:
        arr = openArrs[req['arrayName']]
    else:
        arr = fileDistribArray.Open(FileDistribArrayMount / req['arrayName'])
        openArrs[req['arrayName']] = arr

    # Internally, it's easier to work with absolute numbers, so we convert -1
    # nbyte's from the request into their real value
    nbyte = req['nbyte']
    if nbyte == -1:
        nbyte = arr.shape.lens[req['partID']]

    return partRef(arr, partID=req['partID'], start=req['start'], nbyte=nbyte)


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


def getPartRefs(req: dict):
    """Returns a list of partRefs from a sort request dictionary."""

    if req['arrType'] == "file":
        return [__fileGetRef(r) for r in req['input']]
    else:
        raise ValueError("Invalid request type: " + str(req['arrType']))


def __fileGetOutputArray(req, shape: ArrayShape) -> fileDistribArray:
    return fileDistribArray.Create(FileDistribArrayMount / req['output'], shape)

def getOutputArray(req: dict):
    """Returns a FileDistribArray to use for the output of req"""

    if req['arrType'] == "file":
        return __fileGetOutputArray(req)
    else:
        raise ValueError("Invalid request type: " + str(req['arrType']))


def writeOutput(req: dict, rawBytes, boundaries):
    caps = np.array(boundaries)
    caps *= 4
    caps = np.diff(caps, append=len(rawBytes))
    
    shape = ArrayShape.fromCaps(caps.tolist())
    outArr = fileDistribArray.Create(FileDistribArrayMount / req['output'], shape)
    outArr.WriteAll(rawBytes)
    outArr.Close()


# Generates n random integers and returns a bytearray of them
def generateInputs(n):
    b = bytearray(n*4)
    cInts = (ctypes.c_uint32*n).from_buffer(b)
    __state.sortLib.populateInput(cInts, n)
    return b
