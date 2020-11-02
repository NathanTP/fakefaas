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

FileMount = None

class ArrayError(Exception):
    def __init__(self, cause):
        self.cause = cause


    def __str__(self):
        return self.cause


def SetFileMount(newRoot: pathlib.Path):
    """Change the default mount point for File distributed arrays to newRoot.
    It is not necessary to call this, the default is '/shared'"""
    global FileMount
    FileMount = pathlib.Path(newRoot)


class Array(abc.ABC):
    """A distributed array, it can be of arbitrary size and
    will be available remotely (indexed by name)."""


    @abc.abstractmethod
    def __init__(self, name, noCreate=False):
        """Create a new local Array reference. If noCreate is True, the array
        must already exist, otherwise a new Array will be created if it does
        not already exist."""
        pass


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
    def Read(self, start=0, nbyte=-1, dest=None):
        """Read nbytes from the array starting from offset 'start'. If nbyte is
        -1, the whole array is read. If dest is provided, the data will be read
        into dest (a mutable bytes-like object)."""
        pass


    @abc.abstractmethod
    def Write(self, buf, start=0):
        """Write the contents of buf to the array starting at start. Starts
        after the end of the array will be zero filled. The array will be
        extended as needed."""
        pass


class FileArray(Array):
    """An Array backed by a file, users must call SetFileMount() before using
    FileArrays. Files have practical and performance implications. Arrays can
    be large (limited by your disk space), but having many small arrays is
    probably not ideal and may stress your OS. Be careful. Note that each open
    array implies a file handle, there are OS limits to how many open files you
    can have. Best to close arrays aggresively."""

    def __init__(self, name, noCreate=False):
        """Create a new local Array reference. If noCreate is True, the array
        must already exist, otherwise a new Array will be created if it does
        not already exist."""
        self.rootPath = FileMount / name
        self.datPath = self.rootPath / 'data.dat'

        if not self.datPath.exists():
            if noCreate:
                raise ArrayError("Array {} does not exist".format(self.rootPath))
            else:
                # These need open permissions because of docker user mismatches (docker
                # will use root so the host can't re-open the file).
                self.rootPath.mkdir(0o777)
                self.datPath.touch(0o666)

        self.dataF = open(self.datPath, 'r+b')
        self.cap = self.datPath.stat().st_size
        self.closed = False


    def Close(self):
        # Being idempotent just makes things easier
        if not self.closed:
            self.dataF.close()
            self.closed = True


    def Destroy(self):
        shutil.rmtree(self.rootPath)

    def Read(self, start=0, nbyte=-1, dest=None):
        if start >= self.cap:
            raise ArrayError("Read beyond end of array: start location ({}) beyond end of array ({})".format(start, self.cap))


        if nbyte == -1:
            nbyte = self.cap - start

        if start + nbyte > self.cap:
            raise ArrayError("Read beyond end of array: requested too many bytes ({})".format(nbyte))

        self.dataF.seek(start) 
        if dest is None:
            return bytearray(self.dataF.read(nbyte))
        else:
            self.dataF.readinto(dest[:nbyte])


    def Write(self, buf, start=0):
        self.dataF.seek(start)
        self.dataF.write(buf)
        self.cap += len(buf)
