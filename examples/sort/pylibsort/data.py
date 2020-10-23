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
import libff.array as farr

from . import __state

# # {arrayName : fileDistribArray}, minimize the number of re-opened files. This
# # makes the partRefs list non-threadsafe and makes it so you have to close all
# # or none, closing one may close many others.
openArrs = {}

def __fileGetRef(req):
    """Return a partRef from an entry in the 'input' field of a req"""
    arr = None
    if req['arrayName'] in openArrs:
        arr = openArrs[req['arrayName']]
    else:
        arr = farr.fileDistribArray.Open(req['arrayName'])
        openArrs[req['arrayName']] = arr

    # Internally, it's easier to work with absolute numbers, so we convert -1
    # nbyte's from the request into their real value
    nbyte = req['nbyte']
    if nbyte == -1:
        nbyte = arr.shape.lens[req['partID']]

    return farr.partRef(arr, partID=req['partID'], start=req['start'], nbyte=nbyte)


def getPartRefs(req: dict):
    """Returns a list of partRefs from a sort request dictionary."""

    if req['arrType'] == "file":
        return [__fileGetRef(r) for r in req['input']]
    else:
        raise ValueError("Invalid request type: " + str(req['arrType']))


def __fileGetOutputArray(req, shape):
    return farr.fileDistribArray.Create(req['output'], shape)

def getOutputArray(req: dict):
    """Returns a FileDistribArray to use for the output of req"""

    if req['arrType'] == "file":
        return __fileGetOutputArray(req)
    else:
        raise ValueError("Invalid request type: " + str(req['arrType']))


def writeOutput(req: dict, rawBytes, boundaries):
    # int boundaries to byte capacities
    caps = np.diff(boundaries*4, append=len(rawBytes))

    shape = farr.ArrayShape.fromCaps(caps.tolist())
    outArr = farr.fileDistribArray.Create(req['output'], shape)
    outArr.WriteAll(rawBytes)
    outArr.Close()


# Generates n random integers and returns a bytearray of them
def generateInputs(n):
    b = bytearray(n*4)
    cInts = (ctypes.c_uint32*n).from_buffer(b)
    __state.sortLib.populateInput(cInts, n)
    return b
