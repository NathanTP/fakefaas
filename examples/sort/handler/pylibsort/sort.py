import ctypes
import ctypes.util
import sys
import numpy as np

# from memory_profiler import profile
# import cProfile

from . import __state

class sortException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return('Sort error: {}'.format(self.msg))


class sortResultException(Exception):
    def __init__(self, idx, expect, got, msg=None):
        self.idx = idx
        self.expect = expect
        self.got = got
        self.msg = msg

    def __str__(self):
        desc = "Sorted Incorrectly at index {}, Expected {:#08x}, Got {:#08x}".format(
                self.idx, self.expect, self.got)

        if self.msg is not None:
            desc += " ({})".format(self.msg)

        return desc

# Returns the group ID of integer v for width group bits starting at pos
def groupBits(v, pos, width):
    return ((v >> pos) & ((1 << width) - 1))


def bytesToInts(barr):
    """Convert bytes to a list of python integers"""
    return np.frombuffer(barr, dtype=np.uint32)
    # nInt = int(len(barr) / 4)
    # respInts = []
    # for i in range(nInt):
    #     respInts.append(int.from_bytes(barr[i*4:i*4+4], sys.byteorder))
    # return respInts


def checkOrder(arr):
    """Verify that arr is in-order"""
    prev = arr[0]
    for i in range(len(arr)):
        if arr[i] < prev:
            raise sortResultException(i, prev, arr[i], "value should be >= to expected")
        prev = arr[i]


def checkSortFull(new, orig):
    """Verify that new is a correctly sorted copy of orig"""
    origSorted = sorted(orig)
    for i in range(len(orig)):
        if new[i] != origSorted[i]:
            raise sortResultException(i, origSorted[i], new[i])


def checkPartial(refBytes, testBytes, caps, pos, width):
    refInts = bytesToInts(refBytes)
    testInts = bytesToInts(testBytes)
    caps = np.array(caps)

    if len(refInts) != len(testInts):
        raise sortException("test length doesnt match reference: expected {}, got {}".format(
            len(refInts), len(testInts)))

    if len(caps) != (1 << width):
        raise sortException("Not enough output buckets: expected {}, got {}".format(
            len(caps), (1 << width)))

    # convert to word offset
    caps = caps // 4
    expectGroups = np.repeat(np.arange(len(caps)), caps)
    # boundaries = [i / 4 for i in boundaries]

    testGroups = testInts & ((1 << width) - 1)
    if not np.array_equal(testGroups, expectGroups):
        raise sortException("Output does not have expected groups")

    # Verify membership (this test is sloooow, probably best to disable for big tests)
    # if not np.all(np.isin(testInts, refInts)):
    #     raise sortException("Test does not contain same elements as ref")


def sortFull(buf: bytearray):
    """Interpret buf as an array of C uint32s and sort it in place."""
    nElem = int(len(buf) / 4)
    cRaw = (ctypes.c_uint8 * len(buf)).from_buffer(buf)
    cInt = ctypes.cast(cRaw, ctypes.POINTER(ctypes.c_uint))

    # Sort cIn in-place
    res = __state.sortLib.providedGpu(cInt, ctypes.c_size_t(nElem))

    if not res:
        raise RuntimeError("Libsort had an internal error")


# @profile
def sortPartial(buf: bytearray, offset, width):
    """Perform a partial sort of buf in place (width bits starting at bit
    offset) and return a list of the int boundaries between each radix group."""

    nElem = int(len(buf) / 4)
    cRaw = (ctypes.c_uint8 * len(buf)).from_buffer(buf)
    cInt = ctypes.cast(cRaw, ctypes.POINTER(ctypes.c_uint32))

    boundaries = (ctypes.c_uint32 * (1 << width))()

    res = __state.sortLib.gpuPartial(cInt, boundaries,
        ctypes.c_size_t(nElem),
        ctypes.c_uint32(offset),
        ctypes.c_uint32(width))

    if not res:
        raise RuntimeError("Libsort had an internal error")

    return list(boundaries)
