# The KaaS server (executor) is implemented as a normal libff function (for
# now) but it accepts KaaS requests from different clients. You can think of it
# like a 2-level implementation (libff for the internode and kaasServer for
# intranode). Eventually it will be its own service because scheduling
# decisions will be different.

import sys
import pathlib
import collections
from pprint import pprint

import libff as ff
import libff.invoke


serverPackage = pathlib.Path(__file__).resolve().parent

# Python's module system is garbage, I'm sure there's a 'right' way to do this
# but I can't be bothered
# print(serverPath.parent)
# sys.path.append(serverPath.parent)

from ._server import *


class KaasError(Exception):
    def __init__(self, cause):
        self.cause = cause


    def __str__(self):
        return self.cause


class bufferSpec():
    """A specification for a buffer to be used by a kernel."""

    @classmethod
    def fromDict(cls, d):
        return cls(d['name'],
            d['size'],
            d.get('ephemeral', False),
            d.get('const', False))


    def __init__(self, name, size, ephemeral=False, const=False):
        # Key to use in the kv store
        self.name = name

        # Size of the buffer in bytes
        self.size = size

        # Ephemeral buffers never leave the current node. If not ephemeral, the
        # buffer will be backed by a KV store.
        self.ephemeral = ephemeral

        # This is a hack to avoid needing to implment a proper cache.
        # Eventually, we'll integrate nicely with some proper caching system
        # like cloudburst's anna cache but just to get things working we're
        # gonna mark some inputs as constant so we don't have to reload them.
        # non-const buffers will get reloaded on every invocation.
        self.const = const


    def toDict(self):
        return {
                'name' : self.name,
                'size' : self.size,
                'ephemeral' : self.ephemeral,
                'const' : self.const
            }


    def __eq__(self, other):
        if not isinstance(other, bufferSpec):
            return NotImplemented

        return self.name == other.name


# XXX get to this later
# class scalarSpec():
#     def __init__(self, t, val):
#         if t not in ['f', 'd', 'Q']:
#             raise KaasError("Type " + str(t) + " not permitted for scalars")
#         self.t = t
#
#         if t == 'f':
#             cval = ctypes.c_float(val)
#         elif t == 'd':
#             cval = ctypes.c_double(val)
#         else:
#             cval == ctypes.c_uint64(val)
#
#         # We let python handle the sanity checking on the server side
#         self.val = cval


class kernelSpec():
    """Kernel specs describe a kernel for a particular request."""
    @classmethod
    def fromDict(cls, d):
        inputs = d.get('inputs', [])
        temps = d.get('temps', [])
        outputs = d.get('outputs', [])

        inputs  = [ bufferSpec.fromDict(b) for b in inputs ]
        temps   = [ bufferSpec.fromDict(b) for b in temps ]
        outputs = [ bufferSpec.fromDict(b) for b in outputs ]

        return cls(d['library'],
                   d['kernel'],
                   tuple(d['gridDim']),
                   tuple(d['blockDim']),
                   d['sharedSize'],
                   inputs,
                   temps,
                   outputs)


    def __init__(self, library, kernel, gridDim, blockDim, sharedSize=0, inputs=[], temps=[], outputs=[]):
        self.libPath = pathlib.Path(library).resolve()
        self.kernel = kernel
        self.name = self.libPath.stem + "." + kernel
        
        self.gridDim = gridDim
        self.blockDim = blockDim
        self.sharedSize = sharedSize 

        self.inputs = inputs 
        self.temps = temps 
        self.outputs = outputs 

        # Some outputs are also inputs, uniqueOutputs are just the new buffers
        # that have to be created for outputs
        self.uniqueOutputs = []
        for o in self.outputs:
            if o not in self.inputs:
                self.uniqueOutputs.append(o)


    def toDict(self):
        d = {}
        d['library'] = str(self.libPath)
        d['kernel'] = self.kernel
        d['gridDim'] = self.gridDim
        d['blockDim'] = self.blockDim
        d['sharedSize'] = self.sharedSize

        d['inputs'] = [ b.toDict() for b in self.inputs ]
        d['temps'] = [ b.toDict() for b in self.temps ]
        d['outputs'] = [ b.toDict() for b in self.outputs ]
        return d

    def __eq__(self, other):
        return self.name == other.name


class kaasReq():
    @classmethod
    def fromDict(cls, d):
        kernels = [ kernelSpec.fromDict(ks) for ks in d['kernels'] ]
        return cls(kernels)


    def __init__(self, kernels):
        """Turn a list of kernelSpecs into a kaas Request"""
        self.kernels = kernels


    def toDict(self):
        return {
                "kernels" : [ k.toDict() for k in self.kernels ]
               }


def getHandle(mode, ctx):
    """Returns a libff RemoteFunc object representing the KaaS service"""
    if mode == 'direct':
        # return libff.invoke.DirectRemoteFunc(serverPackage / '__init__.py', 'invoke', ctx)
        return libff.invoke.DirectRemoteFunc(serverPackage, 'invoke', ctx)
    elif mode == 'process':
        return libff.invoke.ProcessRemoteFunc(serverPackage, 'invoke', ctx)
    else:
        raise KaasError("Unrecognized execution mode: " + str(mode))


def kaasServe(req, ctx):
    # Convert the dictionary req into a kaasReq object
    kReq = kaasReq.fromDict(req)

    kaasServeInternal(kReq, ctx)


def LibffInvokeRegister():
    """Callback required by libff.invoke in DirectRemoteFunc mode"""

    return { "invoke" : kaasServe }
