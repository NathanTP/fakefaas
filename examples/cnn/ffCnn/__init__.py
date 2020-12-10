import types
import ctypes
import ctypes.util
import pathlib
import numpy as np

class layerParams(ctypes.Structure):
    _fields_ = [    # Vector of N floats
                    ('bias', ctypes.POINTER(ctypes.c_float)),

                    # Matrix of NxM floats
                    ('weight', ctypes.POINTER(ctypes.c_float)),

                    # Vector of O floats
                    ('output', ctypes.POINTER(ctypes.c_float)),

                    # Vector of O floats (only ever exists on the device)
                    ('preact', ctypes.POINTER(ctypes.c_float)),

                    ('M', ctypes.c_int),
                    ('N', ctypes.c_int),
                    ('O', ctypes.c_int),
                    ('onDevice', ctypes.c_bool)
            ]

def __setup():
    s = types.SimpleNamespace()

    packagePath = pathlib.Path(__file__).parent.parent
    libcnnPath = packagePath / "libkaascnn" / "libkaascnn.so"

    s.cnnLib = ctypes.cdll.LoadLibrary(libcnnPath)
    
    # Function signatures
    s.cnnLib.initLibkaascnn.restype = ctypes.c_bool

    s.cnnLib.layerParamsFromFile.restype = ctypes.POINTER(layerParams)
    s.cnnLib.layerParamsFromFile.argtypes = [ ctypes.c_char_p ]

    s.cnnLib.layerParamsToDevice.restype = ctypes.POINTER(layerParams)
    s.cnnLib.layerParamsToDevice.argtypes = [ctypes.POINTER(layerParams)]

    s.cnnLib.newModel.restype = ctypes.c_void_p
    s.cnnLib.newModel.argtypes = [ctypes.c_void_p]*4

    s.cnnLib.printLayerWeights.argtypes = [ ctypes.POINTER(layerParams) ]

    s.cnnLib.classify.restype = ctypes.c_uint32
    s.cnnLib.classify.argtypes = [ ctypes.c_void_p, np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS") ]

    # Must be called exactly once per process
    ret = s.cnnLib.initLibkaascnn()
    if not ret:
        raise RuntimeError("Failed to initialize libfaascnn")

    return s

# Internal globals
__state = __setup()

# Makes all files in the package act like one module
from .cnn import *
