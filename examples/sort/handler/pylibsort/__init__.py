import types
import ctypes
import ctypes.util

def __setup():
    s = types.SimpleNamespace()

    # Not sure what is wrong here, ctypes.util.find_library is the right
    # function to call but it doesn't find libsort. The internal function
    # here does find it. The problem is in ctypes.util._get_soname which has
    # some error about the dynamic section when calling objdump -p -j .dynamic
    # libsort.so. This hack works for now.
    libsortPath = ctypes.util._findLib_ld("sort")
    if libsortPath is None:
        raise RuntimeError("libsort could not be located, be sure libsort.so is on your library search path (e.g. with LD_LIBRARY_PATH)")

    s.sortLib = ctypes.cdll.LoadLibrary(libsortPath)
    
    # Must be called exactly once per process
    s.sortLib.initLibSort()

    return s

# Internal globals
__state = __setup()

# Makes all files in the package act like one module
from .data import *
from .sort import *
