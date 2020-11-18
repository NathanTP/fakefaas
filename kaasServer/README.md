# Request Format

## Top Level Request
This is the request format expected by the kaasServer.

    { "kernels" : [ # List of kernelSpecs to invoke (in order) ] }

## KernelSpec
Kernel specs represent a single invocation of some kernel. Inputs are read from
a KV store, while outputs are temps are created as needed. Only outputs are
saved back to the KV store.

    {
        "library" : # Absolute path to the CUDA wrapper shared library,
        "kernel"  : # name of the wrapper func in library to call,
        "inputs"  : [ # list of input buffersSpecs to use ],
        "temps"   : [ # list of temporary buffersSpecs to use ],
        "outputs" : [ # list of output buffersSpecs to use ],
        "nGrid"   : # Number of CUDA grids to use when invoking,
        "nBlock"  : # Number of CUDA blocks to use when invoking 
    }

## BufferSpec
Describes a single buffer that will be used by kaas.

    {
        "name"      : # string identifier for this buffer, will be the key in the KV store (if used)
        "size"      : # Number of bytes in the buffer,
        "ephemeral" : # boolean indicating whether the buffer should be
                        persisted to the KV store or not. Ephemeral buffers have a lifetime equal to a
                        single kaas request.,
        "const"     : # A guarantee that the buffer will never change
                        externally. This is a bit of a hack until we get a proper cache figured out.
    }
