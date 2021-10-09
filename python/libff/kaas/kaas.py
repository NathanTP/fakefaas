# The KaaS server (executor) is implemented as a normal libff function (for
# now) but it accepts KaaS requests from different clients. You can think of it
# like a 2-level implementation (libff for the internode and kaasServer for
# intranode). Eventually it will be its own service because scheduling
# decisions will be different.

import pathlib
import collections
import os

serverPackage = pathlib.Path(__file__).resolve().parent
dir_path = os.path.dirname(os.path.realpath(__file__))


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
                   key=d.get('key', None),
                   ephemeral=d.get('ephemeral', False),
                   const=d.get('const', False))

    def __init__(self, name, size, key=None, ephemeral=False, const=False):
        """Arguments:
            name - internal name for this buffer
            size - size of the buffer (in bytes)
            key - For non-epehemeral buffers, this is the key in the kv store
                  to use. If key is not specified, name is used as the key.
            epehemeral - If true, this buffer does not interact with the KV store
            const - If true, this buffer is guaranteed to never change in the KV store
        """
        self.name = name

        if key is None:
            # This is for backwards compatibility with older KaaS code that
            # uses name for both internal and kv name. It should not matter for
            # new code and shouldn't be relied on going forward.
            self.key = name
        else:
            self.key = key

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
                'name': self.name,
                'size': self.size,
                'key': self.key,
                'ephemeral': self.ephemeral,
                'const': self.const
            }

    def __eq__(self, other):
        if not isinstance(other, bufferSpec):
            return NotImplemented

        return self.name == other.name


class literalSpec():
    @classmethod
    def fromDict(cls, d):
        return cls(d['type'], d['val'])

    def __init__(self, t, val):
        """Literals are passed by value to kernsls and can be of several types,
        defined using the same symbol's as python's struct definition language
        (see pycuda's docs for function.prepare()). Val must be json
        serializable and convertable to t."""
        if t not in ['i', 'f', 'd', 'Q']:
            raise KaasError("Type " + str(t) + " not permitted for scalars")
        self.t = t
        self.val = val

    def toDict(self):
        return {"type": self.t, "val": self.val}

    def __str__(self):
        return f"Literal {self.t} {self.val}"


builtins = {
    "cutlass": pathlib.Path(dir_path) / "cutlass" / "cutlass.cubin",
    "complexCutlass": pathlib.Path(dir_path) / "complexCutlass" / "cutlass.cubin"
}

cutlassLib = {
    "sgemm0": "_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock12MmaPipelinedINS1_9GemmShapeILi128ELi128ELi8EEENS_9transform11threadblock22PredicatedTileIteratorINS_11MatrixShapeILi128ELi8EEEfNS_6layout8RowMajorELi1ENS8_30PitchLinearStripminedThreadMapINSD_16PitchLinearShapeILi8ELi128EEELi256ELi1EEELi1EEENS9_19RegularTileIteratorISC_fNSD_11ColumnMajorELi1ENS8_33TransposePitchLinearThreadMapSimtISI_EELi4EEENSA_INSB_ILi8ELi128EEEfSE_Li0ENSF_INSG_ILi128ELi8EEELi256ELi1EEELi1EEENSK_ISP_fSE_Li0ESR_Li4EEEfSE_NS4_9MmaPolicyINS1_4warp7MmaSimtINS6_ILi32ELi64ELi8EEEfSL_fSE_fSE_NSV_13MmaSimtPolicyINSB_ILi4ELi8EEENSD_19RowMajorInterleavedILi2EEENS6_ILi4ELi4ELi1EEEEELi1ELNS_16ComplexTransformE0ELS14_0EbEENSB_ILi4ELi0EEENSB_ILi0ELi0EEELi1EEENS_21NumericArrayConverterIffLi4ELNS_15FloatRoundStyleE2EEES1B_bEENS_8epilogue11threadblock8EpilogueIS7_S15_Li1ENS1E_22PredicatedTileIteratorINS1E_26OutputTileOptimalThreadMapINS1E_15OutputTileShapeILi128ELi1ELi4ELi4ELi1EEENS1I_ILi1ELi4ELi2ELi1ELi8EEELi256ELi1ELi32EEEfEENS1D_4warp20FragmentIteratorSimtISX_NS1_6thread3MmaINS6_ILi8ELi8ELi1EEEfSL_fSE_fSE_NS_4arch13OpMultiplyAddEbEESE_S13_EENS1N_16TileIteratorSimtISX_S1U_fSE_S13_EENS1E_18SharedLoadIteratorINS1L_18CompactedThreadMapEfLi4EEENS1D_6thread17LinearCombinationIfLi1EffLNS21_9ScaleType4KindE0ELS1A_2EEENSB_ILi0ELi17EEELi1EEENS4_30GemmIdentityThreadblockSwizzleILi1EEELb0EEEEEvNT_6ParamsE",
    "sgemm1": "ReferenceGemm_kernel",
    "complexGemm0": "_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock12MmaPipelinedINS1_9GemmShapeILi128ELi128ELi8EEENS_9transform11threadblock22PredicatedTileIteratorINS_11MatrixShapeILi128ELi8EEENS_7complexIfEENS_6layout8RowMajorELi1ENS8_30PitchLinearStripminedThreadMapINSF_16PitchLinearShapeILi8ELi128EEELi256ELi1EEELi1EEENS9_19RegularTileIteratorISC_SE_NSF_11ColumnMajorELi1ENS8_33TransposePitchLinearThreadMapSimtISK_EELi8EEENSA_INSB_ILi8ELi128EEESE_SG_Li0ENSH_INSI_ILi128ELi8EEELi256ELi1EEELi1EEENSM_ISR_SE_SG_Li0EST_Li8EEESE_SG_NS4_9MmaPolicyINS1_4warp7MmaSimtINS6_ILi32ELi64ELi8EEESE_SN_SE_SG_SE_SG_NSX_13MmaSimtPolicyINSB_ILi4ELi8EEENSF_19RowMajorInterleavedILi2EEENS6_ILi2ELi2ELi1EEEEELi1ELNS_16ComplexTransformE0ELS16_0EbEENSB_ILi2ELi0EEENSB_ILi0ELi0EEELi1EEENS_21NumericArrayConverterISE_SE_Li4ELNS_15FloatRoundStyleE2EEES1D_bEENS_8epilogue11threadblock8EpilogueIS7_S17_Li1ENS1G_22PredicatedTileIteratorINS1G_26OutputTileOptimalThreadMapINS1G_15OutputTileShapeILi128ELi1ELi4ELi4ELi1EEENS1K_ILi1ELi2ELi4ELi1ELi8EEELi256ELi1ELi64EEESE_EENS1F_4warp20FragmentIteratorSimtISZ_NS1_6thread3MmaINS6_ILi8ELi8ELi1EEESE_SN_SE_SG_SE_SG_NS_4arch13OpMultiplyAddEbEESG_S15_EENS1P_16TileIteratorSimtISZ_S1W_SE_SG_S15_EENS1G_18SharedLoadIteratorINS1N_18CompactedThreadMapESE_Li8EEENS1F_6thread17LinearCombinationISE_Li1ESE_SE_LNS23_9ScaleType4KindE0ELS1C_2EEENSB_ILi0ELi9EEELi1EEENS4_30GemmIdentityThreadblockSwizzleILi1EEELb0EEEEEvNT_6ParamsE"}


class kernelSpec():
    """Kernel specs describe a kernel for a particular request."""
    @classmethod
    def fromDict(cls, d):
        literals = d.get('literals', [])
        arguments = d.get('arguments', [])
        type_list = d.get('type_list', [])

        literals = [literalSpec.fromDict(lit) for lit in literals]
        args = []
        for i in range(len(arguments)):
            args.append((bufferSpec.fromDict(arguments[i]), type_list[i]))

        return cls(d['library'],
                   d['kernel'],
                   tuple(d['gridDim']),
                   tuple(d['blockDim']),
                   d['sharedSize'],
                   literals,
                   args)

    def __init__(self, library, kernel, gridDim, blockDim, sharedSize=0,
                 literals=[], arguments=[]):
        self.libPath = pathlib.Path(library).resolve()
        if library == builtins["cutlass"]:
            self.kernel = cutlassLib[kernel]
        elif library == builtins["complexCutlass"]:
            self.kernel = cutlassLib[kernel]
        else:
            self.kernel = kernel
        self.name = self.libPath.stem + "." + kernel

        self.gridDim = gridDim
        self.blockDim = blockDim
        self.sharedSize = sharedSize

        self.literals = literals

        self.arguments = []
        self.type_list = []
        self.inputs = []
        self.outputs = []
        self.temps = []
        for i in range(len(arguments)):
            arg = arguments[i]
            self.arguments.append(arg[0])
            self.type_list.append(arg[1])
            if 'i' in arg[1]:
                self.inputs.append(arg[0])
            if 'o' in arg[1]:
                self.outputs.append(arg[0])
            if 't' in arg[1]:
                self.temps.append(arg[0])

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
        d['literals'] = [lit.toDict() for lit in self.literals]
        d['arguments'] = [a.toDict() for a in self.arguments]
        d['type_list'] = [i for i in self.type_list]
        d['inputs'] = [b.toDict() for b in self.inputs]
        d['temps'] = [b.toDict() for b in self.temps]
        d['outputs'] = [b.toDict() for b in self.outputs]
        return d

    def __eq__(self, other):
        return self.name == other.name


class kaasReq():
    @classmethod
    def fromDict(cls, d):
        kernels = [kernelSpec.fromDict(ks) for ks in d['kernels']]
        if 'nIter' in d:
            nIter = d['nIter']
        else:
            nIter = 1
        return cls(kernels, nIter=nIter)

    def __init__(self, kernels, nIter=1):
        """Turn a list of kernelSpecs into a kaas Request"""
        self.kernels = kernels
        self.nIter = nIter

    def reKey(self, keyMap):
        """Renames kv keys for non-ephemeral buffers based on a keyMap.
        Internal names remain the same and epehemeral buffers are not affected.
        keyMap: {internalName -> newKey}
        """
        for kern in self.kernels:
            for buf in kern.arguments:
                if not buf.ephemeral:
                    if buf.name in keyMap:
                        buf.key = keyMap[buf.name]

    def toDict(self):
        return {"kernels": [k.toDict() for k in self.kernels],
                "nIter" : self.nIter}


denseBuf = collections.namedtuple("denseBuf",
                                  ['name', 'size', 'key', 'ephemeral', 'const'])

denseLiteral = collections.namedtuple("denseLiteral", ['type', 'val'])

denseKern = collections.namedtuple("denseKern",
                                   ['library', 'kernel', 'gridDim', 'blockDim',
                                    'sharedSize', 'literals', 'arguments'])


class kaasReqDense():
    """A high performance version of kaasReq that is fast to serialize and
    modify, though it's less pleasant to work with. We use tuples for everything:
        - buffers:  (0 name, 1 size, 2 key, 3 ephemeral?, 4 const?)
        - kernels:  (0 name, 1 libraryPath, 2 kernelFunc,
                     3 gridDim, 4 blockDim, 5 sharedSize,
                     6 literals, 7 arguments, 8 ioTypes)
        - literals: (0 type, 1 value)
    """
    @classmethod
    def fromDict(cls, d):
        kernels = [kernelSpec.fromDict(ks) for ks in d['kernels']]
        if 'nIter' in d:
            nIter = d['nIter']
        else:
            nIter = 1

        return cls(kernels, nIter=nIter)

    def __init__(self, kernels, nIter=1):
        self.bufferMap = {}
        self.kernels = []
        self.nIter = nIter
        for kern in kernels:
            arguments = []
            for buf in kern.arguments:
                if buf.name not in self.bufferMap:
                    self.bufferMap[buf.name] = (buf.name, buf.size, buf.key, buf.ephemeral, buf.const)
                arguments.append(buf.name)

            literals = [(literal.t, literal.val) for literal in kern.literals]
            dKern = (kern.name, str(kern.libPath), kern.kernel,
                     kern.gridDim, kern.blockDim, kern.sharedSize,
                     literals, arguments, kern.type_list)

            self.kernels.append(dKern)

    def reKey(self, keyMap):
        for name, newKey in keyMap.items():
            oldBuf = self.bufferMap[name]
            self.bufferMap[name] = (oldBuf[0], oldBuf[1], newKey, oldBuf[3], oldBuf[4])
