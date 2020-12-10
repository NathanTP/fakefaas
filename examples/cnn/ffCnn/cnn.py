import ctypes
import ctypes.util
import sys
import pathlib
from mnist import MNIST
import numpy as np
import types

import libff as ff
import libff.kv
import libff.invoke
import kaasServer as kaas

redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"

modelLibPath = pathlib.Path("../libkaascnn/libkaascnn.so").resolve()
kernelLibPath = pathlib.Path("../libkaascnn/libkaascnn.cubin").resolve()
modelDir = pathlib.Path("../model").resolve()
dataDir = pathlib.Path("../data").resolve()

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


def getLibCnnHandle():
    """Imports libCnn for local usage. As a KaaS client, we're only interested
    in the helpers for parsing saved models."""

    s = types.SimpleNamespace()

    packagePath = pathlib.Path(__file__).parent.parent
    # libcnnPath = packagePath / "libkaascnn" / "libkaascnn.so"

    s.cnnLib = ctypes.cdll.LoadLibrary(modelLibPath)
    
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

    return s


def getCtx(remote=False):
    if remote:
        objStore = ff.kv.Redis(pwd=redisPwd, serialize=True)
    else:
        objStore = ff.kv.Local(copyObjs=False, serialize=False)

    return libff.invoke.RemoteCtx(None, objStore)


def loadMnist(path, dataset='test'):
    mnistData = MNIST(str(path))

    if dataset == 'train':
        images, labels = mnistData.load_training()
    else:
        images, labels = mnistData.load_testing()

    images = np.asarray(images).astype(np.float32)
    labels = np.asarray(labels).astype(np.uint32)

    return images, labels


def loadLayersLocal(path):
    # host layers
    hlayers = [
            __state.cnnLib.layerParamsFromFile(str(path / 'l_input').encode('utf-8')),
            __state.cnnLib.layerParamsFromFile(str(path / 'l_c1').encode('utf-8')),
            __state.cnnLib.layerParamsFromFile(str(path / 'l_s1').encode('utf-8')),
            __state.cnnLib.layerParamsFromFile(str(path / 'l_f').encode('utf-8')),
           ]

    if None in hlayers:
        raise RuntimeError("Failed to load layers")


def getLayerKerns(name, inputBuf, weights, bias, preact, output):
    kerns = [
            kaas.kernelSpec(kernelLibPath, "fp_preact_"+name,
                (64,1,1), (64,1,1),
                inputs=[inputBuf, weights],
                outputs=[preact]),

            kaas.kernelSpec(kernelLibPath, "fp_bias_"+name,
                (64,1,1), (64,1,1),
                inputs=[bias, preact],
                outputs=[preact]),

            kaas.kernelSpec(kernelLibPath, "apply_step_function",
                (64,1,1), (64,1,1),
                literals=[ kaas.literalSpec('Q', output.size // 4) ],
                inputs=[preact],
                outputs=[output])
            ]
    return kerns

class model():
    def __init__(self, ctx, kaasHandle):
        libHandle = getLibCnnHandle()

        hlayers = [ libHandle.cnnLib.layerParamsFromFile(str(modelDir / 'l_c1').encode('utf-8')).contents,
                    libHandle.cnnLib.layerParamsFromFile(str(modelDir / 'l_s1').encode('utf-8')).contents,
                    libHandle.cnnLib.layerParamsFromFile(str(modelDir / 'l_f').encode('utf-8')).contents ]

        layerModels = []
        layerPreacts  = []
        # Layer setup
        for name, layer in zip(['c1', 's1', 'f'], hlayers): 
            ctx.kv.put(name+"-bias", np.ctypeslib.as_array(layer.bias, shape=(layer.N,)))
            ctx.kv.put(name + "-weight", np.ctypeslib.as_array(layer.weight, shape=(layer.N * layer.M,)))

            layerModels.append( (
                kaas.bufferSpec(name+"-weight", layer.N*layer.M*4, const=True),
                kaas.bufferSpec(name+"-bias", layer.N*4, const=True)
                ))

            layerPreacts.append(kaas.bufferSpec(name+"-preact", layer.O*4, ephemeral=True))


        c1Out = kaas.bufferSpec('c1-out', hlayers[0].O*4, ephemeral=True)
        s1Out = kaas.bufferSpec('s1-out', hlayers[1].O*4, ephemeral=True)

        # We need to know the size of the output to generate the kernels. This
        # bufferSpec will be replaced when we actually call the model.
        fOut = kaas.bufferSpec('outputPlaceholder', hlayers[2].O*4)

        # The first input and last output will get filled in when the model actually gets called
        allKerns = []
        allKerns += getLayerKerns('c1',  None, layerModels[0][0], layerModels[0][1], layerPreacts[0], c1Out)
        allKerns += getLayerKerns('s1',  c1Out, layerModels[1][0], layerModels[1][1], layerPreacts[1], s1Out)
        allKerns += getLayerKerns('f',   s1Out, layerModels[2][0], layerModels[2][1], layerPreacts[2], fOut)

        self.kerns = allKerns
        self.ctx = ctx
        self.handle = kaasHandle
        self.inSize = hlayers[0].N*4
        self.outSize = hlayers[2].O*4


    def classify(self, inputName, outputName):
        inputBuf = kaas.bufferSpec(inputName, self.inSize) 
        outBuf = kaas.bufferSpec(outputName, self.outSize)

        self.kerns[0].inputs[0] = inputBuf
        self.kerns[-1].outputs[0] = outBuf

        req = kaas.kaasReq(self.kerns)
        self.handle.Invoke(req.toDict())

        pred = self.ctx.kv.get(outputName)
        return np.argmax(np.frombuffer(pred.data, dtype=np.float32))


if __name__ == '__main__':
    ffCtx = getCtx(remote=False)
    kaasHandle = kaas.getHandle('direct', ffCtx)

    model = model(ffCtx, kaasHandle)

    imgs, lbls = loadMnist(dataDir)
    
    nWrong = 0
    for i in range(len(imgs)):
        ffCtx.kv.put('testInput', imgs[i])

        model.classify('testInput', 'testOutput')

        preds = ffCtx.kv.get('testOutput')
        topPred = np.argmax(np.frombuffer(preds.data, dtype=np.float32))
        if topPred != lbls[i]:
            nWrong += 1

    print("Error Rate: ", (nWrong / len(imgs))*100)
