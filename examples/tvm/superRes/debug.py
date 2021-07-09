import numpy as np
import pathlib
import pickle
import tempfile
import os
import sys
import io

# Defaults to home dir which I don't want. Have to set the env before loading
# the module because of python weirdness.
if "TEST_DATA_ROOT_PATH" not in os.environ:
    os.environ['TEST_DATA_ROOT_PATH'] = os.path.join(os.getcwd(), "downloads")

import onnx
import tvm
from tvm import te
import tvm.relay as relay
#from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor as graph_executor
from PIL import Image
import matplotlib.pyplot as plt
import json

from tvm.contrib.download import download_testdata

def importModelBuffer(buf):
    """Load the graph executor from an in-memory buffer containing the .so 
    file. This is a bit roundabout (we have to write it to a tmp file anyway),
    but it may be useful in faas settings."""
    with tempfile.TemporaryDirectory() as dpath:
        fpath = os.path.join(dpath, 'tmp.so')
        with open(fpath, 'wb') as f:
            f.write(buf)

        graphMod = tvm.runtime.load_module(fpath)

    ex = graph_executor.GraphModule(graphMod['default'](tvm.cuda()))
    return ex


def _loadOnnx():
    onnxPath = pathlib.Path.cwd() / "superres.onnx"
    onnxModel = onnx.load(onnxPath)
    #meta = getOnnxInfo(onnxModel)

    # Some onnx models seem to have dynamic parameters. I don't really know
    # what this is but the freeze_params and DynamicToStatic call seem to
    # resolve the issue.
    mod, params = relay.frontend.from_onnx(onnxModel, freeze_params=True)
    relay.transform.DynamicToStatic()(mod)
    with tvm.transform.PassContext(opt_level=3):
        module = relay.build(mod, tvm.target.cuda(), params=params)

    #afs = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))

    #print(afs)

    #dumpModel(module, afs)

    

    # Cache Results
    #libraryPath, metaPath = getCachePath(onnxPath.stem)
    #module.export_library(libraryPath)
    #with open(metaPath, 'w') as f:
        #json.dump(meta, f)

    #model = graph_executor.GraphModule(module['default'](tvm.cuda()))
    model = graph_executor.create(module.get_graph_json(), module.get_lib(), tvm.cuda())
    return model

    


def importOnnx(onnxPath, shape):
    """Will load the onnx model (*.onnx) from onnxPath and store it in a
    pre-compiled .so file. It will return a TVM executor capable of running the
    model. Shape is the input shape for the model and must be derived
    manually."""
    #libraryPath = pathlib.Path.cwd() / (onnxPath.stem + ".so")
    libraryPath = pathlib.Path.cwd() / "lib.so"

    if True:
        graphMod = tvm.runtime.load_module(libraryPath)
    #print(type(graphMod))
    #lib = graphMod.get_lib()
    #print(type(graphMod))
    #print(graphMod.get_lib)
    f = open("graph.json", "r")
    js = f.read()#graphMod.get_graph_json()
    f.close()
    #print(graphMod.get_function('fused_nn_conv2d_add_nn_relu_2'))
    return graph_executor.create(js, graphMod, tvm.cuda())
    #return graph_executor#graph_executor.GraphModule(graphMod['default'](tvm.cuda()))
    #return graph_executor.GraphModule(graphMod['default'](tvm.cuda()))

def getSuperRes():
    """Downloads the superres model used in this example"""
    # Pre-Trained ONNX Model
    modelUrl = "".join(
        [
            "https://gist.github.com/zhreshold/",
            "bcda4716699ac97ea44f791c24310193/raw/",
            "93672b029103648953c4e5ad3ac3aadf346a4cdc/",
            "super_resolution_0.2.onnx",
        ]
    )

    modelPath = download_testdata(modelUrl, "super_resolution.onnx", module="onnx")
    return pathlib.Path(modelPath)


def imagePreprocess(buf):
    # mode and size were manually read from the png (used Image.open and then
    # inspected the img.mode and img.size attributes). We're gonna just go with
    # this for now.
    img = Image.frombytes("RGB", (256,256), buf)

    img = img.resize((224,224))
    img = img.convert("YCbCr")
    img_y, img_cb, img_cr = img.split()
    imgNp = (np.array(img_y)[np.newaxis, np.newaxis, :, :]).astype("float32")
    return (img, imgNp)


def imagePostProcess(imgPil, outNp):
    img_y, img_cb, img_cr = imgPil.split()
    out_y = Image.fromarray(outNp, mode="L")
    out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
    out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
    result = Image.merge("YCbCr", [out_y, out_cb, out_cr]).convert("RGB")
    canvas = np.full((672, 672 * 2, 3), 255)
    canvas[0:224, 0:224, :] = np.asarray(imgPil)
    canvas[:, 672:, :] = np.asarray(result)
    plt.imshow(canvas.astype(np.uint8))

    with io.BytesIO() as f:
        plt.savefig(f, format="png")
        pngBuf = f.getvalue()
    return pngBuf


def getImage():
    """Downloads an example image for the test"""
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    img = Image.open(img_path)

    # We go through bytes to better align with faas requirements
    return img.tobytes()


def loadParams():
    params = pickle.load(open("params.pickle", 'rb'))
    return params




def executeModel(ex, img):
    params = loadParams()
    for key in params.keys():
        params[key] = tvm.nd.array(params[key])
    img_tvm_arr = tvm.nd.array(img)
    ex.set_input('1', img_tvm_arr)
    for key in params.keys():
        ex.set_input(key, params[key])
    inp = tvm.nd.array(img)
    a = tvm.nd.empty((1, 9, 224, 224), dtype='float32') 
    ex.run()
    return ex.get_output(0)


def main():
    imgBuf = getImage()
    imgPil, imgNp = imagePreprocess(imgBuf)


    modelPath = getSuperRes()
    #ex = importOnnx(modelPath, imgNp.shape)
    ex = importOnnx(1, 1)

    # Execute
    print("Running model")
    #print(imgNp)
    out = executeModel(ex, imgNp)

    print(out)
    #print(out)
    #print(out.shape)
    #print(out.dtype)
    # Display Result
    '''
    print("Success, saving output to test.png")
    pngBuf = imagePostProcess(imgPil, out)

    with open("test.png", "wb") as f:
        f.write(pngBuf)
    '''

if __name__ == "__main__":
    main()
