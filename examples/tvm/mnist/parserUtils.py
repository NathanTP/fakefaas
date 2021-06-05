from mnist import MNIST
from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_runtime
import pickle
import pathlib
import numpy as np
import json


def loadParams():
    params = pickle.load(open("mnist/params", 'rb'))
    return params


def getPath(): 
    path = pathlib.Path(__file__).resolve().parent / 'code.cubin' 
    return path


def getShapes():
    return [
        [(1, 1, 1), (784, 1, 1)],
        [(128, 1, 1), (64, 1, 1)],
        [(64, 1, 1), (64, 1, 1)],
        [(10, 1, 1), (64, 1, 1)],
        [(1, 1, 1), (32, 1, 1)]
        ]    


def getGraph(): 
    with open("mnist/graph.json") as json_file:
            data = json.load(json_file)
    return data


def processImage(img):
    new_img = np.zeros(shape=(1, 1, 28, 28), dtype=np.float32)
    for i in range(28):
        for j in range(28):
            new_img[0][0][i][j] = img[i * 28 + j]/255

    return new_img


def readData():
    index = 0
    mnistData, images, labels = loadMnist()
    return processImage(images[index]) 


def loadMnist(path=pathlib.Path("mnist").resolve() / 'fakedata', dataset='test'):
	mnistData = MNIST(str(path))
	if dataset == 'train':
		images, labels = mnistData.load_training()
	else:
		images, labels = mnistData.load_testing()

	images = np.asarray(images).astype(np.float32)
	labels = np.asarray(labels).astype(np.uint32)

	return mnistData, images, labels


def makeParams():
    batch_size = 1
    num_class = 10
    image_shape = (1, 28, 28)
    
    mod, params = relay.testing.mlp.get_workload(batch_size)

    target = tvm.target.cuda()
    with tvm.transform.PassContext():
        graphMod = relay.build(mod, target, params=params)
    
    p = graphMod.get_params()
    keys = p.keys()
    new_dict = dict()
    for key in keys:
        new_dict[key] = p[key].asnumpy()

    pickle.dump(new_dict, open("params", "wb"))
    
if __name__ == "__main__":
    makeParams() 
