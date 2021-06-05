import sys
import numpy as np
import pickle 
import json
from mnist import MNIST
import pathlib
from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
import tvm.contrib.graph_executor as runtime


''' Program that uses TVM to completely run an MNIST example. Generates source.cu, graph.txt, and graph.json. Specify which MNIST input to use with its index. '''


def loadMnist(path, dataset='test'):
	mnistData = MNIST(str(path))	
	if dataset == 'train':
		images, labels = mnistData.load_training()
	else:
		images, labels = mnistData.load_testing()

	images = np.asarray(images).astype(np.float32)
	labels = np.asarray(labels).astype(np.uint32)
	return mnistData, images, labels

def main(index):
    batch_size = 1
    num_class = 10
    image_shape = (1, 28, 28)

    out_shape = (batch_size, num_class)

    mod, params = relay.testing.mlp.get_workload(batch_size)

    target = tvm.target.cuda()
    with tvm.transform.PassContext():
        graphMod = relay.build(mod, target, params=params)

    lib = graphMod.get_lib()
    cudaLib = lib.imported_modules[0]

    ''' saves the source locally in a file named source.cu'''
    with open("source.cu", 'w') as out:
        out.write(cudaLib.get_source())

    outputName = "mnist"

    ''' this code generates ptx code and a json of metadata. I don't use it, so I've left it commented out'''
    #cudaLib.save(outputName + ".ptx")
    
    ctx = tvm.gpu()

    js = str(graphMod.get_graph_json())

    with open("graph.txt", 'w') as out:
	    out.write(js)

    js = json.loads(js)

    with open("graph.json", 'w') as outfile:
	    json.dump(js, outfile)


    dataDir = pathlib.Path("fakedata").resolve()

    mndata, imgs, lbls = loadMnist(dataDir)

    #index = 4       #0#11#101

    image = imgs[index]
    image = (1/255) * image
    temp_image = np.zeros((28, 28))

    for i in range(28):
	    for j in range(28):
		    temp_image[i][j] = image[i * 28 + j]
    '''
    for i in range(28):
	    for j in range(14, 28):
		    temp_image[i][j-14] = image[i * 28 + j]

    #print(temp_image)
    '''
    #print(lbls[index])
    true_image = np.zeros((1, 1, 28, 28))
    for i in range(28):
	    for j in range(28):
		    value = temp_image[i][j]
		    true_image[0][0][i][j] = value
    true_image = true_image.astype(np.float32)

    ''' #code that tests a perfect one 
    true_image = np.zeros((1, 1, 28, 28))

    for i in range(4, 24):
	    for j in range(12, 15):
		    true_image[0][0][i][j] = 1

    true_image = true_image.astype(np.float32)
    print(true_image)
    '''
    module = runtime.GraphModule(graphMod["default"](ctx))
    module.set_input("data", true_image)
    module.run()
    out = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()
    #print(out)

    thing = np.ndarray.flatten(out)
    thing2 = []
    for i in thing:
	    thing2.append(i)
    #print(thing2.index(max(thing2)))



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('index', metavar='index', type=int)

    args = parser.parse_args()
    
    main(args.index)
