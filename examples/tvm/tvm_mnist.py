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
from tvm.contrib import graph_runtime
'''
def loadMnist(path, dataset='test'):
    mnistData = MNIST(str(path))

    if dataset == 'train':
        images, labels = mnistData.load_training()
    else:
        images, labels = mnistData.load_testing()

	
    print(len(images))

    images = np.asarray(images).astype(np.float32)
    labels = np.asarray(labels).astype(np.uint32)

    print(images.shape)

    return images, labels
'''




def loadMnist(path, dataset='test'):
	mnistData = MNIST(str(path))
	
	if dataset == 'train':
		images, labels = mnistData.load_training()
	else:
		images, labels = mnistData.load_testing()

	#print(len(images))

	images = np.asarray(images).astype(np.float32)
	labels = np.asarray(labels).astype(np.uint32)

	#print(images.dtype)
	#print(images.shape)
	#print(images[0])

	return mnistData, images, labels



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
#print("Raw source code for generated cuda:")
#print(cudaLib.get_source())

#saves the source locally in a file named source.cu
with open("source.cu", 'w') as out:
    out.write(cudaLib.get_source())

outputName = "mnist"
#print("\n\nSaving CUDA to {}.ptx and {}.tvm_meta.json:".format(outputName,outputName))
#cudaLib.save(outputName + ".ptx")



ctx = tvm.gpu()

#print(type(graphMod))
#print(type(graphMod))
#print(graphMod.get_params())

js = str(graphMod.get_json())

#print(type(json))

with open("graph.txt", 'w') as out:
	out.write(js)

js = json.loads(js)

#f = open("graph.jso")

#js = dict(js)

with open("graph.json", 'w') as outfile:
	json.dump(js, outfile)

#f.write(js)

#f.close()


#print(type(params))
#print(params)



dataDir = pathlib.Path("fakedata").resolve()

mndata, imgs, lbls = loadMnist(dataDir)

#print(imgs[0])
'''
for i in range(100):
	if lbls[i] == 1:
		index = i
		break
'''

#print(index)

index = 4#0#11#101

image = imgs[index]

#print(image)

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

print(lbls[index])

#print(image)
#print(image.shape)
#print(image[1])
#print(type(image[1]))


#print(image.shape)
#print(len(image))

true_image = np.zeros((1, 1, 28, 28))

for i in range(28):
	for j in range(28):
		value = temp_image[i][j]
		true_image[0][0][i][j] = value
		#if value > 0:
			#true_image[0][0][i][j] = 1
		#true_image[0][0][i][j] = temp_image[i][j]#image[i * 28 + j]

#print(true_image.dtype)

true_image = true_image.astype(np.float32)

#print(true_image.dtype)
#print(true_image)
#print(true_image)

''' #code that tests a perfect one 
true_image = np.zeros((1, 1, 28, 28))

for i in range(4, 24):
	for j in range(12, 15):
		true_image[0][0][i][j] = 1

true_image = true_image.astype(np.float32)
print(true_image)
'''


module = graph_runtime.GraphModule(graphMod["default"](ctx))




#print(imgs.shape)

#print(true_image)

module.set_input("data", true_image)


#print('boomer')

#import os
#print(os.getpid())


#import pdb
#pdb.set_trace()


module.run()

#print('boomer2')

out = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()


#import os
#print(os.getpid())



print(out)

#print(max(out))

thing = np.ndarray.flatten(out)

thing2 = []
for i in thing:
	thing2.append(i)

#print(thing2)


#print(thing2.index(max(thing2)))

print(thing2.index(max(thing2)))




print("HELLO WORLD!")
