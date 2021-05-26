import pathlib
import math
from pprint import pprint
import sys

import libff as ff
import libff.kv
import libff.invoke

import libff.kaas as kaas
import numpy as np

import json

import copy 

from mnist import MNIST

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_runtime

redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"
redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"

def getCtx(remote=False):
        if remote:
                objStore = ff.kv.Redis(pwd=redisPwd, serialize=True)
        else:
                objStore = ff.kv.Local(copyObjs=False, serialize=False)

        return libff.invoke.RemoteCtx(None, objStore)







def parse(inp, libffCtx, mode='process'):

        #fetches data
        with open("graph.json") as json_file:
                data = json.load(json_file)

        #print(data)

        params = loadParams()        

        #libffCtx = getCtx(remote=(mode == 'process'))
        

        #kaasHandle = kaas.getHandle(mode, libffCtx)

        #print(data["nodes"])           
        nodes = data["nodes"]

        kerns = []

        buffers = []

        #path = pathlib.Path(__file__).resolve().parent / 'mnist.ptx'
        path = pathlib.Path(__file__).resolve().parent / 'libkaasMicro.cubin'

        param_counter = 0

        shapes = getShapes()


        copy_inp = np.copy(inp)
        libffCtx.kv.put("0", copy_inp)
        input_buffer = kaas.bufferSpec("0", copy_inp.nbytes)
        buffers.append(input_buffer)    

    
        #print(copy_inp.shape)
        
        for i in range(1, len(nodes)):
        #for i in range(1, 2):

                output_size = np.array(data["attrs"]["shape"][1][i])
                ty = data["attrs"]["dltype"][1][i]
                node = nodes[i]

                #print(output_size)

                if node["op"] == "null":
                    index = "p" + str(param_counter)
                    param = params[index].asnumpy()
                    
                    print(type(param))
                    #zeros = copy.deepcopy(param)
                    zeros = np.zeros(output_size).astype(ty)
                    
                    np.copyto(zeros, param)
                
                    param_counter = param_counter + 1
                                        
                    buff = addToKV(libffCtx.kv, i, zeros)
                    buffers.append(buff)    
                else:
                #output_size = np.array(data["attrs"]["shape"][1][i])
                    ty = data["attrs"]["dltype"][1][i]
                    arr = np.zeros(output_size).astype(ty)

                    buff = addToKV(libffCtx.kv, i, arr)
                    buffers.append(buff)


                    #node = nodes[i]
                    kern = makeKern(node, i, path, shapes, i - param_counter - 1, buffers, libffCtx.kv)                       
                    


                #output_size = np.array(data["attrs"]["shape"][1][i])
                #buff = addToKV(nodes[i], libffCtx.kv, i, output_size)
                #buffers.append(buff)
                #kern = createKern(node, buffers, path, i)
                    kerns.append(kern)

        #print(buffers)


        
        req = kaas.kaasReq(kerns)
        
        return req, buffers      


#side note: probably there's a structure that contains the inputs with key-value being their name from the graph?


#output size is probably temporary since the function shouldn't actually think about the size they have to make 

def addToKV(kv, node_num, arr):
    kv.put(str(node_num), arr)
    nByte = arr.nbytes
    buff = kaas.bufferSpec(str(node_num), nByte)
    return buff 

'''
def addToKV(node_num, kv, output_size, ty):
        kv.put(str(node_num), np.zeros(output_size).astype(ty))
        nByte = output_size.nbytes
        buff = kaas.bufferSpec(str(node_num), nByte)
        return buff
'''    

def makeNullKern(node, param_counter, params):
    return None


def makeKern(node, i, path, shapes, shape_number, buffers, kv):
    name_func = node["name"] + "_kernel0" 
    shape = shapes[shape_number]

    
    #print(name_func)
    #print(shape[0], shape[1])
    #print(buffers) 

    inputs = []
    for index in range(len(node["inputs"])):
        #print(node["inputs"][index][0])
        inputs.append(buffers[node["inputs"][index][0]])


    #inputs.append(buffers[0])
    #inputs.append(buffers[0])
    #print(inputs)
    #print(np.frombuffer(kv.get('1'), dtype=np.float32).shape)

    #print(inputs)

    kern = kaas.kernelSpec(path, name_func, shape[0], shape[1], inputs=inputs, outputs=[buffers[i]])

    return kern


'''
def addToKV(node, kv, i, output_size):
        if node["op"] == "null":
                kv.put(str(i), np.zeros(output_size).astype('float32'))
                #nByte = np.prod(output_size) * 4
                nByte = output_size.nbytes
                buff = kaas.bufferSpec(str(i), nByte)
                return buff
        else:
                #nByte = np.prod(output_size) * 4
                nByte = output_size.nbytes
                buff = kaas.bufferSpec(str(i), nByte)
                return buff
'''             

        
#def createKern(path, name, input_size, output_size, inputs, ouputs):
        #kern = kaas.kernelSpec 



def getShapes():
    return [
        [(1, 1, 1), (784, 1, 1)],
        [(128, 1, 1), (64, 1, 1)],
        [(64, 1, 1), (64, 1, 1)],
        [(10, 1, 1), (64, 1, 1)],
        [(1, 1, 1), (32, 1, 1)]
        ]    




def loadParams():
    batch_size = 1
    num_class = 10
    image_shape = (1, 28, 28)

    mod, params = relay.testing.mlp.get_workload(batch_size)

    #print(type(mod))

    target = tvm.target.cuda()
    with tvm.transform.PassContext():
        graphMod = relay.build(mod, target, params=params)

    return graphMod.get_params()



def processImage(img):
    new_img = np.zeros(shape=(1, 1, 28, 28), dtype=np.float32)
    for i in range(28):
        for j in range(28):
            new_img[0][0][i][j] = img[i * 28 + j]/255


    return new_img



def readData(index):
    
    mnistData, images, labels = loadMnist()
    
    return processImage(images[index]), labels[index]


def loadMnist(path=pathlib.Path("fakedata").resolve(), dataset='test'):
	mnistData = MNIST(str(path))
	
	if dataset == 'train':
		images, labels = mnistData.load_training()
	else:
		images, labels = mnistData.load_testing()


	images = np.asarray(images).astype(np.float32)
	labels = np.asarray(labels).astype(np.uint32)


	return mnistData, images, labels






def run(mode='direct'):

    index = 4

    image, label = readData(index) 

    
    libffCtx = getCtx(remote=(mode == 'process'))

    req, buffers = parse(image, libffCtx)
    
    kaasHandle = kaas.getHandle(mode, libffCtx)

    kaasHandle.Invoke(req.toDict())

    c = np.frombuffer(libffCtx.kv.get('11'), dtype=np.float32)

    #print(image)
    print(c)

run(mode='direct')
        
