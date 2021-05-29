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







def parse(inp, data, libffCtx, path, params, shapes):



        nodes = data["nodes"]

        kerns = []

        buffers = []


        param_counter = 0


        copy_inp = np.copy(inp)
        libffCtx.kv.put("0", copy_inp)
        input_buffer = kaas.bufferSpec("0", copy_inp.nbytes)
        buffers.append(input_buffer)    

        
        for i in range(1, len(nodes)):

                output_size = np.array(data["attrs"]["shape"][1][i])
                ty = data["attrs"]["dltype"][1][i]
                node = nodes[i]


                if node["op"] == "null":
                    index = "p" + str(param_counter)
                    param = params[index]
                    
                    zeros = np.zeros(output_size).astype(ty)
                    
                    np.copyto(zeros, param)
                
                    param_counter = param_counter + 1
        
                                
                    buff = addToKV(libffCtx.kv, i, zeros)
                    buffers.append(buff)    
                else:
                    ty = data["attrs"]["dltype"][1][i]
                    arr = np.zeros(output_size).astype(ty)

                    buff = addToKV(libffCtx.kv, i, arr)
                    buffers.append(buff)


                    kern = makeKern(node, i, path, shapes, i - param_counter - 1, buffers, libffCtx.kv)                       

                    kerns.append(kern)



        
        req = kaas.kaasReq(kerns)
        
        return req, buffers      



def addToKV(kv, node_num, arr):
    kv.put(str(node_num), arr)
    nByte = arr.nbytes
    buff = kaas.bufferSpec(str(node_num), nByte)
    return buff 



def makeKern(node, i, path, shapes, shape_number, buffers, kv):
    name_func = node["name"] + "_kernel0" 
    shape = shapes[shape_number]

    

    inputs = []
    for index in range(len(node["inputs"])):
        inputs.append(buffers[node["inputs"][index][0]])


    kern = kaas.kernelSpec(path, name_func, shape[0], shape[1], inputs=inputs, outputs=[buffers[i]])

    return kern

def run(folder, mode='direct'):

    
    #load params, input, graph json, code, shapes  


    #loads in the source code
    path = pathlib.Path(__file__).resolve().parent / folder / 'code.cubin' 

    #loads in the graph as a json
    with open(folder + "/graph.json") as json_file:
                data = json.load(json_file)


    sys.path.insert(1, './' + folder)
    import parserUtils
    image = parserUtils.readData() 

    
    libffCtx = getCtx(remote=(mode == 'process'))

    params = parserUtils.loadParams()

    shapes = parserUtils.getShapes()

    req, buffers = parse(image, data, libffCtx, path, params, shapes)
    
    kaasHandle = kaas.getHandle(mode, libffCtx)

    kaasHandle.Invoke(req.toDict())

    c = np.frombuffer(libffCtx.kv.get('11'), dtype=np.float32)

    #print(image)
    print(c)

if __name__ == "__main__":
    
    folder = 'mnist'

    run(folder, mode='direct')     




   
