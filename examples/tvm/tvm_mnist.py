import sys
import numpy as np
import pickle

from tvm import relay
from tvm.relay import testing
import tvm

batch_size = 1
num_class = 10
image_shape = (1, 28, 28)

mod, params = relay.testing.mlp.get_workload(batch_size)

target = tvm.target.cuda()
with tvm.transform.PassContext():
    graphMod = relay.build(mod, target, params=params)

lib = graphMod.get_lib()
cudaLib = lib.imported_modules[0]

outputName = "mnist"

print("\n")
print("Saving execution graph to: " + outputName + "_graph.json")
with open(outputName + "_graph.json", 'w') as f:
    f.write(graphMod.get_json())

print("Saving parameters to: " + outputName + "_params.json")
with open(outputName + "_params.pickle", "wb") as f:
    # Can't pickle tvm.ndarray, gotta convert to numpy
    pickle.dump({ k : p.asnumpy() for k,p in params.items() }, f)

print("Saving Raw CUDA Source to: " + outputName + ".cu")
with open(outputName + ".cu", 'w') as f:
    f.write(cudaLib.get_source())

print("Saving CUDA to: {}.ptx and {}.tvm_meta.json:".format(outputName,outputName))
cudaLib.save(outputName + ".ptx")
