import sys
import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_runtime

batch_size = 1
num_class = 10
image_shape = (1, 28, 28)

mod, params = relay.testing.mlp.get_workload(batch_size)

target = tvm.target.cuda()
with tvm.transform.PassContext():
    graphMod = relay.build(mod, target, params=params)

lib = graphMod.get_lib()
cudaLib = lib.imported_modules[0]
print("Raw source code for generated cuda:")
print(cudaLib.get_source())

outputName = "mnist"
print("\n\nSaving CUDA to {}.ptx and {}.tvm_meta.json:".format(outputName,outputName))
cudaLib.save(outputName + ".ptx")
