import sys
import libff as ff

import pygemm

# Cache between calls
gemmFunc = None

def handler(req, ctx):
    """Handle a chained sgemm request. Request fields:
        - 'input' : key for the first array
        - 'bArrs' : list of keys for the B (constant) arrays 
        - 'shapes' : list of shape tuples (M,N,K) for each layer
        - 'output' : key to use when writing output
        - 'useCuda' : Boolean requesting GPU acceleration
    """
    shapes = [ pygemm.mmShape(s[0], s[1], s[2]) for s in req['shapes'] ]

    global gemmFunc
    if gemmFunc is None:
        bArrs = []
        for bKey, shape in zip(req['bArrs'], shapes):
            bArrs.append(pygemm.getData(ctx, bKey, shape.b)) 

        gemmFunc = pygemm.local.ChainedMults('faasWorker', shapes, bArrs=bArrs, useCuda=req['useCuda'])

    inArr = pygemm.getData(ctx, req['input'], shapes[0].a)

    outArr = gemmFunc.invoke(inArr)

    ctx.kv.put(req['output'], outArr)

    return {}

funcMap = {'sgemm' : handler }
def LibffInvokeRegister():
    return funcMap 

if __name__ == "__main__":
    import libff.invoke
    libff.invoke.RemoteProcessServer(funcMap, sys.argv[1:])
