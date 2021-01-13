import sys
import time
import libff as ff

import pygemm

#XXX
# Cache between calls
# gemmFunc = None


class sgemmState():
    def __init__(self, shapes, bKeys, func=None):
        self.shapes = shapes
        self.bKeys = bKeys
        self.func = func

    def __eq__(self, other):
        if not isinstance(other, sgemmState):
            return False

        if len(other.shapes) != len(self.shapes):
            return False

        for myShape, theirShape in zip(self.shapes, other.shapes):
            if myShape != theirShape:
                return False

        if len(self.bKeys) != len(other.bKeys):
            return False

        for myB, theirB in zip(self.bKeys, other.bKeys):
            if myB != theirB:
                return False

        return True


def sgemm(req, ctx):
    """Handle a chained sgemm request. Request fields:
        - 'input' : key for the first array
        - 'bArrs' : list of keys for the B (constant) arrays 
        - 'shapes' : list of shape tuples (M,N,K) for each layer
        - 'output' : key to use when writing output
        - 'useCuda' : Boolean requesting GPU acceleration
        - 'preprocess' : Time (in ms) to preprocess before executing the gemm.
    """
    # print("INVOKING", file=sys.stderr)
    # print(type(req), file=sys.stderr)
    # print(req, file=sys.stderr)
    # print("\n\n", file=sys.stderr)
    shapes = [ pygemm.mmShape(s[0], s[1], s[2]) for s in req['shapes'] ]

    reqState = sgemmState(shapes, req['bArrs'])
    if reqState != ctx.scratch:
        if ctx.scratch is not None:
            ctx.scratch.func.destroy()

        bArrs = []
        for bKey, shape in zip(req['bArrs'], shapes):
            bArrs.append(pygemm.getData(ctx, bKey, shape.b)) 

        reqState.func = pygemm.local.ChainedMults('faasWorker', shapes, bArrs=bArrs, useCuda=req['useCuda'])
        ctx.scratch = reqState

    # If you include preprocess in sgemm directly, we simulate a work that does
    # both preprocessing and GPU calculation in the same function. You can also
    # omit preprocess in this req and instead directly invoke the preprocess()
    # function in this file to simulate two separate invocations.
    if 'preprocess' in req:
        time.sleep(req['preprocess'] / 1000)

    inArr = pygemm.getData(ctx, req['input'], shapes[0].a)

    outArr = ctx.scratch.func.invoke(inArr)

    ctx.kv.put(req['output'], outArr)

    return {}


def preprocess(req, ctx):
    """Simulate a preprocessing phase that precedes sgemm."""
    inp = ctx.kv.get(req['input'])
    time.sleep(req['processTime'] / 1000)
    ctx.kv.put(req['output'], inp)


funcMap = {'sgemm' : sgemm, 'preprocess' : preprocess}
def LibffInvokeRegister():
    return funcMap 

if __name__ == "__main__":
    import libff.invoke
    libff.invoke.RemoteProcessServer(funcMap, sys.argv[1:])
