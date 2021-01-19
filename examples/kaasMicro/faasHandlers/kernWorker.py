import sys
import time
import libff as ff

sys.path.append("../")
import pygemm


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
    shapes = [ pygemm.mmShape(s[0], s[1], s[2]) for s in req['shapes'] ]

    reqState = sgemmState(shapes, req['bArrs'])
    if reqState != ctx.scratch:
        if ctx.scratch is not None:
            ctx.scratch.func.destroy()

        bArrs = []
        for bKey, shape in zip(req['bArrs'], shapes):
            bArrs.append(pygemm.getData(ctx, bKey, shape.b, stats=ctx.profs.mod('kv'))) 

        reqState.func = pygemm.local.ChainedMults('faasWorker', shapes, bArrs=bArrs, useCuda=req['useCuda'], stats=ctx.profs)
        ctx.scratch = reqState

    # Gotta reset on each call, can't cache
    ctx.scratch.func.stats = ctx.profs

    # If you include preprocess in sgemm directly, we simulate a work that does
    # both preprocessing and GPU calculation in the same function. You can also
    # omit preprocess in this req and instead directly invoke the preprocess()
    # function in this file to simulate two separate invocations.
    if 'preprocess' in req:
        with ff.timer('t_preprocess', ctx.profs):
            time.sleep(req['preprocess'] / 1000)

    inArr = pygemm.getData(ctx, req['input'], shapes[0].a, stats=ctx.profs.mod('kv'))

    outArr = ctx.scratch.func.invoke(inArr)

    ctx.kv.put(req['output'], outArr, profile=ctx.profs.mod('kv'))

    return {}


funcMap = {'sgemm' : sgemm}
def LibffInvokeRegister():
    return funcMap 

if __name__ == "__main__":
    import libff.invoke
    libff.invoke.RemoteProcessServer(funcMap, sys.argv[1:])
