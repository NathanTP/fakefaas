import sys
import time
import libff as ff

def preprocess(req, ctx):
    """Simulate a preprocessing phase that precedes sgemm."""
    inp = ctx.kv.get(req['input'], profile=ctx.profs.mod('kv'))
    time.sleep(req['processTime'] / 1000)
    ctx.kv.put(req['output'], inp, profile=ctx.profs.mod('kv'))


funcMap = {'preprocess' : preprocess}
def LibffInvokeRegister():
    return funcMap 

if __name__ == "__main__":
    import libff.invoke
    libff.invoke.RemoteProcessServer(funcMap, sys.argv[1:])
