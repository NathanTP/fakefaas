import sys
import time
import random
import libff

def echo(req, ctx):
    return req


def state(req, ctx):
    if 'scratchData' in req:
        ctx.scratch = req['scratchData']

    return {"cachedData" : ctx.scratch}


random.seed(time.time())
def perfSim(req, ctx):
    # You can time stuff
    with libff.timer('runtime', ctx.profs):
        time.sleep(req['runtime'] / 1000)

    # You can also record non-time stuff using libff.prof() objects
    if 'randSample' not in ctx.profs:
        ctx.profs['randSample'] = libff.prof()
    randomMetric = random.randint(0,100)
    ctx.profs['randSample'].increment(randomMetric)

    # returns the random metric to aid in testing
    return {"validateMetric" : randomMetric}


funcMap = {"echo" : echo, "perfSim" : perfSim, 'state' : state}

def LibffInvokeRegister():
    return funcMap 

if __name__ == "__main__":
    import libff.invoke
    libff.invoke.RemoteProcessServer(funcMap, sys.argv[1:])
