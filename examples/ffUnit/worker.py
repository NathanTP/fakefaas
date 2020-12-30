import sys
import time
import random
import libff

def echo(req, ctx):
    return req


random.seed(time.time())
def perfSim(req, ctx):
    # You can time stuff
    with libff.timer('runtime', ctx.profs):
        time.sleep(req['runtime'])

    # You can also record non-time stuff using libff.prof() objects
    if 'randSample' not in ctx.profs:
        ctx.profs['randSample'] = libff.prof()
    randomMetric = random.randint(0,100)
    ctx.profs['randSample'].increment(randomMetric)

    return {}


funcMap = {"echo" : echo, "perfSim" : perfSim}

def LibffInvokeRegister():
    return funcMap 

if __name__ == "__main__":
    import libff.invoke
    libff.invoke.RemoteProcessServer(funcMap, sys.argv[1:])