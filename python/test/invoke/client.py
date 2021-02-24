import pathlib
import time
import sys
from pprint import pprint

import libff
from libff import invoke

workerPath = pathlib.Path(__file__).parent.resolve() / "worker.py"

def helloWorld(constructor):
    ctx = libff.invoke.RemoteCtx(None, None)

    func = constructor(workerPath, "echo", ctx)

    resp = func.Invoke({"hello" : "world"})
    print(resp)


def testNonBlock(constructor):
    """non-blocking only works for 'process' mode"""
    if constructor is libff.invoke.DirectRemoteFunc:
        print("Non-blocking mode only works with process remote funcs") 
        return False

    ctx = libff.invoke.RemoteCtx(None, None)
    func = constructor(workerPath, "perfSim", ctx)

    start = time.time()
    fut = func.InvokeAsync({"runtime" : 1000})
    reqFinishTime = time.time() - start

    start = time.time()
    resp = fut.get(block=False)
    nonBlockRuntime = time.time() - start
    if resp is not None:
        print("Future returned too early after non-blocking get")
        return False

    start = time.time()
    resp = fut.get()
    blockRuntime = time.time() - start
    if resp is None:
        print("Future didn't block when asked to")
        return False

    if reqFinishTime > 1:
        print("InvokeAsync took too long")
        return False
    if nonBlockRuntime > 1:
        print("Non-blocking get took too long")
        return False
    if blockRuntime < 1:
        print("Blocking call was too fast")
        return False

    return True
   

def testCuda(constructor):
    ctx = libff.invoke.RemoteCtx(None, None)

    f0 = constructor(workerPath, "cuda", ctx, enableGpu=True)
    f1 = constructor(workerPath, "cuda", ctx, enableGpu=True)

    # Make sure multiple functions for the same client work
    r0 = f0.Invoke({})
    r1 = f1.Invoke({})
    if r0['deviceID'] != r1['deviceID']:
        print("TEST INVALID: libff gave different devices to two different functions from the same client. While legal, this invalidates the rest of the test, exiting early.")
        print(r0)
        print(r1)
        return True

    # Different clients should get different devices
    f2 = constructor(workerPath, "cuda", ctx, clientID=1, enableGpu=True)
    r2 = f2.Invoke({})
    if r2['deviceID'] == r0['deviceID'] or r2['deviceID'] == r1['deviceID']:
        # In theory this isn't completely valid, it's possible that libff
        # killed client0's executor and re-used the GPU. In practice, it's not
        # going to do this (until it does at which point we'll have to fix this
        # test).
        print("FAIL: New client got old GPU")
        print("client0, func0:", r0)
        print("client0, func1:", r1)
        print("client1, func0:", r2)
        return

    # The test system only has 2 GPUs, libff is gonna have to kill an executor
    # for this to work.
    f3 = constructor(workerPath, "cuda", ctx, clientID=2, enableGpu=True)
    r3 = f3.Invoke({})

    return True


# XXX this test isn't strictly valid. libff is free to give a new executor on
# each invocation if it chooses which would break this test. In practice,
# without heavy load it won't.
def testState(constructor):
    ctx = libff.invoke.RemoteCtx(None, None)

    f0 = constructor(workerPath, "state", ctx)

    resp = f0.Invoke({"scratchData" : "firstData"})
    if resp['cachedData'] != 'firstData':
        print("FAIL: First invoke didn't return scrach data")
        return False

    resp = f0.Invoke({})
    if resp['cachedData'] != 'firstData':
        print("FAIL: function didn't cache data")
        return False

    resp = f0.Invoke({"scratchData" : "secondData"})
    if resp['cachedData'] != 'secondData':
        print("FAIL: function didn't return new data")
        return False

    resp = f0.Invoke({})
    if resp['cachedData'] != 'secondData':
        print("FAIL: function didn't replace cache data")
        return False

    # New clients need to get new executors
    f1 = constructor(workerPath, "state", ctx, clientID=1)
    resp = f1.Invoke({})
    if resp['cachedData'] is not None:
        print("FAIL: new client ID got old client's cached data")
        return False

    return True


def _checkStats(stats, measured, expect):
    reported = stats['worker:runtime']
    if reported < expect or measured < expect:
        print("FAIL: runtime too fast")
        return False
    elif reported > 2*expect or measured > 2*expect:
        print("FAIL: runtime too slow")
    else:
        return True


def _runStatsFunc(func, runTime):
    start = time.time()
    resp = func.Invoke({"runtime" : runTime})
    measured = (time.time() - start)*1000
    stats = func.getStats().report()

    return (stats, measured)

def stats(constructor):
    sleepTime = 1000
    repeat = 2
    ctx = libff.invoke.RemoteCtx(None, None)

    stats = libff.profCollection()
    f1 = constructor(workerPath, "perfSim", ctx, stats=stats.mod('f1'))
    f2 = constructor(workerPath, "perfSim", ctx, stats=stats.mod('f2'))

    #==========================================================================
    # Cold start and single function behavior 
    #==========================================================================
    coldStats, coldMeasured = _runStatsFunc(f1, sleepTime)
    if not _checkStats(coldStats, coldMeasured, sleepTime):
        print("Cold start ran too fast")
        print("Direct Stats: " + str(coldMeasured))
        print("Reported Stats: " + str(coldStats))
        return False
    f1.resetStats()
    stats.reset()

    #==========================================================================
    # Warm start and multiple function behavior 
    #==========================================================================
    # Multiple functions can be warm at the same time and should maintain stats separately
    for i in range(repeat):
        stat1, measured1 = _runStatsFunc(f1, sleepTime)
        stat2, measured2 = _runStatsFunc(f2, sleepTime*2)
        if not _checkStats(stat1, measured1, sleepTime) or not _checkStats(stat2, measured2, sleepTime*2):
            print("Warm start failed on iteration " + str(i))
            print("Direct Stats 1: " + str(measured1))
            print("Reported Stats 1: ")
            pprint(stat1)
            print("\nDirect Stats 2: " + str(measured2))
            print("Reported Stats 2: ")
            pprint(stat2)
            return False
    f1.resetStats()
    f2.resetStats()
    stats.reset()

    #==========================================================================
    # Reset logic
    #==========================================================================
    resp = f1.Invoke({"runtime" : sleepTime})
    resp = f2.Invoke({"runtime" : sleepTime})
    f1.resetStats()

    # Reset stats should clear everything, even on the worker 
    stats1 = f1.getStats().report()
    if len(stats1) != 0:
        print("Stats were not cleaned")
        pprint(stats1)
        return False

    # But not on other functions
    stats2 = f2.getStats().report()
    if len(stats2) == 0 or stats2['worker:runtime'] < sleepTime:
        print("Wrong client stats got reset")
        return False

    return True


def testAsync(constructor):
    ctx = libff.invoke.RemoteCtx(None, None)
    func = constructor(workerPath, "perfSim", ctx)

    start = time.time()
    fut0 = func.InvokeAsync({"runtime" : 1000})
    fut1 = func.InvokeAsync({"runtime" : 1000})
    reqFinishTime = time.time() - start

    start = time.time()
    resp1 = fut1.get()
    resp0 = fut0.get()
    asyncRuntime = time.time() - start

    start = time.time()
    resp2 = func.Invoke({"runtime" : 1000})
    syncRuntime = time.time() - start

    if reqFinishTime >= 1:
        print("Async FAIL: InvokeAsync didn't return immediately")
        return False

    if asyncRuntime < 1:
        print("Async FAIL: Future returned too soon")
        return False

    if syncRuntime < 1:
        print("Async FAIL: synchronous call returned too soon")
        return False

    respVals = [resp0['validateMetric'], resp1['validateMetric'], resp2['validateMetric']]
    if len(set(respVals)) != len(respVals):
        print("Async FAIL: some futures returned the same response")
        print(respVals)
        return False

    return True

if __name__ == "__main__":
    funcConstructor = libff.invoke.ProcessRemoteFunc
    # funcConstructor = libff.invoke.DirectRemoteFunc

    print("Basic Hello World Test:")
    helloWorld(funcConstructor)
    print("PASS")

    if invoke.cudaAvailable:
        print("Testing GPU support")
        if not testCuda(funcConstructor):
            sys.exit(1)
        print("PASS")
    else:
        print("GPU support unavailable, skipping test")

    print("Testing Stats")
    if not stats(funcConstructor):
        sys.exit(1)
    print("PASS")

    print("Testing Async")
    if not testAsync(funcConstructor):
        sys.exit(1)
    print("PASS")

    print("Testing Non-blocking futures")
    if not testNonBlock(funcConstructor):
        sys.exit(1)
    print("PASS")

    print("Testing Private State")
    if not testState(funcConstructor):
        sys.exit(1)
    print("PASS")
