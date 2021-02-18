import pathlib
import time
import sys
from pprint import pprint

import libff
from libff import kv, array, invoke

workerPath = pathlib.Path(__file__).parent.resolve() / "worker.py"

def helloWorld():
    ctx = libff.invoke.RemoteCtx(None, None)
    # func = libff.invoke.DirectRemoteFunc(workerPath, "echo", ctx)
    func = libff.invoke.ProcessRemoteFunc(workerPath, "echo", ctx)
    resp = func.Invoke({"hello" : "world"})
    print(resp)


def testCuda(mode):
    ctx = libff.invoke.RemoteCtx(None, None)

    if mode == 'process':
        objType = libff.invoke.ProcessRemoteFunc
    else:
        objType = libff.invoke.DirectRemoteFunc
    f0 = objType(workerPath, "cuda", ctx, enableGpu=True)
    f1 = objType(workerPath, "cuda", ctx, enableGpu=True)

    r0 = f0.Invoke({})
    r1 = f1.Invoke({})
    print(r0)
    print(r1)

    # if r0['deviceID'] == r1['deviceID']:
    #     print("FAIL")
    #     print("Functions didn't get unique devices:")
    #     print(r0)
    #     print(r1)
    #     return

    f1.Close()

    f2 = objType(workerPath, "cuda", ctx, enableGpu=True)
    r2 = f2.Invoke({})
    print(r2)
    # if r2['deviceID'] == r0['deviceID']:
    #     print("FAIL")
    #     print("Didn't get free device after re-allocation")
    #     print(r0)
    #     print(r2)
    #     return

    print("PASS")

def testState(mode):
    ctx = libff.invoke.RemoteCtx(None, None)

    if mode == 'process':
        func = libff.invoke.ProcessRemoteFunc(workerPath, "state", ctx)
    else:
        func = libff.invoke.DirectRemoteFunc(workerPath, "state", ctx)


    resp = func.Invoke({"scratchData" : "firstData"})
    if resp['cachedData'] != 'firstData':
        print("FAIL: First invoke didn't return scrach data")
        return False

    resp = func.Invoke({})
    if resp['cachedData'] != 'firstData':
        print("FAIL: function didn't cache data")
        return False

    resp = func.Invoke({"scratchData" : "secondData"})
    if resp['cachedData'] != 'secondData':
        print("FAIL: function didn't return new data")
        return False

    resp = func.Invoke({})
    if resp['cachedData'] != 'secondData':
        print("FAIL: function didn't replace cache data")
        return False

    print("PASS")
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

def stats(mode='direct'):
    sleepTime = 1000
    repeat = 2
    ctx = libff.invoke.RemoteCtx(None, None)

    stats = libff.profCollection()
    if mode == 'direct':
        f1 = libff.invoke.DirectRemoteFunc(workerPath, "perfSim", ctx, stats=stats.mod('f1'))
        f2 = libff.invoke.DirectRemoteFunc(workerPath, "perfSim", ctx, stats=stats.mod('f2'))
    else:
        f1 = libff.invoke.ProcessRemoteFunc(workerPath, "perfSim", ctx, stats=stats.mod('f1'))
        f2 = libff.invoke.ProcessRemoteFunc(workerPath, "perfSim", ctx, stats=stats.mod('f2'))

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

    print("stats: PASS")
    return True


def testAsync():
    ctx = libff.invoke.RemoteCtx(None, None)
    # func = libff.invoke.ProcessRemoteFunc(workerPath, "perfSim", ctx)
    func = libff.invoke.DirectRemoteFunc(workerPath, "perfSim", ctx)

    start = time.time()
    fut0 = func.InvokeAsync({"runtime" : 1})
    fut1 = func.InvokeAsync({"runtime" : 1})
    reqFinishTime = time.time() - start

    start = time.time()
    resp1 = fut1.get()
    resp2 = func.Invoke({"runtime" : 1})
    resp0 = fut0.get()
    funcRuntime = time.time() - start

    if reqFinishTime >= 1:
        print("Async FAIL: InvokeAsync didn't return immediately")
        return False

    if funcRuntime < 1:
        print("Async FAIL: Future returned too soon")
        return False

    respVals = [resp0['validateMetric'], resp1['validateMetric'], resp2['validateMetric']]
    if len(set(respVals)) != len(respVals):
        print("Async FAIL: some futures returned the same response")
        print(respVals)
        return False

    print("Async: PASS")
    return True

if __name__ == "__main__":
    testCuda(mode='process')
    # if not stats(mode='direct'):
    #     sys.exit(1)

    # helloWorld()
    #
    # if not testAsync():
    #     sys.exit(1)
    #
    # if not testState('process'):
    #     sys.exit(1)
