import pathlib
import time

import libff
from libff import kv, array, invoke

workerPath = pathlib.Path(__file__).parent.resolve() / "worker.py"

def helloWorld():
    ctx = libff.invoke.RemoteCtx(None, None)
    # func = libff.invoke.DirectRemoteFunc(workerPath, "echo", ctx)
    func = libff.invoke.ProcessRemoteFunc(workerPath, "echo", ctx)
    resp = func.Invoke({"hello" : "world"})
    print(resp)


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


def stats():
    repeat = 2
    ctx = libff.invoke.RemoteCtx(None, None)

    # func = libff.invoke.DirectRemoteFunc(workerPath, "perfSim", ctx)
    func = libff.invoke.ProcessRemoteFunc(workerPath, "perfSim", ctx)

    # Cold Start
    start = time.time()
    resp = func.Invoke({"runtime" : 1})
    coldTime = (time.time() - start)*1000
    coldStats = func.Stats(reset=True)

    start = time.time()
    for i in range(repeat):
        resp = func.Invoke({"runtime" : 1})
    runtimeMeasured = ((time.time() - start) / repeat)*1000

    fail = False
    if coldStats['WorkerStats']['runtime'] < 1000 or coldTime < 1000:
        fail = True
        print("FAIL: runtime too fast")

    if fail:
        print("Cold Start (direct): " + str(coldTime))
        print("Cold Start (from func): " + str(coldStats))
        print("")
        print("Warm Start (direct): " + str(runtimeMeasured))
        print("Warm Start (from func): " + str(func.Stats()))
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
    # if not stats():
    #     sys.exit(1)

    # helloWorld()

    # if not testAsync():
    #     sys.exit(1)

    if not testState('process'):
        sys.exit(1)
