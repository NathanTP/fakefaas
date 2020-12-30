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


def stats():
    repeat = 10
    ctx = libff.invoke.RemoteCtx(None, None)

    # func = libff.invoke.DirectRemoteFunc(workerPath, "perfSim", ctx)
    func = libff.invoke.ProcessRemoteFunc(workerPath, "perfSim", ctx)

    # Cold Start
    start = time.time()
    resp = func.Invoke({"runtime" : 1})
    coldTime = time.time() - start
    coldStats = func.Stats(reset=True)

    start = time.time()
    for i in range(repeat):
        resp = func.Invoke({"runtime" : 1})
    runtimeMeasured = (time.time() - start) / repeat

    print("Cold Start (direct): " + str(coldTime * 1000))
    print("Cold Start (from func): " + str(coldStats))
    print("")
    print("Warm Start (direct): " + str(runtimeMeasured * 1000))
    print("Warm Start (from func): " + str(func.Stats()))


if __name__ == "__main__":
    stats()
    # helloWorld()
