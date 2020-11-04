import cProfile
import pathlib
from contextlib import contextmanager
import models.ferplus as ferplus
import models.bertsquad as bertsquad 
import time
import argparse
import redis
import pickle
import libff as ff
import libff.kv
import ModelWorker

modelDir = pathlib.Path(__file__).parent.resolve() / "models"

def runTest(state, objStore, profTimes, niter=1):
    # This is generally not counted against runtime as we assume the inputs are
    # already loaded somewhere
    # inputs = state.inputs()
    inputKey = state.inputs("runTest")

    for repeat in range(niter):
        with ff.timer("e2e", profTimes):
            iterID = str(repeat) 

            with ff.timer("pre", profTimes):
                processedKey = state.pre(iterID, inputKey=inputKey)

            with ff.timer("run", profTimes):
                modelOutKey = state.run(iterID)

            with ff.timer("post", profTimes):
                finalKey = state.post(iterID)

        objStore.delete(processedKey, modelOutKey, finalKey)

    objStore.delete(inputKey)
    stateTimes = state.close()
    ff.mergeTimers(profTimes, stateTimes, "model.")   

def main(args):
    if args.serialize:
        objStore = ff.kv.Redis(pwd="Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy", serialize=True)
    else:
        objStore = ff.kv.Local(copyObjs=False, serialize=False)
    libffCtx = libff.invoke.RemoteCtx(None, objStore)

    if args.cpu:
        provider = "CPUExecutionProvider"
    else:
        provider = "CUDAExecutionProvider"

    times = {}
    
    with ff.timer("init", times):
        modelTimes = {}
        if args.remote:
            state = ModelWorker.remoteModel(modelDir / args.model / (args.model + ".py"), provider, "process", libffCtx)
        else:
            state = ModelWorker.remoteModel(modelDir / args.model / (args.model + ".py"), provider, "direct", libffCtx)

    with ff.timer("run", times):
        runTest(state, objStore, times, niter=args.niter)

    # prof_file = state.session.end_profiling()
    # print("onnxruntime profile at: ", prof_file)

    print("Times (ms): ")
    ff.printTimers(times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for faas-like onnxruntime benchmarking")
    parser.add_argument("-c", "--cpu", action='store_true',
            help="use CPU execution provider (rather than the default CUDA provider)")
    parser.add_argument("-n", "--niter", type=int, default=1, help="Number of test iterations to perform")
    parser.add_argument("-m", "--model", type=str, default="ferplus", help="Which model to run, either 'bertsquad' or 'ferplus'")
    parser.add_argument("-r", "--remote", action='store_true', help="Run the model as a remote executor rather than in the same process. remote==True implies serialize.")
    parser.add_argument("-s", "--serialize", action='store_true', help="Serialize data (and store in kv store) in between steps. You must have a redis server running for this.")

    args = parser.parse_args()

    if args.remote:
        args.serialize = True

    main(args)

    # Cprofile
    # profFile = "fer10kGPU.prof"
    # cProfile.runctx("main(model, niter=args.niter, use_gpu = not args.cpu, serialize=args.serialize)", filename=profFile)
    # print("Profile at: ", profFile)
