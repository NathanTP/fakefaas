import json
import subprocess as sp
import pathlib
import sys
from . import kv
import signal
from .util import *
import importlib.util

# Allow importing models from sibling directory (python nonsense) 
# sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
# import models.ferplus as ferplus
# import models.bertsquad as bertsquad 

class InvocationError():
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return msg

modelsDir = (pathlib.Path(__file__).parent.parent / "models").resolve()

class LocalModel:
    def __init__(self, modelPath, objStore, provider="CUDA_ExecutionProvider"):
        self.objStore = objStore
        self.times = {}

        spec = importlib.util.spec_from_file_location(modelPath.stem, modelPath)
        modelPackage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modelPackage)
        self.model = modelPackage.Model(provider=provider, profTimes=self.times)

    def pre(self, name, inputKey=None):
        if inputKey is None:
            inputKey = name+".in"

        inputs = self.objStore.get(inputKey, self.times)
        with timer("pre", self.times):
            ret = self.model.pre(inputs)

        self.objStore.put(name+".pre", ret, self.times)
        return name+".pre"
    

    def run(self, name):
        inputs = self.objStore.get(name+".pre", self.times)

        with timer("run", self.times):
            ret = self.model.run(inputs)

        self.objStore.put(name+".run", ret, self.times)
        return name+".run"


    def post(self, name):
        inputs = self.objStore.get(name+".run", self.times)

        with timer("post", self.times):
            ret = self.model.post(inputs)
        
        self.objStore.put(name+".final", ret, self.times)
        return name+".final"


    def inputs(self, name):
        inputs = self.model.inputs()
        self.objStore.put(name+".in", inputs)
        return name+".in"

    def close(self):
        return self.times
        pass

class RemoteModel:
    def __init__(self, modelPath, objStore, provider="CUDA_ExecutionProvider"):
        """Create a new model executor for modelName. Arguments will be passed through objStore."""
        self.objStore = objStore
        self.provider = provider

        self.proc = sp.Popen(["python3", str(modelPath)], bufsize=1, stdin=sp.PIPE, stdout=sp.PIPE, text=True)

        # Note: local models would wait until everything was ready before
        # returning. For remote, that's all hapening in a different process so
        # it can overlap with local computation. The remote init time will show
        # up as a slower first invocation of a function. I'm not sure how best
        # to report this.


    def _invoke(self, arg):
        self.proc.stdin.write(json.dumps(arg) + "\n")
        rawResp = self.proc.stdout.readline()
        resp = json.loads(rawResp)
        if resp['error'] is not None:
            raise InvocationError(resp['error'])
        return resp


    def pre(self, name, inputKey=None):
        if inputKey is None:
            inputKey = name+".in"

        req = {
            "func" : "pre",
            "provider" : self.provider,
            "inputKey" : inputKey,
            "outputKey" : name+".pre"
        }

        self._invoke(req)
        return name+".pre"


    def run(self, name):
        req = {
            "func" : "run",
            "provider" : self.provider,
            "inputKey" : name+".pre",
            "outputKey" : name+".run"
        }

        self._invoke(req)
        return name+".run"


    def post(self, name):
        req = {
            "func" : "post",
            "provider" : self.provider,
            "inputKey" : name+".run",
            "outputKey" : name+".final"
        }

        self._invoke(req)
        return name+".final"

    def inputs(self, name):
        req = {
            "func" : "inputs",
            "provider" : self.provider,
            "inputKey" : None,
            "outputKey" : name+".in"
        }

        self._invoke(req)
        return name+".in"

    def close(self):
        req = { "func" : "reportStats" }
        resp = self._invoke(req)

        self.proc.stdin.close()
        self.proc.wait()

        return { name : prof(fromDict=profile) for name, profile in resp['times'].items() }


argFields = [
        "func", # Function to invoke (either "pre", "run", or "post")
        "provider", # onnxruntime provider 
        "inputKey", # Key name to read from for input
        "outputKey", # Key name that outputs should be written to
        ]

def remoteServer(modelClass):
    def onExit(sig, frame):
        print("Function executor exiting")
        sys.exit(0)

    signal.signal(signal.SIGINT, onExit)

    times = {}

    # Eagerly set up any needed state. It's not clear how realistic this is,
    # might switch it around later.
    with timer("init", times):
        with timer("imports", times):
            modelClass.imports()

        cpuTimes = {}
        gpuTimes = {}
        modelStates = {
                "CPUExecutionProvider" : modelClass(provider="CPUExecutionProvider", profTimes=cpuTimes),
                "CUDAExecutionProvider" : modelClass(provider="CUDAExecutionProvider", profTimes=gpuTimes)
                }
        mergeTimers(times, cpuTimes, "cpuModel.")
        mergeTimers(times, gpuTimes, "gpuModel.")

    objStore = kv.Redis(pwd="Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy", serialize=True)

    for rawCmd in sys.stdin:
        try:
            cmd = json.loads(rawCmd)
        except json.decoder.JSONDecodeError as e:
            err = "Failed to parse command (must be valid JSON): " + str(e)
            print(json.dumps({ "error" : err }), flush=True)
            continue

        if cmd['func'] in ['reportStats']:
            # System commands
            if cmd['func'] == 'reportStats':
                resp = {}
                resp['times'] = { name : timer.__dict__ for name, timer in times.items() }
                resp['error'] = None
                print(json.dumps(resp), flush=True)
            else:
                err = "unrecognized function " + str(cmd['func'])
                print(json.dumps({ "error" : err }), flush=True)
            continue
        else:
            # Function commands
            curModel = modelStates[cmd['provider']]

            if cmd['func'] == "pre":
                funcInputs = objStore.get(cmd['inputKey'], times)
                with timer("pre", times):
                    funcOut = curModel.pre(funcInputs)
            elif cmd['func'] == 'run':
                funcInputs = objStore.get(cmd['inputKey'], times)
                with timer("run", times):
                    funcOut = curModel.run(funcInputs)
            elif cmd['func'] == 'post':
                funcInputs = objStore.get(cmd['inputKey'], times)
                with timer("post", times):
                    funcOut = curModel.post(funcInputs)
            elif cmd['func'] == 'inputs':
                funcOut = curModel.inputs()
            else:
                err = "unrecognized function " + str(cmd['func'])
                print(json.dumps({ "error" : err }), flush=True)
                continue

            objStore.put(cmd['outputKey'], funcOut, times)
            print(json.dumps({"error" : None}), flush=True)
            continue
