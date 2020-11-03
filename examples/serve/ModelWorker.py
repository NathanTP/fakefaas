"""Utilities for models to run as remote workers for serve.py (uses libff)"""
import libff
import libff.kv
import libff.invoke

class remoteModel():
    """Used by a client wishing to interact with remote model functions"""

    def __init__(self, modelPath, provider, mode):
        self.provider = provider
        if mode == 'direct':
            self._pre = libff.invoke.DirectRemoteFunc(modelPath, "pre", None)
            self._run = libff.invoke.DirectRemoteFunc(modelPath, "run", None)
            self._post = libff.invoke.DirectRemoteFunc(modelPath, "post", None)
            self._inputs = libff.invoke.DirectRemoteFunc(modelPath, "inputs", None)
        else:
            self._pre = libff.invoke.ProcessRemoteFunc(modelPath, "pre", "")
            self._run = libff.invoke.ProcessRemoteFunc(modelPath, "run", "")
            self._post = libff.invoke.ProcessRemoteFunc(modelPath, "post", "")
            self._inputs = libff.invoke.ProcessRemoteFunc(modelPath, "inputs", "")
        
    def pre(self, name, inputKey=None):
        if inputKey is None:
            inputKey = name+".in"

        req = {
            "provider" : self.provider,
            "inputKey" : inputKey,
            "outputKey" : name+".pre"
        }

        self._pre.Invoke(req)
        return name+".pre"


    def run(self, name):
        req = {
            "provider" : self.provider,
            "inputKey" : name+".pre",
            "outputKey" : name+".run"
        }

        self._run.Invoke(req)
        return name+".run"


    def post(self, name):
        req = {
            "provider" : self.provider,
            "inputKey" : name+".run",
            "outputKey" : name+".final"
        }

        self._post.Invoke(req)
        return name+".final"

    def inputs(self, name):
        req = {
            "provider" : self.provider,
            "inputKey" : None,
            "outputKey" : name+".in"
        }

        self._inputs.Invoke(req)
        return name+".in"

    def close(self):
        # XXX Not gonna bother with stats yet, add back later
        # self._pre.Close()
        # self._run.Close()
        # self._post.Close()
        # self._inputs.Close()
        return {}

   
class modelServer():
    """Used by models wishing to be run as ProcessRemoteFunctions."""
    def __init__(self, modelClass):
        self.times = {}

        # Eagerly set up any needed state. It's not clear how realistic this is,
        # might switch it around later.
        with libff.timer("init", self.times):
            with libff.timer("imports", self.times):
                modelClass.imports()

            cpuTimes = {}
            gpuTimes = {}
            self.modelStates = {
                    "CPUExecutionProvider" : modelClass(provider="CPUExecutionProvider", profTimes=cpuTimes),
                    "CUDAExecutionProvider" : modelClass(provider="CUDAExecutionProvider", profTimes=gpuTimes)
                    }
            libff.mergeTimers(self.times, cpuTimes, "cpuModel.")
            libff.mergeTimers(self.times, gpuTimes, "gpuModel.")

        self.objStore = libff.kv.Redis(pwd="Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy", serialize=True)


    def reportStats(self, req):
        # XXX We should really figure out a more generic way of handling this,
        # libff has its own reportStats. Ideally we'd not duplicate that
        # effort.
        resp = {}
        resp['times'] = { name : libff.timer.__dict__ for name, libff.timer in self.times.items() }
        resp['error'] = None
        return resp


    def pre(self, req):
        with libff.timer("pre", self.times):
            curModel = self.modelStates[req['provider']]
            funcInputs = self.objStore.get(req['inputKey'], self.times)
            funcOut = curModel.pre(funcInputs)
            self.objStore.put(req['outputKey'], funcOut, self.times)
        return { "error" : None }


    def run(self, req):
        with libff.timer("run", self.times):
            curModel = self.modelStates[req['provider']]
            funcInputs = self.objStore.get(req['inputKey'], self.times)
            funcOut = curModel.run(funcInputs)
            self.objStore.put(req['outputKey'], funcOut, self.times)
        return { "error" : None }


    def post(self, req):
        with libff.timer("post", self.times):
            curModel = self.modelStates[req['provider']]
            funcInputs = self.objStore.get(req['inputKey'], self.times)
            funcOut = curModel.post(funcInputs)
            self.objStore.put(req['outputKey'], funcOut, self.times)
        return { "error" : None }


    def inputs(self, req):
        curModel = self.modelStates[req['provider']]
        funcOut = curModel.inputs()
        self.objStore.put(req['outputKey'], funcOut, self.times)
        return { "error" : None }


def getFuncMap(serverObj):
    """libff.invoke.RemoteProcessServer expects a map of functions that can be
    invoked for this package. This function converts a model server object into
    such a mapping."""

    return {
        "reportStats" : serverObj.reportStats,
        "pre" : serverObj.pre,
        "run" : serverObj.run,
        "post" : serverObj.post,
        "inputs" : serverObj.inputs,
    }
