import pathlib
import sys
import os
import subprocess as sp
import abc
import importlib.util
import argparse
import jsonpickle as json
import traceback
import time
import copy

from . import array
from . import kv
from . import util


class InvocationError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return "Remote function invocation error: " + self.msg


cudaFreeDevs = None


# Not the end of the world, but getting the device count adds ~10ms to the
# import time so we only do it if we have to
def initCuda():
    global cudaFreeDevs
    if cudaFreeDevs is not None:
        return
    else:
        proc = sp.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                      stdout=sp.PIPE, text=True, check=True)
        cudaNDev = proc.stdout.count('\n')
        cudaFreeDevs = [i for i in range(cudaNDev)]


class RemoteFunc(abc.ABC):
    """Represents a remote (or at least logically separate) function"""

    @abc.abstractmethod
    def __init__(self, packagePath, funcName, context, clientID=0, stats=None, enableGpu=False):
        """Create a remote function from the provided package.funcName.
        arrayMnt should point to wherever child workers should look for
        arrays. Stats is a reference to a libff.profCollection() to be used for
        any internal statistics. The object will be modified in-place (no
        copies are made). However, to ensure all updates are finalized, users
        must call getStats().

        clientID identifies a unique tennant in the system. Functions from
        different clients will get independent executors while functions from
        the same client may or may not get a different executor on each call.
        ClientID -1 is reserved for system services like KaaS and may be
        treated differently."""
        pass

    @abc.abstractmethod
    def Invoke(self, arg):
        """Invoke the function with the dictionary-typed argument arg. Will
        return the response dictionary from the function."""
        pass

    @abc.abstractmethod
    def getStats(self):
        """Update the stats object passed during initialization and return a
        reference to it. You must call this to ensure that all statistics are
        available, otherwise you may only have partial results. getStats() is
        idempotent"""
        pass

    @abc.abstractmethod
    def resetStats(self):
        """Reset any internally managed stats, including the stats object
        passed at initialization"""
        pass

    @abc.abstractmethod
    def Close(self):
        """Clean up the function executor"""
        pass


class RemoteCtx():
    """Passed to remote workers when they run. Contains handles for data
    backends etc."""
    def __init__(self, arrayStore, kvStore):
        self.array = arrayStore
        self.kv = kvStore

        # These are managed by the remote func objects
        self.cudaDev = None
        self.stats = util.profCollection()

        # Can be used by the function to store intermediate ephemeral state
        # (not guaranteed to persist between calls)
        self.scratch = None


# We avoid importing and registering the same package multiple times, doing so
# is inefficient and may be incorrect (we don't require that direct function
# registration be idempotent). You still need a DirectRemoteFunc object per
# function, but the heavy state is memoized here.
_importedFuncPackages = {}


# For now we just do delayed execution instead of asynchronous because I don't
# want to deal with multiprocessing or dask or something.
class DirectRemoteFuncFuture():
    def __init__(self, arg, func):
        self.arg = arg
        self.func = func

    def get(self):
        return self.func.Invoke(self.arg)


class DirectRemoteFunc(RemoteFunc):
    """Invokes the function directly in the callers process. This will import
    the function's package.

    The package must implement a function named "LibffInvokeRegister" that
    returns a mapping of function name -> function object. This is similar to
    the 'funcs' argument to RemoteProcessServer."""

    def __init__(self, packagePath, funcName, context: RemoteCtx, clientID=0, stats=None, enableGpu=False):
        self.fName = funcName
        self.clientID = clientID

        # From now on, if packagePath is a pathlib object, it's actually a
        # path. If it's a string, then it's a module name rather than a
        # filesystem path.
        if os.path.exists(packagePath):
            self.packagePath = pathlib.Path(packagePath)
        else:
            self.packagePath = packagePath

        # The profs in ctx are used only for passing to the function,
        # self.stats is for the client-side RemoteFunc
        self.ctx = copy.copy(context)
        if stats is None:
            self.stats = util.profCollection()
        else:
            self.stats = stats

        if enableGpu:
            initCuda()
            if len(cudaFreeDevs) == 0:
                raise RuntimeError("No more GPUs available!")
            self.ctx.cudaDev = cudaFreeDevs.pop()

        # Initialization is delated to the first call to properly account for
        # cold starts.
        self.initialized = False

    # Not actually async, just delays execution until you ask for it
    def InvokeAsync(self, arg):
        return DirectRemoteFuncFuture(arg, self)

    def Invoke(self, arg):
        if not self.initialized:
            with util.timer("t_init", self.stats):
                if self.packagePath in _importedFuncPackages:
                    self.funcs = _importedFuncPackages[self.packagePath]
                else:
                    if isinstance(self.packagePath, pathlib.Path):
                        name = self.packagePath.stem
                        if self.packagePath.is_dir():
                            self.packagePath = self.packagePath / "__init__.py"

                        spec = importlib.util.spec_from_file_location(name, self.packagePath)
                    else:
                        spec = importlib.util.find_spec(self.packagePath)

                    if spec is None:
                        raise RuntimeError("Failed to load function from " + str(self.packagePath))

                    package = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(package)
                    self.funcs = package.LibffInvokeRegister()
                    _importedFuncPackages[self.packagePath] = self.funcs

        with util.timer('t_invoke', self.stats):
            if self.fName not in self.funcs:
                raise RuntimeError("Function '" + self.fName + "' not registered")

            self.ctx.stats = self.stats.mod('worker')
            res = self.funcs[self.fName](arg, self.ctx)
            return res

    def getStats(self):
        return self.stats

    def resetStats(self, newStats=None):
        if newStats is not None:
            self.stats = newStats
        else:
            self.stats.reset()

    def Close(self):
        if self.ctx.cudaDev is not None:
            cudaFreeDevs.append(self.ctx.cudaDev)


class _processExecutor():
    # executors may be shared by multiple clients. As such, they have no
    # persistent stats of their own. Each call has a stats argument that can be
    # filled in by whoever is calling.
    def __init__(self, packagePath, clientID, arrayMnt=None, enableGpu=False):
        """Represents a handle to a remote process worker. Workers are
        independent of any particular client. Many fexecutor functions take
        statistics, these will be updated with internal statistics of the
        executor object locally, but not the worker. You must manage worker
        statistics separately (using the 'getStats' command)."""
        self.arrayMnt = arrayMnt
        self.packagePath = packagePath
        self.clientID = clientID
        self.enableGpu = enableGpu

        # Next id to use for outgoing mesages
        self.nextMsgId = 0
        self.dead = False

        if enableGpu:
            if len(cudaFreeDevs) == 0:
                raise RuntimeError("No more GPUs available!")
            self.cudaDev = cudaFreeDevs.pop()
        else:
            self.cudaDev = None

        # Latest Id we have a response for. For now we assume everything is in
        # order wrt a single executor.
        self.lastResp = -1
        self.resps = {}  # reqID -> raw message

        # Python's package management is garbage, if the worker is
        # a module, you have to run it with -m, but if it's a
        # stand-alone file, you can't use -m and have to call it
        # directly.

        # if it's a module name, we still treat it like a path since everything
        # will still work.
        self.packagePath = pathlib.Path(self.packagePath)
        if self.packagePath.is_file():
            cmd = ["python3", str(self.packagePath.name)]
        else:
            # Could be a real path or it could be an installed package
            cmd = ["python3", "-m", str(self.packagePath.name)]

        if self.arrayMnt is not None:
            cmd += ['-m', self.arrayMnt]

        if self.cudaDev is not None:
            cmd += ['-g', str(self.cudaDev)]

        self.proc = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, text=True, cwd=self.packagePath.parent)
        self.ready = False
        self.blocking = True

    def _setBlock(self, block):
        if self.blocking == block:
            return
        else:
            os.set_blocking(self.proc.stdout.fileno(), block)
            self.blocking = block

    def waitReady(self, stats=None, block=True):
        """Optional block until the function is ready. This is mostly just
        useful for profiling since send() is safe to call before the executor
        is ready. If block==False, this polls for readiness. Returns True if
        the executor is ready, false otherwise (False can only be returned if
        block==False)"""

        self._setBlock(block)
        if self.ready is False:
            # t_init really only measures from the time you wait for it, not
            # the true init time (hard to measure that without proper
            # asynchrony)
            with util.timer("t_init", stats):
                readyMsg = self.proc.stdout.readline()

            if not block and readyMsg == "":
                return False

            if readyMsg != "READY\n":
                self.proc.terminate()
                output = self.proc.stdout.read()
                raise InvocationError("Executor failed to initialize: " + output)

            self.ready = True
            return True

    def send(self, msg, funcID, stats=None):
        if self.dead:
            raise InvocationError("Tried to send to a dead executor")

        msgId = self.nextMsgId
        self.nextMsgId += 1

        msg['funcID'] = funcID
        with util.timer('t_requestEncode', stats):
            jMsg = json.dumps(msg) + "\n"
            self.proc.stdin.write(jMsg)
            self.proc.stdin.flush()

        return msgId

    def recv(self, reqId, funcID, stats=None, block=True):
        """Recieve the response for the request with ID reqId. Users may only
        recv each ID exactly once. Responses are expected to come in order."""

        self._setBlock(block)
        if not self.ready:
            if not self.waitReady(stats=stats, block=block):
                return None

        while self.lastResp < reqId:
            resp = self.proc.stdout.readline()
            if not block and resp == "":
                return None
            else:
                self.lastResp += 1
                self.resps[self.lastResp] = resp

        with util.timer('t_responseDecode', stats):
            try:
                resp = json.loads(self.resps[reqId])
            except Exception:
                raise InvocationError(self.resps[reqId])

        if resp['error'] is not None:
            raise InvocationError(resp['error'])

        return resp['resp']

    def getConfig(self):
        """Returns a tuple of the configuration properties for this executor.
        Configuration properties uniquely identify functionality of the
        executor. i.e. executors with the same config are interchangeable
        w.r.t. client requests."""
        return (self.packagePath, self.clientID, self.enableGpu, self.arrayMnt)

    def kill(self, force=False):
        """Wait for all outstanding work to complete, then kill the worker.
        Users can still call recv on this object, but they can no longer send.

        This operation is synchronous in order to ensure that all resources are
        indeed freed."""
        self._setBlock(True)
        while self.lastResp < self.nextMsgId - 1:
            self.lastResp += 1
            self.resps[self.lastResp] = self.proc.stdout.readline()

        self.proc.stdin.close()
        self.proc.wait()
        self.dead = True

        if self.enableGpu:
            cudaFreeDevs.append(self.cudaDev)


class _processPool():
    def __init__(self):
        self.execs = []

    def getExecutor(self, packagePath, clientID, arrayMnt=None, stats=None, enableGpu=False):
        if enableGpu:
            initCuda()

        executor = None
        config = (packagePath, clientID, enableGpu, arrayMnt)
        for e in self.execs:
            if e.getConfig() == config:
                executor = e
                break

        if executor is None:
            if enableGpu and len(cudaFreeDevs) == 0:
                # Gonna have to kill a gpu-enabled executor to get enough resources
                for i in range(len(self.execs)):
                    e = self.execs[i]
                    if e.enableGpu:
                        e.kill()
                        del self.execs[i]
                        break

            # t_init will be finalized in waitReady. This implies that you have to
            # pass the same stats to both these functions...
            with util.timer('t_init', stats, final=False):
                executor = _processExecutor(packagePath, clientID, arrayMnt=arrayMnt, enableGpu=enableGpu)
                self.execs.append(executor)

        return executor

    def destroy(self):
        for ex in self.execs:
            ex.kill()

        self.execs = []


class ProcessRemoteFuncFuture():
    def __init__(self, reqID, proc, funcID, stats=None):
        self.reqID = reqID
        self.proc = proc
        self.stats = stats
        self.funcID = funcID

    def get(self, block=True):
        with util.timer('t_futureWait', self.stats):
            resp = self.proc.recv(self.reqID, self.funcID, stats=self.stats, block=block)
        return resp


def DestroyFuncs():
    """Creating a RemoteFunction may introduce global state, this function
    resets that state to the extent possilbe. Note that true cleaning may not
    be possible in all cases, the only way to completely reset state is to
    restart the process."""
    _processExecutors.destroy()


# There may be multiple handles for the same function, we need to disambiguate these for various reasons (mostly stats). This global ensures unique ids.
_processFuncNextID = 0


class ProcessRemoteFunc(RemoteFunc):
    def __init__(self, packagePath, funcName, context, clientID=0, stats=None, enableGpu=False):
        self.stats = stats
        self.clientID = clientID

        global _processFuncNextID
        self.funcID = _processFuncNextID
        _processFuncNextID += 1

        # ID of the next request to make and the last completed request (respectively)
        self.nextID = 0
        self.lastCompleted = -1
        # Maps completed request IDs to their responses, responses are removed once _awaitResp is called
        self.completedReqs = {}

        self.fname = funcName
        self.packagePath = packagePath
        self.ctx = copy.copy(context)
        self.enableGpu = enableGpu

    def InvokeAsync(self, arg):
        proc = _processExecutors.getExecutor(self.packagePath, self.clientID, stats=self.stats, enableGpu=self.enableGpu)

        req = {"command": "invoke",
               "stats": util.profCollection(),
               "fName": self.fname,
               "fArg": arg}

        msgId = proc.send(req, self.funcID, stats=self.stats)

        fut = ProcessRemoteFuncFuture(msgId, proc, self.funcID, self.stats)
        return fut

    def Invoke(self, arg):
        with util.timer('t_invoke', self.stats):
            fut = self.InvokeAsync(arg)
            resp = fut.get()
        return resp

    def getStats(self):
        """Update the stats object passed during initialization and return a
        reference to it. You must call this to ensure that all statistics are
        available, otherwise you may only have partial results. getStats() is
        idempotent"""
        # XXX strictly speaking, it's possible to get a different executor here, not sure how to deal with it
        proc = _processExecutors.getExecutor(self.packagePath, self.clientID, enableGpu=self.enableGpu)
        statsReq = {
                'command': 'reportStats',
                # We always reset the worker since we now have those stats in
                # our local stats and don't want to count them twice
                'reset': True
        }
        msgID = proc.send(statsReq, self.funcID)
        workerStats = proc.recv(msgID, self.funcID)

        if self.stats is None:
            return None
        else:
            self.stats.mod('worker').merge(workerStats)
            return self.stats

    def resetStats(self, newStats=None):
        # getStats resets everything on the client
        self.getStats()
        if newStats is None:
            self.stats.reset()
        else:
            self.stats = newStats

    def Close(self):
        # State is global now, so there's nothing to clean, might even remove
        # this function completely...
        pass


def _remoteServerRespond(msg):
    print(json.dumps(msg), flush=True)


def RemoteProcessServer(funcs, serverArgs):
    """Begin serving requests from stdin (your module is being called as a
    process). This function will return when the client is done with you
    (closes stdin). Server args are generally provided by libff.invoke on the
    command line (you should usually pass sys.argv).

    funcs is a dictionary mapping cannonical function names (what a remote
    invoker would call) to the actual python function object"""

    parser = argparse.ArgumentParser(description="libff invocation server")
    parser.add_argument("-m", "--mount", help="Directory where libff.array's are mounted")
    parser.add_argument("-k", "--kv", action="store_true", help="Enable the key-value store")
    parser.add_argument("-g", "--gpu", type=int, help="Specify a GPU to use (otherwise no GPU will be available)")
    args = parser.parse_args(serverArgs)

    if args.mount is not None:
        arrStore = array.ArrayStore('file', args.mount)
    else:
        arrStore = None

    objStore = kv.Redis(pwd=util.redisPwd, serialize=True)
    ctx = RemoteCtx(arrStore, objStore)

    # Global stats are maintained to keep stats reporting off the critical path
    # (serializing profCollection adds non-trivial overheads)
    # {funcID -> libff.profCollection()}
    stats = {}

    print("READY", flush=True)

    for rawReq in sys.stdin:
        start = time.time()
        try:
            req = json.loads(rawReq)
        except json.decoder.JSONDecodeError as e:
            err = "Failed to parse command (must be valid JSON): " + str(e)
            _remoteServerRespond({"error": err})
            continue
        decodeTime = time.time() - start

        try:
            if req['command'] == 'invoke':
                if req['funcID'] not in stats:
                    stats[req['funcID']] = util.profCollection()
                ctx.profs = stats[req['funcID']]
                ctx.cudaDev = args.gpu
                if req['fName'] in funcs:
                    resp = funcs[req['fName']](req['fArg'], ctx)
                else:
                    _remoteServerRespond({"error": "Unrecognized function: " + req['fName']})

                ctx.profs['t_requestDecode'].increment(decodeTime*1000)
                with util.timer('t_responseEncode', ctx.profs):
                    _remoteServerRespond({"error": None, "resp": resp})
            elif req['command'] == 'reportStats':
                if req['funcID'] not in stats:
                    respStats = util.profCollection()
                else:
                    respStats = stats[req['funcID']]
                _remoteServerRespond({"error": None, "resp": respStats})
                if req['reset']:
                    stats.pop(req['funcID'], None)
            else:
                _remoteServerRespond({"error": "Unrecognized command: " + req['command']})
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            _remoteServerRespond({"error": "Unhandled internal error: " + repr(e)})


# ==============================================================================
# Global Init
# ==============================================================================
_processExecutors = _processPool()
