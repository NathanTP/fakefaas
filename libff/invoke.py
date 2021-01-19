import pathlib
import sys
import subprocess as sp
import abc
import importlib.util
import argparse
import jsonpickle as json
import traceback
from . import array
from . import kv 
from . import util

class InvocationError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return "Remote function invocation error: " + self.msg


class RemoteFunc(abc.ABC):
    """Represents a remote (or at least logically separate) function"""

    @abc.abstractmethod
    def __init__(self, packagePath, funcName, context, stats=None):
        """Create a remote function from the provided package.funcName.
        arrayMnt should point to wherever child workers should look for
        arrays."""
        pass
    

    @abc.abstractmethod
    def Invoke(self, arg):
        """Invoke the function with the dictionary-typed argument arg. Will
        return the response dictionary from the function."""
        pass

    def Stats(self, reset=False):
        """Report the statistics collected so far. If reset is True, stats will
        be cleared."""
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

    def __init__(self, packagePath, funcName, context, stats=None):
        self.fName = funcName

        # The profs in ctx are used only for passing to the function,
        # self.stats is for the client-side RemoteFun
        self.ctx = context
        if stats is None:
            self.stats = util.profCollection()
        else:
            self.stats = stats 

        with util.timer("t_init", self.stats):
            if packagePath in _importedFuncPackages:
                self.funcs = _importedFuncPackages[packagePath]
            else:
                name = packagePath.stem
                if packagePath.is_dir():
                    packagePath = packagePath / "__init__.py"

                spec = importlib.util.spec_from_file_location(name, packagePath)
                if spec is None:
                    raise RuntimeError("Failed to load function from " + str(packagePath))

                package = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(package)
                self.funcs = package.LibffInvokeRegister()
                _importedFuncPackages[packagePath] = self.funcs 


    # Not actually async, just delays execution until you ask for it
    def InvokeAsync(self, arg):
        return DirectRemoteFuncFuture(arg, self)


    def Invoke(self, arg):
        with util.timer('t_invoke', self.stats):
            if self.fName not in self.funcs:
                raise RuntimeError("Function '" + self.fName + "' not registered")

            self.ctx.profs = self.stats.mod('worker')
            return self.funcs[self.fName](arg, self.ctx)


    def Stats(self, reset=False):
        report = self.stats.report()
        if reset:
            self.stats.reset()

        return report 


    def Close(self):
        pass


class _processExecutor():
    # executors may be shared by multiple clients. As such, they have no
    # persistent stats of their own. Each call has a stats argument that can be
    # filled in by whoever is calling.
    def __init__(self, packagePath, arrayMnt=None):
        self.arrayMnt = arrayMnt
        self.packagePath = packagePath
        # Next id to use for outgoing mesages
        self.nextId = 0

        # Latest Id we have a response for. For now we assume everything is in
        # order wrt a single executor.
        self.lastResp = -1
        self.resps = {} # reqID -> raw message

        # Python's package management is garbage, if the worker is
        # a module, you have to run it with -m, but if it's a
        # stand-alone file, you can't use -m and have to call it
        # directly.
        if self.packagePath.is_dir():
            cmd = ["python3", "-m", str(self.packagePath.name)]
        else:
            cmd = ["python3", str(self.packagePath.name)]

        if self.arrayMnt is not None:
            cmd += ['-m', self.arrayMnt]

        self.proc = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, text=True, cwd=self.packagePath.parent)
        self.ready = False

    def waitReady(self, stats=None):
        """Optional block until the function is ready. This is mostly just
        useful for profiling since send() is safe to call before the executor
        is ready."""
        if self.ready == False:
            # t_init really only measures from the time you wait for it, not
            # the true init time (hard to measure that without proper
            # asynchrony)
            with util.timer("t_init", stats):
                readyMsg = self.proc.stdout.readline()
            if readyMsg != "READY\n":
                raise InvocationError("Executor failed to initialize: "+readyMsg)
        self.ready = True


    def send(self, msg, stats=None):
        msgId = self.nextId
        self.nextId += 1

        with util.timer('t_requestEncode', stats):
            jMsg = json.dumps(msg) + "\n"

        self.proc.stdin.write(jMsg)
        self.proc.stdin.flush()

        return msgId


    def recv(self, reqId, stats=None):
        """Recieve the response for the request with ID reqId. Users may only
        recv each ID exactly once."""
        if not self.ready:
            self.waitReady(stats=stats)

        while self.lastResp < reqId:
            self.lastResp += 1
            self.resps[self.lastResp] = self.proc.stdout.readline()

        with util.timer('t_responseDecode', stats):
            try:
                resp = json.loads(self.resps[reqId])
            except:
                raise InvocationError(rawResp)

        if resp['error'] is not None:
            raise InvocationError(resp['error'])

        if stats is not None:
            stats.mod('worker').merge(resp['stats'])

        return resp['resp']


    def destroy(self):
        self.proc.stdin.close()
        self.proc.wait()


class _processPool():
    def __init__(self):
        # packagePath -> [ _processExecutor ]
        self.execs = {}

    #XXX deal with arrayMnt
    def getExecutor(self, packagePath, arrayMnt=None):
        # For now there's only ever one executor per package. This will change eventually.
        if packagePath not in self.execs:
            self.execs[packagePath] = [ _processExecutor(packagePath, arrayMnt=arrayMnt) ]

        pkgPool = self.execs[packagePath]
        return pkgPool[0]

    def destroy(self):
        for ex in self.execs.values():
            ex.destroy()

        self.execs = {}


class ProcessRemoteFuncFuture():
    def __init__(self, reqID, proc, stats=None):
        self.reqID = reqID
        self.proc = proc
        self.stats = stats

    def get(self):
        with util.timer('t_futureWait', self.stats):
            resp = self.proc.recv(self.reqID, stats=self.stats)
        return resp


def DestroyFuncs():
    """Creating a RemoteFunction may introduce global state, this function
    resets that state to the extent possilbe. Note that true cleaning may not
    be possible in all cases, the only way to completely reset state is to
    restart the process."""
    _processExecutors.destroy()


# This is a lazy way of caching backend processes for ProcessRemoteFunc
# A major limitation at the moment is that you can't have multiple interleaved
# ProcessRemoteFunc objects. Once you create a new processRemoteFunc object,
# you must not use any older ones for the same package. Obviously this isn't
# ideal, I'll fix it once we need concurrency.
_runningFuncProcesses = {}
class ProcessRemoteFunc(RemoteFunc):
    def __init__(self, packagePath, funcName, context, stats=None):
        self.stats = stats

        # ID of the next request to make and the last completed request (respectively)
        self.nextID = 0 
        self.lastCompleted = -1
        # Maps completed request IDs to their responses, responses are removed once _awaitResp is called
        self.completedReqs = {}

        self.fname = funcName
        self.packagePath = packagePath
        self.ctx = context


    def InvokeAsync(self, arg):
        proc = _processExecutors.getExecutor(self.packagePath)

        req = { "command" : "invoke",
                "stats" : util.profCollection(),
                "fName" : self.fname,
                "fArg" : arg }

        msgId = proc.send(req, stats=self.stats)

        fut = ProcessRemoteFuncFuture(msgId, proc, self.stats)
        return fut


    def Invoke(self, arg):
        with util.timer('t_invoke', self.stats):
            fut = self.InvokeAsync(arg)
            resp = fut.get()
        return resp 


    def Stats(self, reset=False):
        if self.stats is None:
            return {}
        else:
            report = self.stats.report()
            if reset:
                self.stats.reset()

            return report 


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
    args = parser.parse_args(serverArgs)

    if args.mount is not None:
        arrStore = array.ArrayStore('file', args.mount)
    else:
        arrStore = None

    objStore = kv.Redis(pwd="Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy", serialize=True)
    ctx = RemoteCtx(arrStore, objStore)

    print("READY", flush=True)

    for rawReq in sys.stdin:
        try:
            req = json.loads(rawReq)
        except json.decoder.JSONDecodeError as e:
            err = "Failed to parse command (must be valid JSON): " + str(e)
            _remoteServerRespond({ "error" : err })
            continue

        try:
            if req['command'] == 'invoke':
                if req['fName'] in funcs:
                    ctx.profs = req['stats']
                    resp = funcs[req['fName']](req['fArg'], ctx)
                else:
                    _remoteServerRespond({"error" : "Unrecognized function: " + req['fName']})

                _remoteServerRespond({"error" : None, "resp" : resp, 'stats' : ctx.profs})

            elif req['command'] == 'reportStats':
                _remoteServerRespond({"error" : None, "stats" : {}})

            else:
                _remoteServerRespond({"error" : "Unrecognized command: " + req['command']})
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            _remoteServerRespond({"error" : "Unhandled internal error: " + repr(e)})

# ==============================================================================
# Global Init
# ==============================================================================
_processExecutors = _processPool()
