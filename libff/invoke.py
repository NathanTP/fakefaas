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


class ProcessRemoteFuncFuture():
    def __init__(self, reqID, func):
        self.reqID = reqID
        self.func = func

    def get(self):
        return self.func._awaitResp(self.reqID)


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


    def _getProc(self):
        """Returns a Popen object suitable for executing this remote function"""
        if self.ctx.array is not None:
            arrayMnt = self.ctx.array.FileMount
        else:
            arrayMnt = ""

        if self.packagePath in _runningFuncProcesses:
            proc = _runningFuncProcesses[self.packagePath]
        else:
            # Python's package management is garbage, if the worker is
            # a module, you have to run it with -m, but if it's a
            # stand-alone file, you can't use -m and have to call it
            # directly.
            if self.packagePath.is_dir():
                cmd = ["python3", "-m", str(self.packagePath.name)]
            else:
                cmd = ["python3", str(self.packagePath.name)]

            if self.ctx.array is not None:
                arrayMnt = self.ctx.array.FileMount
                cmd += ['-m', arrayMnt]

            proc = sp.Popen(cmd, stdin=sp.PIPE, stdout=sp.PIPE, text=True, cwd=self.packagePath.parent)

            _runningFuncProcesses[self.packagePath] = proc

        return proc


    def InvokeAsync(self, arg):
        proc = self._getProc()

        req = { "command" : "invoke",
                "stats" : util.profCollection(),
                "fName" : self.fname,
                "fArg" : arg }

        # For some reason this masks some errors (e.g. failed Redis connection). I can't figure out why.
        # if self.proc.poll() is None:
        #     out, err = self.proc.communicate()
        #     raise InvocationError("Function process exited unexpectedly: stdout\n" + str(out) + "\nstderr:\n" + str(err))

        with util.timer('t_requestEncode', self.stats):
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()

        fut = ProcessRemoteFuncFuture(self.nextID, self)
        self.nextID += 1
        return fut


    def _awaitResp(self, reqID):
        """Wait until request 'reqID' has been processed and return its response"""
        proc = self._getProc()
        while self.lastCompleted < reqID:
            self.lastCompleted += 1
            self.completedReqs[self.lastCompleted] = proc.stdout.readline()

        # Decoding is deferred until future await to make errors more clear
        rawResp = self.completedReqs.pop(reqID)
        with util.timer('t_responseDecode', self.stats):
            try:
                resp = json.loads(rawResp)
            except:
                raise InvocationError(rawResp)
        
        if resp['error'] is not None:
            raise InvocationError(resp['error'])


        if self.stats is not None:
            self.stats.mod('worker').merge(resp['stats'])

        return resp['resp']


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
        proc = self._getProc()
        del _runningFuncProcesses[self.packagePath]
        proc.stdin.close()
        proc.wait()


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
