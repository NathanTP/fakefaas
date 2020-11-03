import pathlib
import sys
import subprocess as sp
import abc
import importlib.util
import argparse
import json
from . import array

class InvocationError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return "Remote function invocation error: " + self.msg


class RemoteFunc(abc.ABC):
    """Represents a remote (or at least logically separate) function"""

    @abc.abstractmethod
    def __init__(self, packagePath, funcName, arrayMnt):
        """Create a remote function from the provided package.funcName.
        arrayMnt should point to wherever child workers should look for
        arrays."""
        pass
    

    @abc.abstractmethod
    def Invoke(self, arg):
        """Invoke the function with the dictionary-typed argument arg. Will
        return the response dictionary from the function."""
        pass

    @abc.abstractmethod
    def Close(self):
        """Clean up the function executor and report any accumulated statistics"""
        pass


# We avoid importing and registering the same package multiple times, doing so
# is inefficient and may be incorrect (we don't require that direct function
# registration be idempotent). You still need a DirectRemoteFunc object per
# function, but the heavy state is memoized here.
_importedFuncPackages = {}

class DirectRemoteFunc(RemoteFunc):
    """Invokes the function directly in the callers process. This will import
    the function's package.
    
    The package must implement a function named "LibffInvokeRegister" that
    returns a mapping of function name -> function object. This is similar to
    the 'funcs' argument to RemoteProcessServer."""

    def __init__(self, packagePath, funcName, arrayMnt):
        self.fName = funcName
        if packagePath in _importedFuncPackages:
            self.funcs = _importedFuncPackages[packagePath]
        else:
            spec = importlib.util.spec_from_file_location(packagePath.stem, packagePath)
            package = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(package)
            self.funcs = package.LibffInvokeRegister(arrayMnt)
            _importedFuncPackages[packagePath] = self.funcs 


    def Invoke(self, arg):
        return self.funcs[self.fName](arg)


    def Close(self):
        # No stats for direct invocation yet
        return {}


_runningFuncProcesses = {}
class ProcessRemoteFunc(RemoteFunc):
    # XXX probably should replace arrayMnt with a more generic handle to the
    # distribued array subsystem. Basically make the whole array system more
    # OOP instead of relying on global state (like logging or viper)
    def __init__(self, packagePath, funcName, arrayMnt):
        if packagePath in _runningFuncProcesses:
            self.proc = _runningFuncProcesses[packagePath]
        else:
            self.proc = sp.Popen(["python3", str(packagePath), '-m', arrayMnt], stdin=sp.PIPE, stdout=sp.PIPE, text=True)

        self.fname = funcName
        self.packagePath = packagePath
        self.arrayMnt = arrayMnt

    def Invoke(self, arg):
        req = { "command" : "invoke",
                "fName" : self.fname,
                "fArg" : arg }

        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        rawResp = self.proc.stdout.readline()
        resp = json.loads(rawResp)
        if resp['error'] is not None:
            raise InvocationError(resp['error'])

        return resp['resp']

    def Close(self):
        req = { "command" : "reportStats" }
        resp = self.Invoke(req)

        del _runningFuncProcesses[self.packagePath]
        self.proc.stdin.close()
        self.proc.wait()

        return { name : prof(fromDict=profile) for name, profile in resp['times'].items() }


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
    args = parser.parse_args(serverArgs)

    array.SetFileMount(args.mount)

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
                    resp = funcs[req['fName']](req['fArg'])
                else:
                    _remoteServerRespond({"error" : "Unrecognized function: " + req['fName']})

                _remoteServerRespond({"error" : None, "resp" : resp})

            elif req['command'] == 'reportStats':
                _remoteServerRespond({"error" : None, "stats" : {}})

            else:
                _remoteServerRespond({"error" : "Unrecognized command: " + req['command']})
        except Exception as e:
            _remoteServerRespond({"error" : "Unhandled internal error: " + repr(e)})
