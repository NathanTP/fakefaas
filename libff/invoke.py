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


_importedFuncPackages = {}
class DirectRemoteFunc(RemoteFunc):
    """Invokes the function directly in the callers process. This will import
    the function's package."""

    def __init__(self, packagePath, funcName, arrayMnt):
        if packagePath in _importedFuncPackages:
            package = __importedFuncPackages[packagePath]
        else:
            spec = importlib.util.spec_from_file_location(packagePath.stem, packagePath)
            package = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(package)
            _importedFuncPackages[packagePath] = package

        self.func = getattr(package, funcName)


    def Invoke(self, arg):
        return self.func(arg)


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
        req = { "func" : "reportStats" }
        resp = self.Invoke(req)

        del _runningFuncProcesses[self.packagePath]
        self.proc.stdin.close()
        self.proc.wait()

        return { name : prof(fromDict=profile) for name, profile in resp['times'].items() }

    def SetArrayMnt(self, mnt):
        self.arrayMnt = mnt

def __remoteServerRespond(msg):
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
            __remoteServerRespond({ "error" : err })
            continue


        try:
            if req['command'] == 'invoke':
                if req['fName'] in funcs:
                    resp = funcs[req['fName']](req['fArg'])
                else:
                    __remoteServerRespond({"error" : "Unrecognized function: " + req['fName']})

                __remoteServerRespond({"error" : None, "resp" : resp})

            elif req['command'] == 'reportStats':
                __remoteServerRespond({"error" : None, "stats" : {}})

            else:
                __remoteServerRespond({"error" : "Unrecognized command: " + req['command']})
        except Exception as e:
            __remoteServerRespond({"error" : "Unhandled internal error: " + repr(e)})
