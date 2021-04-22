import threading
import time
import zmq
import libff.invoke
import pathlib
import pickle
import random
import logging
import sys
import subprocess as sp
import collections
import signal
import enum

from tornado.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream

urlClient = "ipc://libffGateway_client.ipc"
urlExecutor = "ipc://libffGateway_executor.ipc"


def initLogging():
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("gateway: %(message)s")
    # fileHandler = logging.FileHandler("libffGateway.log")
    # fileHandler.setLevel(logging.DEBUG)
    # fileHandler.setFormatter(formatter)
    # rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(formatter)
    rootLogger.addHandler(consoleHandler)


try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    cudaAvailable = True
    cudaNDev = cuda.Device.count()
    cudaFreeDevs = [ i for i in range(cudaNDev) ]
except ImportError:
    cudaAvailable = False


class executorConfig():
    """Represents a configuration for an executor. The config describes what
    the executor can do rather than describing a particular executor. Executors
    with the same config are equivalent in functionality and can correctly
    accept the same requests."""

    def __init__(self, req):
        self.path = req['fPath']
        self.tId = req['tenantId']
        self.gpu = req['enableGpu']

        # Unique short name for this function, used to idenify zmq messages
        self.eId = (str(self.path.stem) + "_" + str(self.tId)).encode('utf-8')

    def __eq__(self, other):
        if not isinstance(other, executorConfig):
            return NotImplemented

        # Executors are expected to always or never need a GPU so we don't
        # bother comparing 
        return ((self.path == other.path) and (self.tId == other.tId)) 

    def __hash__(self):
        return (hash((self.eId)))

    def __str__(self):
        return self.eId.decode('utf-8')


class eState(enum.Enum):
    INIT    = enum.auto() # Launched but not yet ready
    READY   = enum.auto() # Accepting requests
    DYING   = enum.auto() # No longer accepting requests but still processing any backlog 
    DEAD    = enum.auto() # Death acknowledged, all resources freed


class localExecutor():
    def __init__(self, eCfg, socket):
        self.cfg = eCfg
        self.socket = socket
        self.state = eState.INIT

        if cudaAvailable and self.cfg.gpu:
            if len(cudaFreeDevs) == 0:
                logging.error("No more GPUs available!")
                raise RuntimeError("No more GPUs available!")

            self.cudaDev = cudaFreeDevs.pop()
        else:
            self.cudaDev = None

        # if it's a module name, we still treat it like a path since everything
        # will still work.
        self.packagePath = pathlib.Path(self.cfg.path)
        if self.cfg.path.is_file():
            cmd = ["python3", str(self.cfg.path.name)]
        else:
            # Could be a real path or it could be an installed package
            cmd = ["python3", "-m", str(self.cfg.path.name)]

        cmd += ['-i', self.cfg.eId.decode('utf-8')]
        cmd += ['-u', urlExecutor]

        if self.cudaDev is not None:
            cmd += ['-g', str(self.cudaDev)]

        logging.info("Launching executor: " + " ".join(cmd))
        self.proc = sp.Popen(cmd, cwd=self.packagePath.parent)

        # Messages waiting for the executor to be ready
        self.pending = []

    def __str__(self):
        return self.cfg.eId.decode('utf-8')

    def shutdown(self):
        """Politely shutdown the executor. It will handle any work already sent
        to it and then terminate. To impolitely kill, access self.proc directly."""
        self.state = eState.DYING
        self.socket.send_multipart([self.cfg.eId, b'', b'none', b'', b'SHUTDOWN'])


class execPool():
    def __init__(self):
        # Authoritative map of all exectors, for now we only support one
        # executor per config.
        # { executorConfig.eId -> localExecutor }
        self.funcs = {}

        # LRU for GPU-enabled executors, we allow infinite non-GPU executors
        # for now
        self.gpuLRU = collections.deque() 


    def killAll(self):
        """Shutdown all executors, ignoring any pending work."""
        for f in self.funcs.values():
            f.proc.send_signal(signal.SIGINT)

        for func in self.funcs.values():
            try:
                func.proc.wait(timeout=0.5)
            except sp.TimeoutExpired:
                logging.warning("Executor took too long to terminate, killing")
                func.proc.kill()


    def getById(self, eId):
        """Get an existing executor by id"""
        if eId in self.funcs:
            return self.funcs[eId]
        else:
            raise RuntimeError("Requested non-existent function: ", eId.decode("utf-8"))


    def getOrCreate(self, eCfg, socket):
        """Get a new or re-used executor."""
        global cudaFreeDevs

        if eCfg.eId in self.funcs:
            f = self.funcs[eCfg.eId]
            # Reset GPU LRU
            if eCfg.gpu:
                self.gpuLRU.remove(f)
                self.gpuLRU.appendleft(f)

        elif eCfg.gpu and len(cudaFreeDevs) == 0:
            # We don't have an executor for eCfg but we're out of GPUs to
            # create a new one. Kill something to make room.
            f = self.gpuLRU[-1]
            if f.state == eState.DEAD:
                self.gpuLRU.pop()
                del self.funcs[f.cfg.eId]
                cudaFreeDevs.append(f.cudaDev)

                # We are guaranteed to have a free GPU now
                if len(self.gpuLRU) == 0:
                    f = localExecutor(eCfg, socket)
                    self.funcs[eCfg.eId] = f
                    self.gpuLRU.appendleft(f)
                else:
                    f = self.gpuLRU.pop()
                    self.gpuLRU.appendleft(f)

            elif f.state != eState.DYING:
                # We don't have enough resources to create a new GPU executor
                # yet, return a dying executor. The caller cannot give new work
                # to the executor and must wait until it dies before replaying
                # any messages. 
                f.shutdown()

        else:
            f = localExecutor(eCfg, socket)
            self.funcs[eCfg.eId] = f

            if eCfg.gpu:
                self.gpuLRU.appendleft(f)

        return f


# This is based on git@github.com:booksbyus/zguide.git examples/Python/lbbroker3.py
# Also: https://zguide.zeromq.org/docs/chapter3/#A-Load-Balancing-Message-Broker
class gatewayLoop(object):
    """Main event loop for the libff gateway server. For the moment, it does
    not create replicas for the same function but will create multiple exectors
    for different functions."""

    def __init__(self, clientSock, executorSock):
        self.pool = execPool()

        self.loop = IOLoop.instance()

        self.clientStream = ZMQStream(clientSock)
        self.clientStream.on_recv(self.handleClients)

        self.executorStream = ZMQStream(executorSock)
        self.executorStream.on_recv(self.handleExecutors)


    def handleExecutors(self, msg):
        """Handle responses from executors"""
        # For zmq reasons, there are empty frames in between fields in the message
        funcId = msg[0]
        clientAddr = msg[2]
        rawResp = msg[4]

        func = self.pool.getById(funcId)
        
        if rawResp == b'READY':
            assert func.state == eState.INIT

            logging.info("Executor {} ready".format(str(func.cfg)))
            func.state = eState.READY

            # Replay any messages intended for this executor that were waiting
            # for it to be ready
            for msg in func.pending:
                logging.info("Replaying pending message (executor ready)")
                self.handleClients(msg)
            func.pending = []

        elif rawResp == b'DEAD':
            assert func.state == eState.DYING
            try:
                func.proc.wait(timeout=0.5)
            except sp.TimeoutExpired:
                logging.warning("Executor {} failed to shutdown cleanly, killing".format(str(func.cfg)))
                func.proc.kill()
            func.state = eState.DEAD

            # Replay any pending messages now that we have enough resources to
            # start a new executor. At least the first pending message is
            # guaranteed to get an executor (avoids starvation)
            for msg in func.pending:
                logging.info("Replaying pending message (executor died)")
                time.sleep(0.5)
                self.handleClients(msg)
            func.pending = []

        else:
            logging.info("Got response from executor {} for client 0x{}".format(str(func.cfg), clientAddr.hex()))
            # Work done, send back to client
            empty, reply = msg[3:]
            self.clientStream.send_multipart([clientAddr, b'', rawResp])



    def handleClients(self, msg):
        """Handle requests from clients. Requests look like:
            {
                'tenantId'  : ID of a protection domain (tenants are mutually untrusting)
                'enableGpu' : whether or not this func needs a gpu
                'fPath'     : Path to the function package
                'fName'     : Name of the function in the package
                'fArg'      : Argument to the function
                'stats'     : XXX Haven't worked out yet
            }
        """
        clientAddr, empty, rawReq = msg
        assert empty == b""

        req = pickle.loads(rawReq)
        req['clientAddr'] = clientAddr
        req['fPath'] = pathlib.Path(req['fPath'])
        eCfg = executorConfig(req)

        # eId = req['tenantId'] + "_" + req['fPath'].stem

        logging.info("Received request from 0x{} for {}:{}".format(req['clientAddr'].hex(), req['fPath'], req['fName']))

        func = self.pool.getOrCreate(eCfg, self.executorStream)

        if func.state == eState.INIT:
            logging.debug("Executor {} not ready, pushing request to pending queue".format(str(func)))
            func.pending.append(msg)
        elif func.state == eState.DYING:
            logging.debug("Executor {} dying, postponing request until a new executor is available".format(str(func)))
            func.pending.append(msg)
        elif func.state == eState.READY:
            self.executorStream.send_multipart([func.cfg.eId, b'', req['clientAddr'], b'', pickle.dumps(req)])
        else:
            raise RuntimeError("Unexpected function state: " + str(func.state))


    def shutdown(self):
        logging.info("Shutting down!")
        self.executorStream.stop_on_recv()
        self.clientStream.stop_on_recv()

        self.pool.killAll()

        self.executorStream.close()
        self.clientStream.close()

        IOLoop.instance().stop()


def start():
    """Runs the server, does not return"""

    initLogging()

    # Prepare our context and sockets
    context = zmq.Context()

    clientSock = context.socket(zmq.ROUTER)
    clientSock.bind(urlClient)

    executorSock = context.socket(zmq.ROUTER)
    executorSock.bind(urlExecutor)

    logging.info("Starting Event Loop")
    looper = gatewayLoop(clientSock, executorSock)

    signal.signal(signal.SIGINT, lambda s,f: IOLoop.instance().add_callback_from_signal(looper.shutdown))
    IOLoop.instance().start()
