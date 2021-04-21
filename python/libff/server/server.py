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


class localExecutor():
    def __init__(self, eCfg):
        self.cfg = eCfg

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
        self.ready = False

        # Messages waiting for the executor to be ready
        self.pending = []

    def __str__(self):
        return self.cfg.eId.decode('utf-8')

# class execPool():
#     def __init__(self):
#         # Authoritative map of all exectors, for now we only support one
#         # executor per config.
#         # { executorConfig -> localExecutor }
#         funcs = {}
#
#         # LRU for GPU-enabled executors, we allow infinite non-GPU executors
#         # for now
#         gpuLRU = collections.deque() 
#
#     
#     def getExecutor(self, eCfg):
#         toKill = None
#         if eCfg in funcs:
#             f = funcs[eCfg]
#         else:
#             if eCfg.gpu && len(cudaFreeDevs) == 0:
#                 toKill = gpuLRU.pop()
#                 toKill.kill()
#
#             f = localExecutor(eCfg)
#             funcs[eCfg] = f
#
#         if eCfg.gpu:
#             gpuLRU.remove(f)
#             gpuLRU.push(f)
#
#         return (f, toKill)


# This is based on git@github.com:booksbyus/zguide.git examples/Python/lbbroker3.py
# Also: https://zguide.zeromq.org/docs/chapter3/#A-Load-Balancing-Message-Broker
class gatewayLoop(object):
    """Main event loop for the libff gateway server. For the moment, it does
    not create replicas for the same function but will create multiple exectors
    for different functions."""

    def __init__(self, clientSock, executorSock):
        self.funcs = {}

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

        func = self.funcs[funcId]
        
        if rawResp == b'READY':
            logging.info("Executor {} ready".format(str(func.cfg)))
            func.ready = True
            if len(func.pending) != 0:
                # Some requests might have arrived before the executor was
                # ready, we can send them now
                for req in func.pending:
                    self.executorStream.send_multipart([func.cfg.eId, b'', req['clientAddr'], b'', pickle.dumps(req)])
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

        if eCfg.eId not in self.funcs:
            self.funcs[eCfg.eId] = localExecutor(eCfg)
        func = self.funcs[eCfg.eId]

        if not func.ready:
            logging.debug("Executor {} not ready, pushing request to pending queue".format(str(func)))
            func.pending.append(req)
        else:
            self.executorStream.send_multipart([func.cfg.eId, b'', req['clientAddr'], b'', pickle.dumps(req)])


    def shutdown(self):
        logging.info("Shutting down!")
        self.executorStream.stop_on_recv()
        self.clientStream.stop_on_recv()

        # Keyboard interrupts get propagated to children so local workers will
        # shutdown via the signal, but sending sigint explicitly or remote
        # workers will shutdown via this message.
        for func in self.funcs.values():
            self.executorStream.send_multipart([func.cfg.eId, b'', b'none', b'', b'SHUTDOWN'])

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
