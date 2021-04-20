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

from tornado.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream

urlClient = "ipc://libffGateway_client.ipc"
urlExecutor = "ipc://libffGateway_executor.ipc"


def initLogging():
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler("libffGateway.log")
    fileHandler.setLevel(logging.DEBUG)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    rootLogger.addHandler(consoleHandler)


try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    cudaAvailable = True
    cudaNDev = cuda.Device.count()
    cudaFreeDevs = [ i for i in range(cudaNDev) ]
except ImportError:
    cudaAvailable = False


class localExecutor():
    def __init__(self, packagePath, enableGpu=False):
        self.packagePath = packagePath
        self.enableGpu = enableGpu

        # Unique human-readable name of this executor, will be used for the zmq workerID.
        # self.eID = "_".join([self.packagePath.stem, str(random.randint(0,100000))])
        self.eID = self.packagePath.stem

        if cudaAvailable and enableGpu:
            if len(cudaFreeDevs) == 0:
                logging.error("No more GPUs available!")
                raise RuntimeError("No more GPUs available!")

            self.cudaDev = cudaFreeDevs.pop()
        else:
            self.cudaDev = None

        # if it's a module name, we still treat it like a path since everything
        # will still work.
        self.packagePath = pathlib.Path(self.packagePath)
        if self.packagePath.is_file():
            cmd = ["python3", str(self.packagePath.name)]
        else:
            # Could be a real path or it could be an installed package
            cmd = ["python3", "-m", str(self.packagePath.name)]

        cmd += ['-i', self.eID]
        cmd += ['-u', urlExecutor]

        if self.cudaDev is not None:
            cmd += ['-g', str(self.cudaDev)]

        logging.info("Launching executor: " + " ".join(cmd))
        self.proc = sp.Popen(cmd, cwd=self.packagePath.parent)
        self.ready = False

        # Messages waiting for the executor to be ready
        self.pending = []


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
        funcID = msg[0]
        clientAddr = msg[2]
        rawResp = msg[4]

        func = self.funcs[funcID]
        
        if rawResp == b'READY':
            logging.info("Executor {} ready".format(func.eID))
            func.ready = True
            if len(func.pending) != 0:
                # Some requests might have arrived before the executor was
                # ready, we can send them now
                for req in func.pending:
                    self.executorStream.send_multipart([func.eID.encode('utf-8'), b'', req['clientAddr'], b'', pickle.dumps(req)])
        else:
            logging.info("Got response from executor {} for client {}".format(func.eID, clientAddr))
            # Work done, send back to client
            empty, reply = msg[3:]
            self.clientStream.send_multipart([clientAddr, b'', rawResp])


    def handleClients(self, msg):
        """Handle requests from clients. Requests look like:
            {
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
        eId = req['fPath'].stem.encode('utf-8')

        logging.info("Received request from {} for {}:{}".format(req['clientAddr'], req['fPath'], req['fName']))

        if eId not in self.funcs:
            self.funcs[eId] = localExecutor(req['fPath'], req['enableGpu'])
        func = self.funcs[eId]

        if not func.ready:
            logging.debug("Executor {} not ready, pushing request to pending queue".format(func.eID))
            func.pending.append(req)
        else:
            self.executorStream.send_multipart([func.eID.encode('utf-8'), b'', pickle.dumps(req)])


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
    queue = gatewayLoop(clientSock, executorSock)
    try:
        IOLoop.instance().start()
    except KeyboardInterrupt:
        print("Shutting down!")
        IOLoop.instance().stop()
