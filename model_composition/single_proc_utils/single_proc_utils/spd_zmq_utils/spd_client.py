import sys
import os
import zmq
import time
import logging
import struct
import numpy as np

from Queue import Queue
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock
from datetime import datetime

from spd_server import INPUT_DTYPE

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

REQUEST_TIME_OUT_SECONDS = 30
HOST_NAME_LOCALHOST_READABLE = "localhost"
HOST_NAME_LOCALHOST_NUMERIC = "127.0.0.1"

UINT32_SIZE_BYTES = 4
INITIAL_INPUT_HEADER_BUFFER_SIZE = 200 * UINT32_SIZE_BYTES 

class ReplicaAddress:

    def __init__(self, host_name, port):
        if host_name == HOST_NAME_LOCALHOST_READABLE:
            host_name = HOST_NAME_LOCALHOST_NUMERIC
        self.host_name = host_name
        self.port = port

    def __str__(self):
        return "tcp://{}:{}".format(self.host_name, self.port)

    def __repr__(self):
        return "REPLICA ADDRESS OBJ - tcp://{}:{}".format(self.host_name, self.port)

class SPDClient:

    def __init__(self, replica_addrs):
        """
        Parameters
        -------------
        replica_addrs : [ReplicaAddress]
            A list of addresses corresponding to replicas that
            the client should communicate with
        """

        self.active = False
        self.replica_addrs = replica_addrs
        
        self.request_queue = Queue()
        self.threads = []
        self.replicas = {}
        for i in range(len(replica_addrs)):
            address = replica_addrs[i]
            self.replicas[i] = address
            self.threads.append(Thread(target=self._run, args=(i,)))

    def predict(self, inputs, msg_ids, callback):
        """ 
        Parameters
        -------------
        input_item : Proto
            An input proto
            TODO(czumar): Make this more specific

        callback : function
            The function to execute when a response
            is received
        """
        self.request_queue.put((inputs, msg_ids, callback))

    def start(self):
        self.active = True
        for thread in self.threads:
            thread.start()

    def stop(self):
        self.active = False
        for thread in self.threads:
            thread.join()

    def _run(self, replica_num):
        try:
            callback_threadpool = ThreadPoolExecutor(max_workers=1)
            address = str(self.replicas[replica_num])
            context = zmq.Context()
            socket = context.socket(zmq.DEALER) 
            socket.connect(address)
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            logger.info("Client connected to model server at address: {}".format(address))

            input_header_buffer = bytearray(INITIAL_INPUT_HEADER_BUFFER_SIZE)
            while self.active:
                inputs, msg_ids, callback = self.request_queue.get(block=True)

                input_header_size = (1 + len(inputs)) * UINT32_SIZE_BYTES
                if len(input_header_buffer) < input_header_size:
                    input_header_buffer = bytearray(input_header_size * 2)

                buffer_idx = 0 
                struct.pack_into("<I", input_header_buffer, buffer_idx, len(inputs))
                buffer_idx += UINT32_SIZE_BYTES
                for inp in inputs:
                    struct.pack_into("<I", input_header_buffer, buffer_idx, len(inp) * INPUT_DTYPE.itemsize)
                    buffer_idx += UINT32_SIZE_BYTES

                input_header = memoryview(input_header_buffer)[:input_header_size]

                # input_header = np.array([len(inputs)] + [(len(inp) * INPUT_DTYPE.itemsize) for inp in inputs], dtype=np.uint32)
                
                # Send the empty delimeter required at the start of a new message

                socket.send("", zmq.SNDMORE)
                
                socket.send(memoryview(msg_ids.view(np.uint8)), zmq.SNDMORE)
                socket.send(input_header, zmq.SNDMORE)
                for i in range(len(inputs)):
                    if i < len(inputs) - 1:
                        socket.send(memoryview(inputs[i].view(np.uint8)), copy=False, flags=zmq.SNDMORE)
                    else:
                        socket.send(memoryview(inputs[i].view(np.uint8)), copy=False, flags=0)

                receivable_sockets = dict(poller.poll(timeout=None))
                if socket in receivable_sockets and receivable_sockets[socket] == zmq.POLLIN:
                    # Receive delimiter between routing identity and content
                    socket.recv()
                    output_msg_ids = socket.recv()
                    parsed_output_msg_ids = np.frombuffer(output_msg_ids, dtype=np.uint32)

                else:
                    logger.info("Undefined receive behavior")
                    raise

                callback_threadpool.submit(callback, replica_num, parsed_output_msg_ids)
        except Exception as e:
            print("ERROR")
            print(e)

    def __str__(self):
        return ",".join(self.replica_addrs)

    def __repr__(self):
        return ",".join(self.replica_addrs)

