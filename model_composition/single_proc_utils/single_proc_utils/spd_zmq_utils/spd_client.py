import sys
import os
import zmq
import time
import logging
import struct
import numpy as np
import Queue

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

QUEUE_RATE_MEASUREMENT_WINDOW_SECONDS = 15

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
        
        self.request_queue = Queue.Queue()
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
        send_time = datetime.now()
        self.request_queue.put((inputs, msg_ids, send_time, callback))

    def start(self, batch_size, slo_millis, expiration_callback):
        self.batch_size = batch_size
        self.slo_millis = slo_millis
        self.expiration_callback = expiration_callback
        self.active = True
        self.total_dequeued = 0
        self.dequeue_rate = None
        self.dequeue_trial_begin = datetime.now()
        self.dequeue_rate_lock = Lock() 
        for thread in self.threads:
            thread.start()

    def stop(self):
        self.active = False
        for thread in self.threads:
            thread.join()

    def update_dequeue_rate(self, num_dequeued):
        self.dequeue_rate_lock.acquire()
        curr_time = datetime.now()
        self.total_dequeued += num_dequeued
        dequeue_trial_length = (curr_time - self.dequeue_trial_begin).total_seconds()
        if dequeue_trial_length > QUEUE_RATE_MEASUREMENT_WINDOW_SECONDS:
            self.dequeue_rate = float(self.total_dequeued) / dequeue_trial_length
            self.dequeue_trial_begin = curr_time
            self.total_dequeued = 0
        self.dequeue_rate_lock.release()

    def get_dequeue_rate(self):
        self.dequeue_rate_lock.acquire()
        dequeue_rate = float(self.dequeue_rate)
        self.dequeue_rate_lock.release()
        return dequeue_rate

    def _run(self, replica_num):
        try:
            expiration_threadpool = ThreadPoolExecutor(max_workers=1)
            callback_threadpool = ThreadPoolExecutor(max_workers=1)
            address = str(self.replicas[replica_num])
            context = zmq.Context()
            socket = context.socket(zmq.DEALER) 
            socket.connect(address)
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            logger.info("Client connected to model server at address: {}".format(address))

            input_header_buffer = bytearray(INITIAL_INPUT_HEADER_BUFFER_SIZE)
            dequeue_trial_begin = datetime.now()
            while self.active:
                iter_dequeued = 0
                expiration_ids = []
                inputs, msg_ids, send_time, callback = self.request_queue.get(block=True)
                iter_dequeued += len(msg_ids)
                dequeue_time = datetime.now()
                if (dequeue_time - send_time).total_seconds() * 1000 > self.slo_millis:
                    inputs = []
                    msg_ids = []
                    expiration_ids = msg_ids

                while len(inputs) < self.batch_size and self.request_queue.qsize() > 0:
                    try:
                        more_inputs, more_msg_ids, send_time, _ = self.request_queue.get()
                        iter_dequeued += len(more_msg_ids)
                        dequeue_time = datetime.now()
                        if (dequeue_time - send_time).total_seconds() * 1000 > self.slo_millis:
                            expiration_ids += more_msg_ids 
                        else:
                            inputs += more_inputs
                            msg_ids += more_msg_ids
                    
                    except Queue.Empty:
                        break
                
                self.update_dequeue_rate(iter_dequeued) 

                if len(msg_ids) == 0:
                    # We filtered out all incoming requests because they expired. We should
                    # check the queue again
                    continue

                expiration_threadpool.submit(self.expiration_callback, expiration_ids)

                msg_ids = np.array(msg_ids, dtype=np.uint32)

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

    def _expire_queries(self, expired_msg_ids):
        pass

    def __str__(self):
        return ",".join(self.replica_addrs)

    def __repr__(self):
        return ",".join(self.replica_addrs)

