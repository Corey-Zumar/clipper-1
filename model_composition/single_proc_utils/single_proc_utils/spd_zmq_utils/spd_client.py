import sys
import os
import zmq
import time
import logging
import struct
import numpy as np
import Queue

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, RLock
from datetime import datetime
from datetime import timedelta

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

        logger.info("Generating inputs")
        self.inputs = generate_inputs() 

        self.active = False
        self.replica_addrs = replica_addrs
       
        self.total_dequeued = 0
        self.dequeue_rate = 0
        self.dequeue_trial_begin = datetime.now()
        
        self.total_enqueued = 0
        self.enqueue_rate = 0
        self.enqueue_trial_begin = datetime.now()

        self.threads = []
        self.replicas = {}
        for i in range(len(replica_addrs)):
            address = replica_addrs[i]
            self.replicas[i] = address
            self.threads.append(Thread(target=self._run, args=(i,)))

    def start(self,
              fixed_batch_size,
              batch_size, 
              slo_millis, 
              stats_callback, 
              expiration_callback,
              inflight_msgs,
              inflight_msgs_lock,
              arrival_process_millis):

        self.batch_size = batch_size
        self.slo_millis = slo_millis
        self.stats_callback = stats_callback
        self.expiration_callback = expiration_callback
        self.inflight_msgs = inflight_msgs
        self.inflight_msgs_lock = inflight_msgs_lock

        if fixed_batch_size:
            # Send 100000 queries at a rate of 1000 qps
            self.arrival_process_seconds = [.001 for _ in range(100000)]
        else:
            self.arrival_process_seconds = [item * .001 for item in arrival_process_millis] 
        self.active = True

        self.request_queue = deque()
        self.last_dequeued_time = None
        self.queue_lock = RLock()
        self.process_idx = 0

        for thread in self.threads:
            thread.start()

    def stop(self):
        self.active = False
        for thread in self.threads:
            thread.join()

    def update_enqueue_rate(self, num_enqueued):
        # Assumes that "queue_lock" is held
        curr_time = datetime.now()
        self.total_enqueued += num_enqueued
        enqueue_trial_length = (curr_time - self.enqueue_trial_begin).total_seconds()
        if enqueue_trial_length > QUEUE_RATE_MEASUREMENT_WINDOW_SECONDS:
            self.enqueue_rate = float(self.total_enqueued) / enqueue_trial_length
            self.enqueue_trial_begin = curr_time
            self.total_enqueued = 0

    def get_enqueue_rate(self):
        self.queue_lock.acquire()
        enqueue_rate = float(self.enqueue_rate)
        self.queue_lock.release()
        return enqueue_rate

    def update_dequeue_rate(self, num_dequeued):
        # Assumes that "queue_lock" is held
        curr_time = datetime.now()
        window_elapsed = False
        self.total_dequeued += num_dequeued
        dequeue_trial_length = (curr_time - self.dequeue_trial_begin).total_seconds()
        if dequeue_trial_length > QUEUE_RATE_MEASUREMENT_WINDOW_SECONDS:
            self.dequeue_rate = float(self.total_dequeued) / dequeue_trial_length
            self.dequeue_trial_begin = curr_time
            self.total_dequeued = 0
            window_elapsed = True

        return window_elapsed 

    def get_dequeue_rate(self):
        self.queue_lock.acquire()
        dequeue_rate = float(self.dequeue_rate)
        self.queue_lock.release()
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
                self.queue_lock.acquire()
                self.inflight_msgs_lock.acquire()

                if self.last_dequeued_time:
                    curr_time = datetime.now()
                    time_since_dequeue = (curr_time - self.last_dequeued_time).total_seconds()
                    accounted_delay = 0
                    while self.process_idx < len(self.arrival_process_seconds):
                        request_delay_seconds = self.arrival_process_seconds[self.process_idx]
                        if accounted_delay + request_delay_seconds < time_since_dequeue:
                            accounted_delay += request_delay_seconds
                            send_time = self.last_dequeued_time + timedelta(seconds=accounted_delay)
                            
                            input_idx = np.random.randint(0, len(self.inputs))
                            new_input = self.inputs[input_idx]
                            msg_id = self.process_idx

                            self.inflight_msgs[msg_id] = send_time

                            self.request_queue.append((new_input, msg_id, send_time))
                            self.process_idx += 1
                            self.update_enqueue_rate(1)
                        else:
                            time_cut = time_since_dequeue - accounted_delay
                            self.arrival_process_seconds[self.process_idx] -= time_cut
                            break

                else:
                    assert self.process_idx == 0
                    time.sleep(self.arrival_process_seconds[self.process_idx])
                    send_time = datetime.now()
                    
                    input_idx = np.random.randint(0, len(self.inputs))
                    new_input = self.inputs[input_idx]
                    msg_id = self.process_idx
                    self.inflight_msgs[msg_id] = send_time

                    self.request_queue.append((new_input, msg_id, send_time))
                    self.process_idx += 1
                    self.update_enqueue_rate(1)

                inputs = []
                msg_ids = []
                expiration_ids = []

                dequeue_time = datetime.now()
                while len(inputs) < self.batch_size and len(self.request_queue) > 0:
                    inp_item, msg_id, send_time = self.request_queue.popleft()
                    if (dequeue_time - send_time).total_seconds() * 1000 > self.slo_millis:
                        expiration_ids.append(msg_id)
                    else:
                        msg_ids.append(msg_id)
                        inputs.append(inp_item)
                        # Only count a query as "dequeued"
                        # if it has not expired
                        self.update_dequeue_rate(1)

                self.last_dequeued_time = datetime.now()
                
                self.inflight_msgs_lock.release()
                self.queue_lock.release()

                expiration_threadpool.submit(self.expiration_callback, expiration_ids)

                if len(msg_ids) == 0:
                    # We filtered out all incoming requests because they expired. We should
                    # check the queue again
                    continue

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

                callback_threadpool.submit(self.stats_callback, replica_num, parsed_output_msg_ids)
        except Exception as e:
            print("ERROR IN CLIENT: {}".format(e))
            os._exit(1)

    def __str__(self):
        return ",".join(self.replica_addrs)

    def __repr__(self):
        return ",".join(self.replica_addrs)

def generate_inputs():
    inception_inputs = [_get_inception_input() for _ in range(1000)]
    inception_inputs = [i for _ in range(40) for i in inception_inputs]

    return np.array(inception_inputs, dtype=np.float32)

def _get_inception_input():
    inception_input = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
    return inception_input.flatten()
