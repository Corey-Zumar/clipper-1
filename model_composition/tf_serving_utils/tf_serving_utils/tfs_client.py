import sys
import os
import time
import logging
import tensorflow as tf
import numpy as np
import Queue

from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from tf_serving_utils import tfs_utils

from .config_utils.config_creator import CONFIG_KEY_INCEPTION, CONFIG_KEY_RESNET, CONFIG_KEY_LOG_REG, CONFIG_KEY_KSVM

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

REQUEST_PIPE_POLLING_TIMEOUT_SECONDS = 1.0
REQUEST_TIME_OUT_SECONDS = 30

INCEPTION_FEATS_MODEL_NAME = CONFIG_KEY_INCEPTION
RESNET_152_MODEL_NAME = CONFIG_KEY_RESNET
LOG_REG_MODEL_NAME = CONFIG_KEY_LOG_REG
KERNEL_SVM_MODEL_NAME = CONFIG_KEY_KSVM

class ReplicaAddress:

    def __init__(self, host_name, port):
        self.host_name = host_name
        self.port = port

    def get_channel(self):
        return implementations.insecure_channel(self.host_name, int(self.port))

    def __str__(self):
        return "{}:{}".format(self.host_name, self.port)

    def __repr__(self):
        return "{}:{}".format(self.host_name, self.port)

class GRPCClient:

    def __init__(self, model_name, replica_addrs, outbound_handle, inbound_handle, queue_size_val):
        """
        Parameters
        -------------
        replica_addrs : [ReplicaAddress]
            A list of addresses corresponding to replicas that
            the client should communicate with
        """
        self.model_name = model_name
        self.replica_addrs = replica_addrs
        self.clients = [self._create_client(address) for address in replica_addrs]
        self.outbound_handle = outbound_handle 
        self.inbound_handle = inbound_handle
        self.queue_size_val = queue_size_val

        self.request_queue = Queue.Queue()
        self.response_queue = Queue.Queue()

        logger.info("Client generating random inputs...")
        input_gen_fn = self._get_input_gen_fn()
        base_inputs = [input_gen_fn() for _ in range(100)]
        self.inputs = [i for _ in range(10) for i in base_inputs]
        self.inputs = [tfs_utils.create_predict_request(self.model_name, inp_item) for inp_item in self.inputs]

        self.active = False

        self.worker_threads = [Thread(target=self._run, args=(i,)) for i in range(len(self.clients))]
        self.request_thread = Thread(target=self._manage_requests, args=())
        self.response_thread = Thread(target=self._manage_responses, args=())

    def start(self):
        self.active = True
        self.start_time = datetime.now()
        for thread in self.worker_threads:
            thread.start()

        self.request_thread.start()
        self.response_thread.start()

    def get_queue_size(self):
        return self.request_queue.qsize()

    def _create_client(self, address):
        """
        Parameters
        -------------
        address : ReplicaAddress
        """

        return prediction_service_pb2.beta_create_PredictionService_stub(address.get_channel())

    def _manage_requests(self):
        thread_enqueue_latencies = []
        while self.active:
            data_available = self.outbound_handle.poll(REQUEST_PIPE_POLLING_TIMEOUT_SECONDS)
            if not data_available:
                continue
            outbound_msgs = [self.outbound_handle.recv()]
            while self.outbound_handle.poll(0):
                req_item = self.outbound_handle.recv()
                outbound_msgs.append(req_item)

            with self.queue_size_val.get_lock():
                self.queue_size_val.value += len(outbound_msgs)

            curr_time = datetime.now()
            for msg_id, send_time in outbound_msgs:
                self.request_queue.put((msg_id, send_time))
                thread_enqueue_latencies.append((curr_time - send_time).total_seconds())

    def _manage_responses(self):
        while self.active:
            inbound_msg_ids = [self.response_queue.get(block=True)]
            while self.response_queue.qsize() > 0:
                try:
                    msg_id = self.response_queue.get(block=False)
                    inbound_msg_ids.append(msg_id)
                except Queue.Empty:
                    break

            for msg_id in inbound_msg_ids:
                self.inbound_handle.send(msg_id)
        
    def _run(self, replica_num):
        num_enqueued = 0
        trial_start_time = datetime.now()
        pred_lats = []
        queue_sizes = []
        queueing_lats = []
        while self.active:
            outbound_msg_id, send_time = self.request_queue.get(block=True)

            with self.queue_size_val.get_lock():
                self.queue_size_val.value -= 1

            queueing_lats.append((datetime.now() - send_time).total_seconds())

            queue_sizes.append(self.request_queue.qsize())
            item_idx = np.random.randint(len(self.inputs))
            input_item = self.inputs[item_idx]
            
            begin = datetime.now()
            response = self.clients[replica_num].Predict(input_item, REQUEST_TIME_OUT_SECONDS)
            end = datetime.now()

            pred_lat = (end - begin).total_seconds()
            pred_lats.append(pred_lat)

            # self.response_queue.put((outbound_msg_id, end))
            self.response_queue.put((outbound_msg_id, pred_lat))
            # self.response_queue.put(outbound_msg_id)

            if len(queueing_lats) > 500:
                p99 = np.percentile(queueing_lats, 99)
                mean = np.mean(queueing_lats)
                queueing_lats = []

                if self.model_name == RESNET_152_MODEL_NAME:
                    print("Replica queueing delay -  P99: {}, Mean: {}".format(p99, mean))

            # if len(queue_sizes) > 200:
            #     mean_queue_size = np.mean(queue_sizes)
            #     # print(self.model_name, self.replica_addrs[0], mean_queue_size)
            #     queue_sizes = []

            num_enqueued += 1

            # if num_enqueued >= 200 and self.model_name == INCEPTION_FEATS_MODEL_NAME:
            #     trial_end_time = datetime.now()
            #     print(np.mean(pred_lats), np.percentile(pred_lats, 99))
            #     pred_lats = []
            #     num_enqueued = 0
            #     trial_start_time = trial_end_time

    def _get_input_gen_fn(self):
        if self.model_name == INCEPTION_FEATS_MODEL_NAME:
            return self._get_inception_input
        elif self.model_name == RESNET_152_MODEL_NAME:
            return self._get_resnet_input
        elif self.model_name == KERNEL_SVM_MODEL_NAME:
            return self._get_ksvm_input
        elif self.model_name == LOG_REG_MODEL_NAME:
            return self._get_log_reg_input
        else:
            raise

    def _get_ksvm_input(self):
        ksvm_input = np.array(np.random.rand(2048), dtype=np.float32)
        return ksvm_input

    def _get_log_reg_input(self):
        log_reg_input = np.array(np.random.rand(2048), dtype=np.float32)
        return log_reg_input 

    def _get_resnet_input(self):
        resnet_input = np.array(np.random.rand(224, 224, 3) * 255, dtype=np.float32)
        return resnet_input

    def _get_inception_input(self):
        inception_input = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
        return inception_input

    def __str__(self):
        return ",".join(self.replica_addrs)

    def __repr__(self):
        return ",".join(self.replica_addrs)

