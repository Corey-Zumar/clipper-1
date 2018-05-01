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

    def __init__(self, model_name, replica_addrs, outbound_handle, inbound_handle):
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
        while self.active:
            data_available = self.outbound_handle.poll(REQUEST_PIPE_POLLING_TIMEOUT_SECONDS)
            if not data_available:
                continue
            msg_id = self.outbound_handle.recv()
            self.request_queue.put(msg_id)
            while self.outbound_handle.poll(0):
                msg_id = self.outbound_handle.recv()
                self.request_queue.put(msg_id)

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
        while self.active:
            outbound_msg_id = self.request_queue.get(block=True)
            item_idx = np.random.randint(len(self.inputs))
            input_item = self.inputs[item_idx]
            
            begin = datetime.now()
            response = self.clients[replica_num].Predict(input_item, REQUEST_TIME_OUT_SECONDS)
            end = datetime.now()

            pred_lats.append((end - begin).total_seconds())

            self.response_queue.put(outbound_msg_id)
            num_enqueued += 1

            if num_enqueued >= 500 and self.model_name == KERNEL_SVM_MODEL_NAME:
                trial_end_time = datetime.now()
                thru = num_enqueued / (trial_end_time - trial_start_time).total_seconds()
                print("Response queue ingest: {} qps".format(thru))
                print(np.mean(pred_lats))
                pred_lats = []
                num_enqueued = 0
                trial_start_time = trial_end_time

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

