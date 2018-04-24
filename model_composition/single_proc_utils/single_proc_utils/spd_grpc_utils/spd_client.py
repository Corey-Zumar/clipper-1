import sys
import os
import time
import logging
import grpc

from Queue import Queue
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock
from datetime import datetime

from spd_grpc_consts import GRPC_OPTIONS 

import spd_frontend_pb2
import spd_frontend_pb2_grpc

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

REQUEST_TIME_OUT_SECONDS = 30

class ReplicaAddress:

    def __init__(self, host_name, port):
        self.host_name = host_name
        self.port = port

    def get_channel(self):
        address = "{}:{}".format(self.host_name, self.port)
        return grpc.insecure_channel(address, options=GRPC_OPTIONS)

    def __str__(self):
        return "{}:{}".format(self.host_name, self.port)

    def __repr__(self):
        return "{}:{}".format(self.host_name, self.port)

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
            self.replicas[i] = (address, self._create_client(address))
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

    def _create_client(self, address):
        """
        Parameters
        -------------
        address : ReplicaAddress
        """

        logger.info("Creating client with address: {}".format(address))
        return spd_frontend_pb2_grpc.PredictStub(address.get_channel())

    def _run(self, replica_num):
        try:
            callback_threadpool = ThreadPoolExecutor(max_workers=1)
            _, client = self.replicas[replica_num]
            while self.active:
                inputs, msg_ids, callback = self.request_queue.get(block=True)
        
                grpc_inputs = [inp.tobytes() for inp in inputs]
                # grpc_inputs = [spd_frontend_pb2.FloatsInput(input=inp) for inp in inputs]
                predict_request = spd_frontend_pb2.PredictRequest(inputs=grpc_inputs, msg_ids=msg_ids)

                before = datetime.now()
                predict_response = client.PredictFloats(predict_request, REQUEST_TIME_OUT_SECONDS)
                end = datetime.now()
                print((end - before).total_seconds())
                callback_threadpool.submit(callback, replica_num, list(predict_response.msg_ids))
        except Exception as e:
            print(e)

    def __str__(self):
        return ",".join(self.replica_addrs)

    def __repr__(self):
        return ",".join(self.replica_addrs)

