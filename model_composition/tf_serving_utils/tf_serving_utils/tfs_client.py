import sys
import os
import time
import logging
import tensorflow as tf

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from Queue import Queue
from threading import Thread, Lock

REQUEST_QUEUE_POLLING_DELAY_SECONDS = .005
REQUEST_TIME_OUT_SECONDS = 5

class ReplicaAddress:

    def __init__(self, host_name, port):
        self.host_name = host_name
        self.port = port

    def get_channel(self):
        return implementations.insecure_channel(self.host_name, int(self.port))

class GRPCClient:

    def __init__(self, replica_addrs):
        """
        Parameters
        -------------
        replica_addrs : [ReplicaAddress]
            A list of addresses corresponding to replicas that
            the client should communicate with
        """

        self.clients = [self._create_client(address) for address in replica_addrs]
        self.request_queue = Queue(maxsize=len(self.clients))

        # TODO(czumar): Make sure this synchronization var works
        self.active = False

        self.threads = [Thread(target=self._run, args=(i,)) for i in range(len(self.clients))]

    def predict(self, input_item, callback):
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

        self.request_queue.put((input_item, callback))

    def start(self):
        for thread in self.threads:
            thread.start()

    def stop(self):
        self.active = True
        for thread in self.threads:
            thread.join()

    def _create_client(self, address):
        """
        Parameters
        -------------
        address : ReplicaAddress
        """

        return prediction_service_pb2.beta_create_PredictionService_stub(address.get_channel())

    def _run(self, replica_num):
        while self.active:
            if self.request_queue.empty():
                time.sleep(REQUEST_QUEUE_POLLING_DELAY_SECONDS)

            else:
                input_item, callback = self.request_queue.get(block=True)
                print("HERE!")
                response = self.clients[replica_num].predict(input_item, REQUEST_TIME_OUT_SECONDS)
                print("GOT RESPONSE!")
                print(response)
                callback(response)




