import sys
import os
import time
import logging

from grpc.beta import implementations

from Queue import Queue
from threading import Thread, Lock
from datetime import datetime

import spd_frontend_pb2
import spd_frontend_pb2_grpc

REQUEST_TIME_OUT_SECONDS = 30

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

        self.threads = []
        self.replicas = {}
        for i in range(len(replica_addrs)):
            address = replica_addrs[i]
            queue = Queue()
            self.replicas[i] = (address, self._create_client(address), Queue())
            self.threads.append(Thread(target=self._run, args=(i,)))

    def predict(self, replica_num, inputs, msg_ids, callback):
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
        
        _, _, queue = self.replicas[replica_num]
        queue.put(inputs, msg_ids, callback)

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

        return spd_frontend_pb2_grpc.PredictStub(address.get_channel())

    def _run(self, replica_num):
        _, client, request_queue = self.replicas[replica_num]
        while self.active:
            inputs, msg_ids, callback = self.request_queue.get(block=True)
           
            grpc_inputs = [spd_frontend_pb2.FloatsInput(input=inp) for inp in inputs]
            predict_request = spd_frontend_pb2.PredictRequest(inputs=grpc_inputs, msg_ids=msg_ids)

            response = client.PredictFloats(input_item, REQUEST_TIME_OUT_SECONDS)
            callback(replica_num, list(response.msg_ids))

    def __str__(self):
        return ",".join(self.replica_addrs)

    def __repr__(self):
        return ",".join(self.replica_addrs)

