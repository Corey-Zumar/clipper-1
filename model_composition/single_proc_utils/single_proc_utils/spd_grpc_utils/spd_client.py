import sys
import os
import time
import logging
import grpc

# import spd_frontend_pb2
# import spd_frontend_pb2_grpc

from Queue import Queue
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock
from datetime import datetime
from flatbuffers import builder

from spd_grpc_consts import GRPC_OPTIONS, INCEPTION_IMAGE_SIZE
from flatbufs import spd_frontend_grpc_fb, PredictRequest, PredictResponse, FloatsInput


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
        return spd_frontend_grpc_fb.PredictStub(address.get_channel())

    def _run(self, replica_num):
        try:
            callback_threadpool = ThreadPoolExecutor(max_workers=1)
            _, client = self.replicas[replica_num]
            while self.active:
                inputs, msg_ids, callback = self.request_queue.get(block=True)

               
                grpc_inputs = [spd_frontend_pb2.FloatsInput(input=inp) for inp in inputs]
                predict_request = spd_frontend_pb2.PredictRequest(inputs=grpc_inputs, msg_ids=msg_ids)

                before = datetime.now()
                predict_response = client.PredictFloats(predict_request, REQUEST_TIME_OUT_SECONDS)
                end = datetime.now()
                print((end - before).total_seconds())
                callback_threadpool.submit(callback, replica_num, list(predict_response.msg_ids))
        except Exception as e:
            print(e)

    def _create_predict_request(inputs, msg_ids):
        batch_size = len(inputs)
        builder_size = (batch_size + 5) * INCEPTION_IMAGE_SIZE
        builder = flatbuffers.Builder(builder_size)
        floats_input_idxs = []
        for inp in inputs:
            inp_bytes = memoryview(inp.view(np.uint8))
            inp_bytes_len = len(inp_bytes)
            FloatsInput.FloatsInputStartDataVector(builder, inp_bytes_len) 
            builder.Bytes[builder.head : (builder.head + inp_bytes_len)] = inp_bytes
            data = builder.EndVector(inp_bytes_len)
            FloatsInput.FloatsInputStart(builder)
            FloatsInput.FloatsInputAddData(builder, data)
            floats_input_idx = FloatsInput.FloatsInputEnd(builder)
            floats_input_idxs.append(floats_input_idx)

        msg_ids_bytes = memoryview(msg_ids.view(np.uint8))
        msg_ids_len = len(msg_ids_bytes)
        PredictRequest.PredictRequestStartMsgIdsVector(builder, msg_ids_len)
        builder.Bytes[builder.head : (builder.head + msg_ids_len)] = msg_ids_bytes 
        msg_ids_idx = builder.EndVector(msg_ids_len)

        PredictRequest.PredictRequestStartInputsVector(builder, len(floats_input_idxs))
        for float_input_idx in floats_input_idx:
            curr_offset = builder.PrependUOffsetTRelative(float_input_idx)
        inputs_vector_idx = builder.EndVector(curr_offset)
        

        PredictRequest.PredictRequestStart(builder)
        PredictRequest.PredictRequestAddInputs(builder, inputs_vector_idx)
        PredictRequest.PredictRequestAddMsgIds(builder, msg_ids_idx)
        request_idx = PredictRequest.PredictRequestEnd(builder)
        builder.Finish(request_idx)
        request = builder.Output()

        return request

    def __str__(self):
        return ",".join(self.replica_addrs)

    def __repr__(self):
        return ",".join(self.replica_addrs)

