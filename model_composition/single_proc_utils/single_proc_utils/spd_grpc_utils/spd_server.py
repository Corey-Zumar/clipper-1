import grpc

import numpy as np

import spd_frontend_pb2
import spd_frontend_pb2_grpc

from datetime import datetime

from spd_grpc_consts import GRPC_OPTIONS

from concurrent.futures import ThreadPoolExecutor

class SpdFrontend(spd_frontend_pb2_grpc.PredictServicer):
    
    def predict(self, inputs, msg_ids):
        pass

    def PredictFloats(self, request, context):
        t0 = datetime.now()
        
        msg_ids = np.array(request.msg_ids, dtype=np.int32)
        batch_size = len(msg_ids) 
        
        t05 = datetime.now()
        
        inp_item = np.frombuffer(request.inputs[0], dtype=np.float32)
        inputs = np.reshape(inp_item, (batch_size, -1))
        
        t1 = datetime.now()

        output_ids = self.predict(inputs, msg_ids)

        t2 = datetime.now()
        
        response = spd_frontend_pb2.PredictResponse(msg_ids=output_ids)
        
        t3 = datetime.now()

        print((t3 - t2).total_seconds(), (t2 - t1).total_seconds(), (t1 - t05).total_seconds(), (t05 - t0).total_seconds())
        
        return response

class SpdServer:

    def __init__(self, spd_frontend, host, port):
        self.spd_frontend = spd_frontend
        self.host = host
        self.port = port

        self.server = None

    def start(self):
        # Create a threadpool consisting of a single worker
        # for synchronous predictions
        self.server = grpc.server(ThreadPoolExecutor(max_workers=1), options=GRPC_OPTIONS) 

        spd_frontend_pb2_grpc.add_PredictServicer_to_server(self.spd_frontend, self.server)

        address = "{host}:{port}".format(host=self.host, port=self.port)
        self.server.add_insecure_port(address)
        
        # Start server
        self.server.start()

    def stop(self):
        if self.server:
            self.server.stop()

