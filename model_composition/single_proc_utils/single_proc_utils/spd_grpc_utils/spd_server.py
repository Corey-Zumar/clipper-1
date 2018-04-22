import grpc

import numpy as np

import spd_frontend_pb2
import spd_frontend_pb2_grpc

from concurrent.futures import ThreadPoolExecutor

class SpdFrontend(spd_frontend_pb2_grpc.PredictServicer):
    
    def predict(self, inputs, msg_ids):
        pass

    def PredictFloats(self, request, context):
        inputs = np.array([np.array(inp.input, dtype=np.float32) for inp in request.inputs])
        msg_ids = np.array(request.msg_ids, dtype=np.int32)

        output_ids = self.predict(inputs, msg_ids)
        response = spd_frontend_pb2.PredictResponse(msg_ids=output_ids)
        
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
        self.server = grpc.server(ThreadPoolExecutor(max_workers=1)) 

        spd_frontend_pb2_grpc.add_PredictServicer_to_server(self.spd_frontend, self.server)

        address = "{host}:{port}".format(host=self.host, port=self.port)
        self.server.add_insecure_port(address)
        
        # Start server
        self.server.start()

    def stop(self):
        if self.server:
            self.server.stop()

