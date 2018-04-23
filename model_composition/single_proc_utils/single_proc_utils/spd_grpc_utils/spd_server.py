import grpc

import numpy as np

# import spd_frontend_pb2
# import spd_frontend_pb2_grpc
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from flatbuffers import builder

from spd_grpc_consts import GRPC_OPTIONS
from flatbufs import spd_frontend_grpc_fb, PredictRequest, PredictResponse, FloatsInput


# class SpdFrontend(spd_frontend_pb2_grpc.PredictServicer):
class SpdFrontend(spd_frontend_grpc_fb.PredictServicer):
    
    def predict(self, inputs, msg_ids):
        pass

    def PredictFloats(self, request, context):
        t1 = datetime.now()
        msg_ids = request.MsgIdsAsNumpy()
        inputs = [request.Inputs(j).DataAsNumpy() for j in range(request.InputsLength())]
        t2 = datetime.now()

        output_ids = self.predict(inputs, msg_ids)

        output_ids_bytes = memoryview(output_ids.view(np.uint8))
        builder = flatbuffers.Builder(len(output_ids_bytes) * 2)
        PredictResponse.PredictResponseStartMsgIdsVector(builder, len(output_ids_bytes))
        builder.Bytes[builder.head : (builder.head + len(output_ids_bytes))] = output_ids_bytes 
        data = builder.EndVector(len(output_ids_bytes))
        PredictResponse.PredictResponseStart(builder)
        PredictResponse.PredictResponseAddMsgIds(builder, data)
        out_idx = PredictResponse.PredictResponseEnd(builder)
        builder.Finish(out_idx)
        response = builder.Output()

        # response = spd_frontend_pb2.PredictResponse(msg_ids=output_ids)

        after = datetime.now()
        print((after - t2).total_seconds(), (t2 - t1).total_seconds())
        
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

