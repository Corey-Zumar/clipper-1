import spd_frontend_pb2
import spd_frontend_pb2_grpc

class SpdFrontend(spd_frontend_pb2_grpc.PredictServicer):
    
    def predict(self.inputs, msg_ids):
        pass

    def PredictFloats(self, request, context):
        inputs = np.array([np.array(inp.input, dtype=np.float32) for inp in request.inputs])
        msg_ids = np.array(request.msg_ids, dtype=np.int32)

        output_ids = self.predict(inputs, msg_ids)
        response = spd_frontend_pb2.PredictResponse(msg_ids=output_ids)
        
        return response






