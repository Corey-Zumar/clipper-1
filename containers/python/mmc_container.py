from __future__ import print_function
import rpc
import os
import sys
import numpy as np

class MMCContainer(rpc.ModelContainerBase):
    def __init__(self, prediction="1.0"):
        self.prediction = prediction

    def predict(self, inputs):
        outputs = {}
        for model_name in inputs:
            outputs[model_name] = [self.prediction] * len(inputs[model_name])

        return outputs 

def parse_model_info(raw_model_info):
    model_info = []
    names_versions = raw_model_info.split(",")
    for item in names_versions:
        name, version = item.split(":")
        model_info.append((name, version))

    return model_info

if __name__ == "__main__":
    try:
        raw_model_info = os.environ["CLIPPER_MODEL_INFO"]
        model_info = parse_model_info(raw_model_info)
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_INFO environment variable must be set",
            file=sys.stdout)
        sys.exit(1)

    ip = "127.0.0.1"
    if "CLIPPER_IP" in os.environ:
        ip = os.environ["CLIPPER_IP"]
    else:
        print("Connecting to Clipper on localhost")

    predictor = MMCContainer()
    rpc_service = rpc.RPCService()
    rpc_service.start(predictor, ip, model_info) 
