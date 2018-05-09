from __future__ import print_function
import rpc
import os
import sys
import numpy as np
import time
import torch

from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
from datetime import datetime

MODEL_NAME_ALEXNET = "m_alexnet:1"
MODEL_NAME_RESNET152 = "m_resnet:1"
MODEL_NAME_ENSEMBLE = "m_ensemble:1"

INCEPTION_WIDTH = 299
INCEPTION_HEIGHT = 299
INCEPTION_CHANNELS = 3

class TorchMMCContainer(rpc.ModelContainerBase):
    def __init__(self, gpu_num):
        self.gpu_num = gpu_num

	self.alexnet_model = models.alexnet(pretrained=True)
        self.resnet_model = models.resnet152(pretrained=True)

        if torch.cuda.is_available():
            self.alexnet_model.cuda(gpu_num)
            self.resnet_model.cuda(gpu_num)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.preprocessor = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])		

    def predict(self, inputs):
        outputs = {}

        if MODEL_NAME_ALEXNET in inputs:
            alexnet_inputs = inputs[MODEL_NAME_ALEXNET]
            alexnet_inputs = self._preprocess(alexnet_inputs)
            alexnet_outputs = self._predict(self.alexnet_model,
                                            alexnet_inputs)

            outputs[MODEL_NAME_ALEXNET] = alexnet_outputs

        if MODEL_NAME_RESNET152 in inputs:
            res152_inputs = inputs[MODEL_NAME_RESNET152]
            res152_inputs = self._preprocess(res152_inputs)
            res152_outputs = self._predict(self.resnet_model,
                                           res152_inputs)

            outputs[MODEL_NAME_RESNET152] = res152_outputs

        if MODEL_NAME_ENSEMBLE in inputs:
            ensemble_inputs = inputs[MODEL_NAME_ENSEMBLE]
            ensemble_inputs = self._preprocess(ensemble_inputs)
            alexnet_outputs = self._predict(self.alexnet_model, ensemble_inputs)
            res152_outputs = self._predict(self.resnet_model, ensemble_inputs)

            outputs[MODEL_NAME_ENSEMBLE] = alexnet_outputs

        return outputs

    def _preprocess(self, inputs):
	preprocessed = []
        for img in inputs:
            img = img.reshape(INCEPTION_WIDTH, INCEPTION_HEIGHT, INCEPTION_CHANNELS)
            img = Image.fromarray(img, mode="RGB")
            preprocessed.append(self.preprocessor(img))

        return preprocessed

    def _predict(self, model, inputs):
        input_batch = Variable(torch.stack(inputs, dim=0))
        if torch.cuda.is_available():
            input_batch = input_batch.cuda(self.gpu_num)

        logits = model(input_batch)
        maxes, arg_maxes = torch.max(logits, dim=1)
        pred_classes = arg_maxes.squeeze().data.cpu().numpy()
        outputs = [str(l) for l in pred_classes]
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

    predictor = TorchMMCContainer(gpu_num=0)
    rpc_service = rpc.RPCService()
    rpc_service.start(predictor, ip, model_info) 
