import sys
import os
import logging
import torch
import numpy as np

from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
from datetime import datetime

from single_proc_utils import ModelBase

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

CASCADE_MODEL_ARCHITECTURE_RES50 = "res50"
CASCADE_MODEL_ARCHITECTURE_RES152 = "res152"
CASCADE_MODEL_ARCHITECTURE_ALEXNET = "alexnet"

class CascadeModel(ModelBase):

    def __init__(self, model_architecture, gpu_num):
        ModelBase.__init__(self)

        self.gpu_num = gpu_num

        self.model_architecture = model_architecture

        if model_architecture == CASCADE_MODEL_ARCHITECTURE_RES50:
            self.model = models.resnet50(pretrained=True)
        elif model_architecture == CASCADE_MODEL_ARCHITECTURE_RES152:
            self.model = models.resnet152(pretrained=True)
        elif model_architecture == CASCADE_MODEL_ARCHITECTURE_ALEXNET:
            self.model = models.alexnet(pretrained=True)
        else:
            raise Exception("Invalid architecture specified for cascade model: {}".format(model_architecture))

        if torch.cuda.is_available():
            # Place model on the GPU specified by 'gpu_num'
            print("Initializing model with architecture: {arc} on GPU: {gn}".format(arc=model_architecture, gn=self.gpu_num))
            self.model.cuda(self.gpu_num)

        self.model.eval()

    def predict(self, inputs):
        input_batch = Variable(torch.stack(inputs, dim=0))
        if torch.cuda.is_available():
            input_batch = input_batch.cuda(self.gpu_num)
        logits = self.model(input_batch)

        maxes, arg_maxes = torch.max(logits, dim=1)
        pred_classes = arg_maxes.squeeze().data.cpu().numpy()
        outputs = [str(l) for l in pred_classes]
        return outputs

