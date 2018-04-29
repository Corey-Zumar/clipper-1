import sys
import os
import logging
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

        if model_architecture == CASCADE_MODEL_ARCHITECTURE_RES50:
            self.model = models.resnet50(pretrained=True)
        elif model_architecture == CASCADE_MODEL_ARCHITECTURE_RES152:
            self.model = models.resnet152(pretrained=True)
        elif model_architecture == CASCADE_MODEL_ARCHITECTURE_ALEXNET:
            self.model = models.alexnet(pretrained=True)
        else:
            raise Exception("Invalid architecture specified for cascade model: {}".format(model_architecture))

        if torch.cuda.is_available():
            self.model.cuda()

        self.model.eval()
        self.height = 299
        self.width = 299

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.preprocess = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    def predict(self, inputs):
        input_arrs = []
        for t in inputs:
            i = t.reshape(self.height, self.width, 3)
            input_arrs.append(i)
        pred_classes = self._predict_raw(input_arrs)
        outputs = [str(l) for l in pred_classes]
        return outputs

    def _predict_raw(self, input_arrs):
        inputs = []
        for i in input_arrs:
            img = Image.fromarray(i, mode="RGB")
            inputs.append(self.preprocess(img))
        input_batch = Variable(torch.stack(inputs, dim=0))
        if torch.cuda.is_available():
            input_batch = input_batch.cuda()
        logits = self.model(input_batch)
        maxes, arg_maxes = torch.max(logits, dim=1)
        pred_classes = arg_maxes.squeeze().data.cpu().numpy()
        return pred_classes

