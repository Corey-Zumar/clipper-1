from __future__ import print_function, absolute_import, division
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
import logging
from datetime import datetime

from single_proc_utils import ModelBase

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class ResNet18Model(ModelBase):
    def __init__(self):
        ModelBase.__init__(self)

        self.model = models.resnet18(pretrained=True)

        self.model = nn.Sequential(*list(self.model.children())[:-1])

        if torch.cuda.is_available():
            self.model.cuda()

        self.model.eval()
        self.height = 224
        self.width = 224

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
        start = datetime.now()
        input_arrs = []
        for t in inputs:
            i = t.reshape(self.height, self.width, 3)
            input_arrs.append(i)
        pred_classes = self._predict_raw(input_arrs)
        outputs = [str(l) for l in pred_classes]
        end = datetime.now()
        logger.info("BATCH TOOK %f seconds" % (end - start).total_seconds())
        return outputs

    def _predict_raw(self, input_arrs):
        inputs = []
        for i in input_arrs:
            img = Image.fromarray(i, mode="RGB")
            inputs.append(self.preprocess(img))
        input_batch = Variable(torch.stack(inputs, dim=0))
        if torch.cuda.is_available():
            input_batch = input_batch.cuda()
        features = self.model(input_batch)
        return features
