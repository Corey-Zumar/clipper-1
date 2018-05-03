
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

INCEPTION_WIDTH = 299
INCEPTION_HEIGHT = 299
INCEPTION_CHANNELS = 3

class SlowerPreprocessor(ModelBase):

    def __init__(self):
        ModelBase.__init__(self)

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
        outputs = []
        for img in inputs:
            img = img.reshape(INCEPTION_WIDTH, INCEPTION_HEIGHT, INCEPTION_CHANNELS)
            img = Image.fromarray(img, mode="RGB").resize((400,400))

            sharpener = ImageEnhance.Sharpness(img)
            img = sharpener.enhance(1.8)
            brightener = ImageEnhance.Brightness(img)
            img = brightener.enhance(1.5)
            contraster = ImageEnhance.Contrast(img)
            img = contraster.enhance(1.5)

            outputs.append(self.preprocess(img))

        return outputs
