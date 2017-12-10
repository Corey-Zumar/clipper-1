from __future__ import print_function, absolute_import, division
import rpc
import os
import sys
import numpy as np
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
import logging
from datetime import datetime

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


class TorchContainer(rpc.ModelContainerBase):
    def __init__(self):
        self.model = models.resnet152(pretrained=True)
        layers = list(self.model.children())[:-1] # delete the last fc layer.
        self.model = torch.nn.Sequential(*layers)

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

    def predict_doubles(self, inputs):
        start = datetime.now()
        inputs = []
        for t in inputs:
            i = t.reshape(3, self.height, self.width)
            img = Image.fromarray(i, mode="RGB")
            inputs.append(self.preprocess(img))

        input_batch = Variable(torch.stack(inputs, dim=0))
        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        all_logits = self.model(input_batch)
        outputs = []
        for logits in all_logits:
            outputs.append(np.array(np.squeeze(logits), dtype=np.float32).flatten())

        return outputs

    def predict_floats(self, inputs):
        return self.predict_doubles(inputs)

    def predict_bytes(self, inputs):
        start = datetime.now()
        input_arrs = []
        for byte_arr in inputs:
            t = np.frombuffer(byte_arr, dtype=np.float32)
            i = t.reshape(self.height, self.width, 3)
            input_arrs.append(i)
        pred_classes = self._predict_raw(input_arrs)
        outputs = [str(l) for l in pred_classes]
        logger.debug("Outputs: {}".format(outputs))
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
        logits = self.model(input_batch)
        maxes, arg_maxes = torch.max(logits, dim=1)
        pred_classes = arg_maxes.squeeze().data.cpu().numpy()
        return pred_classes


if __name__ == "__main__":
    try:
        model_name = os.environ["CLIPPER_MODEL_NAME"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_NAME environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        model_version = os.environ["CLIPPER_MODEL_VERSION"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_VERSION environment variable must be set",
            file=sys.stdout)
        sys.exit(1)

    ip = "127.0.0.1"
    if "CLIPPER_IP" in os.environ:
        ip = os.environ["CLIPPER_IP"]
    else:
        print("Connecting to Clipper on localhost")

    port = 7000
    if "CLIPPER_PORT" in os.environ:
        port = int(os.environ["CLIPPER_PORT"])
    else:
        print("Connecting to Clipper with default port: 7000")

    input_type = "floats"
    if "CLIPPER_INPUT_TYPE" in os.environ:
        input_type = os.environ["CLIPPER_INPUT_TYPE"]

    model = TorchContainer()
    rpc_service = rpc.RPCService()
    rpc_service.start(model, ip, port, model_name, model_version, input_type)
