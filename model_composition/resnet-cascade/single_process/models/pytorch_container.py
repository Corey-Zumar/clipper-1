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
    def __init__(self, model_arch):
        if model_arch == "alexnet":
            logger.info("Using PyTorch Alexnet")
            self.model = models.alexnet(pretrained=True)
        # elif model_arch == "squeezenet":
        #     logger.info("Using PyTorch Squeezenet 1.1")
        #     self.model = models.squeezenet1_1(pretrained=True)
        elif model_arch == "res50":
            logger.info("Using PyTorch Resnet 50")
            self.model = models.resnet50(pretrained=True)
        elif model_arch == "res18":
            logger.info("Using PyTorch Resnet 18")
            self.model = models.resnet18(pretrained=True)
        elif model_arch == "res152":
            logger.info("Using PyTorch Resnet 152")
            self.model = models.resnet152(pretrained=True)
        # elif model_arch == "inception":
        #     logger.info("Using PyTorch Inception v3")
        #     self.model = models.inception_v3(pretrained=True)
        else:
            logger.error("{} is not currently supported".format(model_arch))
            sys.exit(1)

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

    def predict_doubles(self, inputs):
        start = datetime.now()
        input_arrs = []
        for t in inputs:
            i = t.reshape(self.height, self.width, 3)
            input_arrs.append(i)
        pred_classes = self._predict_raw(input_arrs)
        outputs = [str(l) for l in pred_classes]
        end = datetime.now()
        # logger.info("BATCH TOOK %f seconds" % (end - start).total_seconds())
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
        # logger.debug("Outputs: {}".format(outputs))
        end = datetime.now()
        # logger.info("BATCH TOOK %f seconds" % (end - start).total_seconds())
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

    input_type = "bytes"
    if "CLIPPER_INPUT_TYPE" in os.environ:
        input_type = os.environ["CLIPPER_INPUT_TYPE"]

    mpath = os.environ["CLIPPER_MODEL_PATH"]
    with open(mpath, "rb") as f:
        model_arch = f.read().strip()
    model = TorchContainer(model_arch)
    rpc_service = rpc.RPCService()
    rpc_service.start(model, ip, model_name, model_version, input_type)
