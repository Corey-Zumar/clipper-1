import sys
import os
import argparse
import numpy as np
import logging
import time

from datetime import datetime
from threading import Thread, Lock
from PIL import Image

from single_proc_utils.spd_zmq_utils import SpdFrontend, SpdServer
from models import cascade_model
from models.cascade_model import CASCADE_MODEL_ARCHITECTURE_RES50, CASCADE_MODEL_ARCHITECTURE_RES152, CASCADE_MODEL_ARCHITECTURE_ALEXNET

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(CURR_DIR, "models")

RES50_MODEL_NAME = "res50"
RES152_MODEL_NAME = "res152"
ALEXNET_MODEL_NAME = "alexnet"

# AlexNet is the first model in the pipeline and is 
# therefore always queried
ALEXNET_QUERY_PROBABILITY = 1

RES50_QUERY_PROBABILITY = 1 - (0.192)
RES152_QUERY_PROBABILITY = 1 - (.9)

RESULTS_DIR = "/results"

ONE_DAY_IN_SECONDS = 60 * 60 * 24
SERVER_HOST_IP = "localhost"

WARMING_UP_DEFAULT_RESPONSE = [-1]

########## Setup ##########
def create_res50_model(gpu_num):
    return cascade_model.CascadeModel(CASCADE_MODEL_ARCHITECTURE_RES50, gpu_num)

def create_res152_model(gpu_num):
    return cascade_model.CascadeModel(CASCADE_MODEL_ARCHITECTURE_RES152, gpu_num)

def create_alexnet_model(gpu_num):
    return cascade_model.CascadeModel(CASCADE_MODEL_ARCHITECTURE_ALEXNET, gpu_num)

def load_models(res50_gpu, res152_gpu, alexnet_gpu):
    models_dict = {
        RES50_MODEL_NAME : create_res50_model(res50_gpu),
        RES152_MODEL_NAME : create_res152_model(res152_gpu),
        ALEXNET_MODEL_NAME : create_alexnet_model(alexnet_gpu)
    }

    return models_dict

########## Input Generation ##########

def generate_inputs():
    cascade_inputs = [_get_cascade_input() for _ in range(1000)]
    cascade_inputs = [i for _ in range(40) for i in cascade_inputs]

    return np.array(cascade_inputs, dtype=np.float32)

def _get_cascade_input():
    # These 299 x 299 x 3 ImageNet inputs will be downscaled
    # to the appropriate size for AlexNet and ResNet50/152
    cascade_input = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
    return cascade_input.flatten()

########## Frontend Generation ##########

class CascadeFrontend(SpdFrontend):

    def __init__(self, models_dict, warmup=True):
        self.models_dict = models_dict

        # Models
        self.res50_model = models_dict[RES50_MODEL_NAME]
        self.res152_model = models_dict[RES152_MODEL_NAME]
        self.alexnet_model = models_dict[ALEXNET_MODEL_NAME]

        self.warmup_lock = Lock()
        self.warming_up = False
        if warmup:
            logger.info("Warming up...")
            self.warming_up = True
            warmup_thread = Thread(target=self._warm_up)
            warmup_thread.start()

    def predict(self, inputs, msg_ids):
        """
        Parameters
        ------------
        inputs : np.ndarray
            An array of flattened, 224 * 224 * 3 float-typed input images

        Returns 
        ------------
        [int]
            A list of output msg ids
        """
        self.warmup_lock.acquire()
        warming_up = self.warming_up
        self.warmup_lock.release()
        if warming_up:
            return WARMING_UP_DEFAULT_RESPONSE

        return self._predict(inputs, msg_ids)

    def _predict(self, inputs, msg_ids):
        self.alexnet_model.predict(inputs)

        if np.random.rand() < RES50_QUERY_PROBABILITY:
            self.res50_model.predict(inputs)

            if np.random.rand() < RES152_QUERY_PROBABILITY:
                self.res152_model.predict(inputs)

        return msg_ids

    def _warm_up(self):
        logger.info("Generating warmup inputs...")
        warmup_inputs = generate_inputs()

        logger.info("Running warmup...")
        for batch_size in [32, 70]:
            warmup_lats = []
            warmup_batch_sizes = []
            for i in range(1000):
                bs = max(1, int(batch_size * (1 + np.random.normal(0, .2))))
                batch_idxs = np.random.randint(0, len(warmup_inputs), bs)
                curr_inputs = warmup_inputs[batch_idxs]
                msg_ids = range(bs)

                begin = datetime.now()
                self._predict(curr_inputs, msg_ids)
                end = datetime.now()
                batch_latency = (end - begin).total_seconds()
                
                warmup_lats.append(batch_latency)
                warmup_batch_sizes.append(bs)

                if i % 30 == 0:
                    p99_lat = np.percentile(warmup_lats, 99)
                    mean_batch = np.mean(warmup_batch_sizes)
                    logger.info("Warmup - p99 batch latency: {}, mean_batch: {}".format(p99_lat, mean_batch))
                    warmup_lats = []
                    warmup_batch_sizes = []

        self.warmup_lock.acquire() 
        self.warming_up = False
        self.warmup_lock.release() 

        logger.info("Warmup complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Single Process Image Driver 1')
    parser.add_argument('-r50',  '--res50_gpu', type=int, default=0, help="The GPU on which to run the ResNet 50 featurization model")
    parser.add_argument('-r152', '--res152_gpu', type=int, default=1, help="The GPU on which to run the ResNet 152 classification model")
    parser.add_argument('-a', '--alexnet_gpu', type=int, default=2, help="The GPU on which to run the alexnet classification model")
    parser.add_argument('-p',  '--port', type=int, help="The port on which to run the grpc server")
    parser.add_argument('-nw', '--no_warmup', action="store_true", help="If true, disables warmup")

    args = parser.parse_args()

    models_dict = load_models(args.res50_gpu, args.res152_gpu, args.alexnet_gpu)

    warmup = not args.no_warmup
    frontend = CascadeFrontend(models_dict, warmup)
    server = SpdServer(frontend, SERVER_HOST_IP, args.port)
    server.start()
    
    try:
        while True:
            time.sleep(ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop()
