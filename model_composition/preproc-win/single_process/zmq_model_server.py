import sys
import os
import argparse
import numpy as np
import logging
import time

from datetime import datetime
from threading import Thread, Lock
from PIL import Image, ImageEnhance

from single_proc_utils.spd_zmq_utils import SpdFrontend, SpdServer
from models import cascade_model, preprocessor
from models.cascade_model import CASCADE_MODEL_ARCHITECTURE_ALEXNET

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(CURR_DIR, "models")

ALEXNET_MODEL_NAME = "alexnet"
PREPROCESSOR_MODEL_NAME = "preprocessor"

# AlexNet is the first model in the pipeline and is 
# therefore always queried
ALEXNET_QUERY_PROBABILITY = 1

RESULTS_DIR = "/results"

ONE_DAY_IN_SECONDS = 60 * 60 * 24
SERVER_HOST_IP = "localhost"

WARMING_UP_DEFAULT_RESPONSE = [-1]

########## Setup ##########
def create_alexnet_model(gpu_num):
    return cascade_model.CascadeModel(CASCADE_MODEL_ARCHITECTURE_ALEXNET, gpu_num)

def create_preprocessor():
    return preprocessor.SlowerPreprocessor()

def load_models(alexnet_gpu):
    models_dict = {
        ALEXNET_MODEL_NAME : create_alexnet_model(alexnet_gpu),
        PREPROCESSOR_MODEL_NAME : create_preprocessor()
    }

    return models_dict

########## Input Generation ##########

def generate_inputs():
    cascade_inputs = [_get_cascade_input() for _ in range(1000)]
    cascade_inputs = [i for _ in range(40) for i in cascade_inputs]

    return np.array(cascade_inputs, dtype=np.float32)

def _get_cascade_input():
    # These 299 x 299 x 3 ImageNet inputs will be downscaled
    # to the appropriate size for AlexNet
    cascade_input = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
    return cascade_input.flatten()

########## Frontend Generation ##########

class CascadeFrontend(SpdFrontend):

    def __init__(self, models_dict, warmup=True):
        self.models_dict = models_dict

        # Models
        self.alexnet_model = models_dict[ALEXNET_MODEL_NAME]
        self.preprocessor = models_dict[PREPROCESSOR_MODEL_NAME]

        self.warmup_lock = Lock()
        self.warming_up = False
        if warmup:
            logger.info("Warming up...")
            self.warming_up = True
            warmup_thread = Thread(target=self._warm_up)
            warmup_thread.start()

        self.lats = []

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
        t0 = datetime.now()

        inputs = self.preprocessor.predict(inputs)

        t1 = datetime.now()

        self.alexnet_model.predict(inputs)

        t2 = datetime.now()

        preprocessing_latency = (t1 - t0).total_seconds()
        model_eval_latency = (t2 - t1).total_seconds()

        self.lats.append((preprocessing_latency, model_eval_latency))
        if len(self.lats) >= 100:
            preproc_lats, eval_lats = zip(*self.lats)
            p99_preproc = np.percentile(preproc_lats, 99)
            mean_preproc = np.mean(preproc_lats)
            std_preproc = np.std(preproc_lats)

            p99_eval = np.percentile(eval_lats, 99)
            mean_eval = np.mean(eval_lats)
            std_eval = np.std(eval_lats)

            print("PREPROC - p99: {}, mean: {}, std: {}".format(p99_preproc, 
                                                                mean_preproc, 
                                                                std_preproc))


            print("MODEL EVAL - p99: {}, mean: {}, std: {}".format(p99_eval, 
                                                                   mean_eval, 
                                                                   std_eval))

            self.lats = []

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
    parser.add_argument('-a', '--alexnet_gpu', type=int, default=0, help="The GPU on which to run the alexnet classification model")
    parser.add_argument('-p',  '--port', type=int, help="The port on which to run the grpc server")
    parser.add_argument('-nw', '--no_warmup', action="store_true", help="If true, disables warmup")

    args = parser.parse_args()

    models_dict = load_models(args.alexnet_gpu)

    warmup = not args.no_warmup
    frontend = CascadeFrontend(models_dict, warmup)
    server = SpdServer(frontend, SERVER_HOST_IP, args.port)
    server.start()
    
    try:
        while True:
            time.sleep(ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop()
