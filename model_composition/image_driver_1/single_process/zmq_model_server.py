import sys
import os
import argparse
import numpy as np
import logging
import time

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock
from PIL import Image

from single_proc_utils.spd_zmq_utils import SpdFrontend, SpdServer
from models import tf_resnet_model, inception_feats_model, tf_kernel_svm_model, tf_log_reg_model, tf_alexnet_model

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(CURR_DIR, "models")

TF_INCEPTION_FEATS_MODEL_NAME = "inception_feats"
TF_KERNEL_SVM_MODEL_NAME = "kernel_svm"
TF_LOG_REG_MODEL_NAME = "tf_log_reg"
TF_RESNET_FEATS_MODEL_NAME = "tf_resnet_feats"

TF_ALEXNET_FEATS_MODEL_NAME = "tf_alexnet_feats"

RESULTS_DIR = "/results"

INCEPTION_MODEL_PATH = os.path.join(MODELS_DIR, "inception_model_data", "inception_feats_graph_def.pb")
RESNET_MODEL_PATH = os.path.join(MODELS_DIR, "tf_resnet_model_data")
ALEXNET_MODEL_PATH = os.path.join(MODELS_DIR, "tf_alexnet_model_data")

ONE_DAY_IN_SECONDS = 60 * 60 * 24
SERVER_HOST_IP = "localhost"

INCEPTION_IMAGE_SHAPE = (299, 299, 3)
RESNET_IMAGE_SHAPE = (224, 224, 3)
ALEXNET_IMAGE_SHAPE = (224, 224, 3)

WARMING_UP_DEFAULT_RESPONSE = [-1]

########## Setup ##########
def create_alexnet_model(model_path, gpu_num):
    return tf_alexnet_model.AlexNetModel(model_path, gpu_num)

def create_resnet_model(model_path, gpu_num):
    return tf_resnet_model.TfResNetModel(model_path, gpu_num)

def create_kernel_svm_model():
    return tf_kernel_svm_model.TFKernelSVM()

def create_inception_model(model_path, gpu_num):
    return inception_feats_model.InceptionFeaturizationModel(model_path, gpu_num=gpu_num)

def create_log_reg_model():
    return tf_log_reg_model.TfLogRegModel()

# def load_models(resnet_gpu, inception_gpu):
def load_models(alexnet_gpu, inception_gpu):
    models_dict = {
        # TF_RESNET_FEATS_MODEL_NAME : create_resnet_model(RESNET_MODEL_PATH, gpu_num=resnet_gpu),
        TF_ALEXNET_FEATS_MODEL_NAME : create_alexnet_model(ALEXNET_MODEL_PATH, gpu_num=alexnet_gpu),
        TF_KERNEL_SVM_MODEL_NAME : create_kernel_svm_model(),
        TF_INCEPTION_FEATS_MODEL_NAME : create_inception_model(INCEPTION_MODEL_PATH, gpu_num=inception_gpu),
        TF_LOG_REG_MODEL_NAME : create_log_reg_model()
    }
    return models_dict

########## Input Generation ##########

def generate_inputs():
    inception_inputs = [_get_inception_input() for _ in range(1000)]
    inception_inputs = [i for _ in range(40) for i in inception_inputs]

    return np.array(inception_inputs)

def _get_inception_input():
    inception_input = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
    return inception_input.flatten()

########## Frontend Generation ##########

class ID1Frontend(SpdFrontend):

    def __init__(self, models_dict, warmup=True):
        self.task_execution_thread_pool = ThreadPoolExecutor(max_workers=2) 
        self.models_dict = models_dict

        # Models
        # self.resnet_model = models_dict[TF_RESNET_FEATS_MODEL_NAME]
        self.alexnet_model = models_dict[TF_ALEXNET_FEATS_MODEL_NAME]
        self.kernel_svm_model = models_dict[TF_KERNEL_SVM_MODEL_NAME]
        self.inception_model = models_dict[TF_INCEPTION_FEATS_MODEL_NAME]
        self.log_reg_model = models_dict[TF_LOG_REG_MODEL_NAME]

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
            An array of flattened, inception-sized input images

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
        # resnet_inputs = self._transform_inputs_resnet(inputs)
        alexnet_inputs = self._transform_inputs_alexnet(inputs)
        inception_inputs = self._transform_inputs_inception(inputs)

        # self._predict_parallel(resnet_inputs, inception_inputs)
        self._predict_parallel(alexnet_inputs, inception_inputs)

        return msg_ids

    # def _predict_parallel(self, resnet_inputs, inception_inputs):
    def _predict_parallel(self, alexnet_inputs, inception_inputs):
            alexnet_svm_future = self.task_execution_thread_pool.submit(
                lambda inputs : self.kernel_svm_model.predict(self.alexnet_model.predict(inputs)), alexnet_inputs)
            # resnet_svm_future = self.task_execution_thread_pool.submit(
            #     lambda inputs : self.kernel_svm_model.predict(self.resnet_model.predict(inputs)), resnet_inputs)
            
            inception_log_reg_future = self.task_execution_thread_pool.submit(
                lambda inputs : self.log_reg_model.predict(self.inception_model.predict(inputs)), inception_inputs)

            # resnet_svm_classes = resnet_svm_future.result()
            alexnet_svm_classes = alexnet_svm_future.result()
            inception_log_reg_classes = inception_log_reg_future.result()

    def _warm_up(self):
        logger.info("Generating warmup inputs...")
        warmup_inputs = generate_inputs()

        logger.info("Running warmup...")
        for batch_size in [32, 70, 140]:
            warmup_lats = []
            warmup_batch_sizes = []
            for i in range(1000):
                bs = max(1, int(batch_size * (1 + np.random.normal(0, .2))))
                bs = min(bs, 160)
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

    # def _transform_inputs_resnet(self, inputs):
    def _transform_inputs_alexnet(self, inputs):
        resized = []
        for inp in inputs:
            w, h, c = INCEPTION_IMAGE_SHAPE
            img = Image.fromarray(inp.reshape(w, h, c).astype(np.uint8), mode="RGB")
            # img = img.resize(RESNET_IMAGE_SHAPE[:2])
            img = img.resize(ALEXNET_IMAGE_SHAPE[:2])
            resized.append(np.asarray(img, dtype=np.float32))

        return resized

    def _transform_inputs_inception(self, inputs):
        inception_image_size = np.prod(INCEPTION_IMAGE_SHAPE)
        assert len(inputs.shape) == 2
        assert inputs.shape[1] == inception_image_size

        return inputs.reshape((-1, 299, 299, 3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Single Process Image Driver 1')
    # parser.add_argument('-r',  '--resnet_gpu', type=int, default=0, help="The GPU on which to run the ResNet 152 featurization model")
    parser.add_argument('-a',  '--alexnet_gpu', type=int, default=0, help="The GPU on which to run the AlexNet featurization model")
    parser.add_argument('-i',  '--inception_gpu', type=int, default=1, help="The GPU on which to run the inception featurization model")
    parser.add_argument('-p',  '--port', type=int, help="The port on which to run the grpc server")
    parser.add_argument('-nw', '--no_warmup', action="store_true", help="If true, disables warmup")

    args = parser.parse_args()

    # models_dict = load_models(args.resnet_gpu, args.inception_gpu)
    models_dict = load_models(args.alexnet_gpu, args.inception_gpu)

    warmup = not args.no_warmup
    frontend = ID1Frontend(models_dict, warmup)
    server = SpdServer(frontend, SERVER_HOST_IP, args.port)
    server.start()
    
    try:
        while True:
            time.sleep(ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop()
