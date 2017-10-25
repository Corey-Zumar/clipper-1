import sys
import os
import argparse
import numpy as np
import json
import logging

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from single_proc_utils import HeavyNodeConfig, save_results
from models import inception_model, opencv_svm_model, opencv_sift_feats_model

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(CURR_DIR, "models")

INCEPTION_MODEL_NAME = "inception"
OPENCV_SVM_MODEL_NAME = "opencv_svm"
OPENCV_SIFT_FEATS_MODEL_NAME = "opencv_sift_feats"

INCEPTION_MODEL_PATH = os.path.join(MODELS_DIR, "inception_model_data", "inception_v3.ckpt")
OPENCV_SVM_MODEL_PATH = os.path.join(MODELS_DIR, "opencv_svm_model_data", "opencv_svm_trained.sav")

TRIAL_LENGTH = 200

# The probability that the breed specialization
# branch is executed
SPECIALIZATION_BRANCH_THRESHOLD = .3

########## Setup ##########

def get_heavy_node_configs(batch_size, allocated_cpus, inception_gpus=[]):
    inception_config = HeavyNodeConfig(model_name=INCEPTION_MODEL_NAME,
                                       input_type="floats",
                                       allocated_cpus=allocated_cpus,
                                       gpus=inception_gpus,
                                       batch_size=batch_size)

    opencv_svm_config = HeavyNodeConfig(model_name=OPENCV_SVM_MODEL_NAME,
                                        input_type="floats",
                                        allocated_cpus=allocated_cpus,
                                        gpus=[],
                                        batch_size=batch_size)

    opencv_sift_feats_config = HeavyNodeConfig(model_name=OPENCV_SIFT_FEATS_MODEL_NAME,
                                               input_type="floats",
                                               allocated_cpus=allocated_cpus,
                                               gpus=[],
                                               batch_size=batch_size)

    return [inception_config, opencv_svm_config, opencv_sift_feats_config]

def load_models(inception_gpu):
    models_dict = {
        INCEPTION_MODEL_NAME : create_inception_model(INCEPTION_MODEL_PATH, gpu_num=inception_gpu),
        OPENCV_SVM_MODEL_NAME : create_opencv_svm_model(OPENCV_SVM_MODEL_PATH),
        OPENCV_SIFT_FEATS_MODEL_NAME : create_opencv_sift_feats_model()
    }
    return models_dict

def create_inception_model(model_path, gpu_num):
    return inception_model.InceptionModel(model_path, gpu_num=gpu_num)

def create_opencv_svm_model(model_path):
    return opencv_svm_model.OpenCVSVM(model_path)

def create_opencv_sift_feats_model():
    return opencv_sift_feats_model.SIFTFeaturizationModel()

########## Benchmarking ##########

class Predictor(object):

    def __init__(self, models_dict, trial_length):
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        # Stats
        self.init_stats()
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "mean_lats": []
        }
        self.total_num_complete = 0
        self.trial_length = trial_length

        # Models
        self.inception_model = models_dict[INCEPTION_MODEL_NAME]
        self.opencv_svm_model = models_dict[OPENCV_SVM_MODEL_NAME]
        self.opencv_sift_feats_model = models_dict[OPENCV_SIFT_FEATS_MODEL_NAME]

    def init_stats(self):
        self.latencies = []
        self.trial_num_complete = 0
        self.cur_req_id = 0
        self.start_time = datetime.now()

    def print_stats(self):
        lats = np.array(self.latencies)
        p99 = np.percentile(lats, 99)
        mean = np.mean(lats)
        end_time = datetime.now()
        thru = float(self.trial_num_complete) / (end_time - self.start_time).total_seconds()
        self.stats["thrus"].append(thru)
        self.stats["p99_lats"].append(p99)
        self.stats["mean_lats"].append(mean)
        logger.info("p99: {p99}, mean: {mean}, thruput: {thru}".format(p99=p99,
                                                                       mean=mean,
                                                                       thru=thru))

    def predict(self, inception_inputs):
        """
        Parameters
        ------------
        inception_inputs : [np.ndarray]
            A list of image inputs, each represented as a numpy array
            of shape 299 x 299 x 3
        """

        batch_size = len(inception_inputs)

        begin_time = datetime.now()

        inception_future = self.thread_pool.submit(self.inception_model.predict, inception_inputs)
        inception_future.result()

        if np.random.rand() < SPECIALIZATION_BRANCH_THRESHOLD:
            opencv_future = self.thread_pool.submit(
                lambda inputs : self.opencv_svm_model.predict(
                    self.opencv_sift_feats_model.predict(inputs)), inception_inputs)

            opencv_future.result()

        end_time = datetime.now()

        latency = (end_time - begin_time).total_seconds()
        self.latencies.append(latency)
        self.total_num_complete += batch_size
        self.trial_num_complete += batch_size
        if self.trial_num_complete % self.trial_length == 0:
            self.print_stats()
            self.init_stats()

class DriverBenchmarker(object):
    def __init__(self, models_dict, trial_length):
        self.predictor = Predictor(models_dict, trial_length)

    def set_configs(self, configs):
        self.configs = configs

    def run(self, num_trials, batch_size):
        logger.info("Generating random inputs")

        inception_inputs = [self._get_inception_input() for _ in range(1000)]
        inception_inputs = [i for _ in range(40) for i in inception_inputs]
        
        logger.info("Starting predictions")
        while True:
            batch_idx = np.random.randint(len(inception_inputs) - batch_size)
            inception_batch = inception_inputs[batch_idx : batch_idx + batch_size]

            self.predictor.predict(inception_batch)

            if len(self.predictor.stats["thrus"]) > num_trials:
                break

        save_results(self.configs, [self.predictor.stats], "single_proc_gpu_and_batch_size_experiments")

    def _get_inception_input(self):
        # There's no need to flatten this input for a single-process model
        inception_input = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
        return inception_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Single Process Image Driver 1')
    parser.add_argument('-d', '--duration', type=int, default=120, help='The maximum duration of the benchmarking process in seconds, per iteration')
    parser.add_argument('-b', '--batch_sizes', type=int, nargs='+', help="The batch size configurations to benchmark for the driver. Each configuration will be benchmarked separately.")
    parser.add_argument('-c', '--cpus', type=int, nargs='+', help="The set of cpu cores on which to run the single process driver")
    parser.add_argument('-i', '--inception_gpu', type=int, default=0, help="The GPU on which to run the inception featurization model")
    parser.add_argument('-t', '--num_trials', type=int, default=15, help="The number of trials to run")
    parser.add_argument('-tl', '--trial_length', type=int, default=200, help="The length of each trial, in requests")
    
    args = parser.parse_args()

    if not args.cpus:
        raise Exception("The set of allocated cpus must be specified via the '--cpus' flag!")

    default_batch_size_confs = [2]
    batch_size_confs = args.batch_sizes if args.batch_sizes else default_batch_size_confs
    
    models_dict = load_models(args.inception_gpu)
    benchmarker = DriverBenchmarker(models_dict, args.trial_length)

    for batch_size in batch_size_confs:
        configs = get_heavy_node_configs(batch_size=batch_size,
                                         allocated_cpus=args.cpus,
                                         inception_gpus=[args.inception_gpu])
        benchmarker.set_configs(configs)
        benchmarker.run(args.num_trials, batch_size)
