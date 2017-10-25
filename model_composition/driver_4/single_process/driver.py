import sys
import os
import argparse
import numpy as np
import json
import logging

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from single_proc_utils import HeavyNodeConfig, save_results
from models import resnet_18_model, kernel_svm_model
from models.deps import kernel_svm_utils

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(CURR_DIR, "models")

RESNET_MODEL_NAME = "resnet_18"
KERNEL_SVM_MODEL_NAME = "kernel_svm"

KERNEL_SVM_MODEL_PATH = os.path.join(MODELS_DIR, "kernel_svm_model_path", "kernel_svm_trained.sav")

########## Setup ##########

def get_heavy_node_configs(batch_size, allocated_cpus):
    resnet_config = HeavyNodeConfig(model_name=RESNET_MODEL_NAME,
                                    input_type="floats",
                                    allocated_cpus=allocated_cpus,
                                    gpus=[0],
                                    batch_size=batch_size)

    kernel_svm_config = HeavyNodeConfig(model_name=KERNEL_SVM_MODEL_NAME,
                                        input_type="floats",
                                        allocated_cpus=allocated_cpus,
                                        gpus=[],
                                        batch_size=batch_size)

    return [resnet_config, kernel_svm_config]

def create_resnet_model():
    return resnet_18_model.ResNet18Model()

def create_kernel_svm_model(model_path):
    return kernel_svm_model.KernelSVM(model_path)

def load_models():
    models_dict = {
        RESNET_MODEL_NAME : create_resnet_model(),
        KERNEL_SVM_MODEL_NAME : create_kernel_svm_model(KERNEL_SVM_MODEL_PATH)
    }
    return models_dict

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
        self.resnet_model = models_dict[RESNET_MODEL_NAME]
        self.kernel_svm_model = models_dict[KERNEL_SVM_MODEL_NAME]

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

    def predict(self, resnet_inputs):
        """
        Parameters
        ------------
        resnet_inputs : [np.ndarray]
            A list of image inputs, each represented as a numpy array
            of shape 224 x 224 x 3
        """

        batch_size = len(resnet_inputs)

        begin_time = datetime.now()

        resnet_svm_future = self.thread_pool.submit(
            lambda inputs : self.kernel_svm_model.predict(self.resnet_model.predict(inputs)), resnet_inputs)

        resnet_svm_classes = resnet_svm_future.result()

        end_time = datetime.now()

        latency = (end_time - begin_time).total_seconds()
        self.latencies.append(latency)
        self.total_num_complete += batch_size
        self.trial_num_complete += batch_size
        if self.trial_num_complete % self.trial_length == 0:
            self.print_stats()
            self.init_stats()

class DriverBenchmarker(object):
    def __init__(self, models_dict, trial_length, process_num):
        self.models_dict = models_dict
        self.trial_length = trial_length
        self.process_num = process_num

    def set_configs(self, configs):
        self.configs = configs

    def run(self, num_trials, batch_size):
        predictor = Predictor(self.models_dict, trial_length=self.trial_length)

        logger.info("Generating random inputs")
        resnet_inputs = [self._get_resnet_feats_input() for _ in range(1000)]
        resnet_inputs = [i for _ in range(40) for i in resnet_inputs]
        
        logger.info("Starting predictions")
        while True:
            batch_idx = np.random.randint(len(resnet_inputs) - batch_size)
            resnet_batch = resnet_inputs[batch_idx : batch_idx + batch_size]

            predictor.predict(resnet_batch)

            if len(predictor.stats["thrus"]) > num_trials:
                break

        save_results(self.configs, [predictor.stats], "single_proc_gpu_and_batch_size_experiments", self.process_num)

    def _get_resnet_feats_input(self):
        resnet_input = np.array(np.random.rand(224, 224, 3) * 255, dtype=np.float32)
        return resnet_input.flatten()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Single Process Image Driver 1')
    parser.add_argument('-d',  '--duration', type=int, default=120, help='The maximum duration of the benchmarking process in seconds, per iteration')
    parser.add_argument('-b',  '--batch_sizes', type=int, nargs='+', help="The batch size configurations to benchmark for the driver. Each configuration will be benchmarked separately.")
    parser.add_argument('-c',  '--cpus', type=int, nargs='+', help="The set of cpu cores on which to run the single process driver")
    parser.add_argument('-r',  '--resnet_gpu', type=int, default=0, help="The GPU on which to run the ResNet 152 featurization model")
    parser.add_argument('-t',  '--num_trials', type=int, default=15, help="The number of trials to run")
    parser.add_argument('-tl', '--trial_length', type=int, default=200, help="The length of each trial, in requests")
    parser.add_argument('-p',  '--process_number', type=int, default=0)
    
    args = parser.parse_args()

    if not args.cpus:
        raise Exception("The set of allocated cpus must be specified via the '--cpus' flag!")

    default_batch_size_confs = [2]
    batch_size_confs = args.batch_sizes if args.batch_sizes else default_batch_size_confs
    
    models_dict = load_models()
    benchmarker = DriverBenchmarker(models_dict, args.trial_length, args.process_number)

    for batch_size in batch_size_confs:
        configs = get_heavy_node_configs(batch_size=batch_size,
                                         allocated_cpus=args.cpus)
        benchmarker.set_configs(configs)
        benchmarker.run(args.num_trials, batch_size)