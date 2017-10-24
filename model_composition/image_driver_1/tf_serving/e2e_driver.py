import sys
import os
import argparse
import numpy as np
import time
import base64
import logging
import json
import tensorflow as tf

from threading import Lock
from datetime import datetime
from multiprocessing import Process, Queue

from tf_serving_utils import GRPCClient, ReplicaAddress
from tf_serving_utils import tfs_utils

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

# Models and applications for each heavy node
# will share the same name
INCEPTION_FEATS_MODEL_NAME = "inception"
RESNET_152_MODEL_NAME = "resnet"
LOG_REG_MODEL_NAME = "log_reg"
KERNEL_SVM_MODEL_NAME = "kernel_svm"

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_BASE_DIR_PATH = os.path.join(CURR_DIR, "exported_tf_models")

INCEPTION_FEATS_MODEL_BASE_PATH = os.path.join(MODEL_BASE_DIR_PATH, "inception_tfserve")
RESNET_152_MODEL_BASE_PATH = os.path.join(MODEL_BASE_DIR_PATH, "resnet_tfserve")
LOG_REG_MODEL_BASE_PATH = os.path.join(MODEL_BASE_DIR_PATH, "log_reg_tfserve")
KERNEL_SVM_MODEL_BASE_PATH = os.path.join(MODEL_BASE_DIR_PATH, "kernel_svm_tfserve")

INCEPTION_FEATS_OUTPUT_KEY = "feats"
RESNET_FEATS_OUTPUT_KEY = "feats"
KERNEL_SVM_OUTPUT_KEY = "outputs"
LOG_REG_OUTPUT_KEY = "outputs"

INCEPTION_PORTS = range(9500,9508)
RESNET_152_PORTS = range(9508, 9516)
LOG_REG_PORTS = range(9516, 9524)
KERNEL_SVM_PORTS = range(9524, 9532)

TFS_ADDRESS = "localhost"

########## Setup ##########

def setup_heavy_nodes(configs):
    """
    Parameters
    ------------
    configs : dict
        Dictionary of TFSHeavyNodeConfig objects,
        keyed on model names
    """

    for config in configs.values():
        tfs_utils.setup_heavy_node(config)
    
    time.sleep(5)

def create_clients(configs):
    """
    Parameters
    ------------
    configs : dict
        Dictionary of TFSHeavyNodeConfig objects,
        keyed on model names
    """
    clients = {}
    for key in configs:
        replica_addrs = [ReplicaAddress(TFS_ADDRESS, int(port)) for port in configs[key].ports]
        client = GRPCClient(replica_addrs)
        client.start()
        clients[key] = client

    return clients

def get_heavy_node_config(model_name, batch_size, num_replicas, allocated_cpus, cpus_per_replica=2, allocated_gpus=[]):
    if model_name == INCEPTION_FEATS_MODEL_NAME:
        return tfs_utils.TFSHeavyNodeConfig(name=INCEPTION_FEATS_MODEL_NAME,
                                            model_base_path=INCEPTION_FEATS_MODEL_BASE_PATH,
                                            ports=INCEPTION_PORTS[:num_replicas],
                                            input_type="floats",
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            num_replicas=num_replicas,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size)

    elif model_name == LOG_REG_MODEL_NAME:
        return tfs_utils.TFSHeavyNodeConfig(name=LOG_REG_MODEL_NAME,
                                            model_base_path=LOG_REG_MODEL_BASE_PATH,
                                            ports=LOG_REG_PORTS[:num_replicas],
                                            input_type="floats",
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            num_replicas=num_replicas,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size)


    elif model_name == RESNET_152_MODEL_NAME:
        return tfs_utils.TFSHeavyNodeConfig(name=RESNET_152_MODEL_NAME,
                                            model_base_path=RESNET_152_MODEL_BASE_PATH,
                                            ports=RESNET_152_PORTS[:num_replicas],
                                            input_type="floats",
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            num_replicas=num_replicas,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size)

    elif model_name == KERNEL_SVM_MODEL_NAME:
        return tfs_utils.TFSHeavyNodeConfig(name=KERNEL_SVM_MODEL_NAME,
                                            model_base_path=KERNEL_SVM_MODEL_BASE_PATH,
                                            ports=KERNEL_SVM_PORTS[:num_replicas],
                                            input_type="floats",
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            num_replicas=num_replicas,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size)

########## Benchmarking ##########

class Predictor(object):

    def __init__(self, trial_length, clients):
        self.trial_length = trial_length
        self.outstanding_reqs = {}

        self.resnet_client = clients[RESNET_152_MODEL_NAME]
        self.svm_client = clients[KERNEL_SVM_MODEL_NAME]
        self.inception_client = clients[INCEPTION_FEATS_MODEL_NAME]
        self.log_reg_client = clients[LOG_REG_MODEL_NAME]

        self.init_stats()
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "all_lats": [],
            "mean_lats": []}
        self.total_num_complete = 0

    def init_stats(self):
        self.latencies = []
        self.batch_num_complete = 0
        self.cur_req_id = 0
        self.start_time = datetime.now()

    def print_stats(self):
        lats = np.array(self.latencies)
        p99 = np.percentile(lats, 99)
        mean = np.mean(lats)
        end_time = datetime.now()
        thru = float(self.batch_num_complete) / (end_time - self.start_time).total_seconds()
        self.stats["thrus"].append(thru)
        self.stats["p99_lats"].append(p99)
        self.stats["all_lats"].append(self.latencies)
        self.stats["mean_lats"].append(mean)
        logger.info("p99: {p99}, mean: {mean}, thruput: {thru}".format(p99=p99,
                                                                       mean=mean,
                                                                       thru=thru))

    def predict(self, resnet_input, inception_input):
        begin_time = datetime.now()
        classifications_lock = Lock()
        classifications = {}

        def update_perf_stats():
            end_time = datetime.now()
            latency = (end_time - begin_time).total_seconds()
            self.latencies.append(latency)
            self.total_num_complete += 1
            self.batch_num_complete += 1
            if self.batch_num_complete % self.trial_length == 0:
                self.print_stats()
                self.init_stats()

        def resnet_feats_continuation(resnet_response):
            resnet_features = tfs_utils.parse_predict_response(resnet_response, RESNET_FEATS_OUTPUT_KEY)
            # The SVM expects reduced dimensionality
            resnet_features = resnet_features[0]
            request = tfs_utils.create_predict_request(KERNEL_SVM_MODEL_NAME, resnet_features)
            self.svm_client.predict(request, svm_continuation)

        def svm_continuation(svm_response):
            svm_classification = tfs_utils.parse_predict_response(svm_response, KERNEL_SVM_OUTPUT_KEY)
            classifications_lock.acquire()
            if LOG_REG_MODEL_NAME not in classifications:
                classifications[KERNEL_SVM_MODEL_NAME] = svm_classification
            else:
                update_perf_stats()
            classifications_lock.release()

        def inception_feats_continuation(inception_response):
            inception_features = tfs_utils.parse_predict_response(inception_response, INCEPTION_FEATS_OUTPUT_KEY)
            request = tfs_utils.create_predict_request(LOG_REG_MODEL_NAME, inception_features)
            self.log_reg_client.predict(request, log_reg_continuation)

        def log_reg_continuation(log_reg_response):
            log_reg_vals = tfs_utils.parse_predict_response(log_reg_response, LOG_REG_OUTPUT_KEY)
            classifications_lock.acquire()
            if KERNEL_SVM_MODEL_NAME not in classifications:
                classifications[LOG_REG_MODEL_NAME] = log_reg_vals
            else:
                update_perf_stats()
            classifications_lock.release()


        resnet_request = tfs_utils.create_predict_request(RESNET_152_MODEL_NAME, resnet_input)
        self.resnet_client.predict(resnet_request, resnet_feats_continuation)

        inception_request = tfs_utils.create_predict_request(INCEPTION_FEATS_MODEL_NAME, inception_input)
        self.inception_client.predict(inception_request, inception_feats_continuation)

    def _get_resnet_request(self, resnet_input):
        """
        Parameters
        ------------
        resnet_input : np.ndarray
            A numpy array of type and structure compatible
            with the TF ResNet model

        Returns
        ------------
        predict_pb2.PredictRequest
        """

        return tfs_utils.create_predict_request(model_name=RESNET_152_MODEL_NAME,
                                                data=resnet_input)

    def _get_svm_request(self, svm_input):
        """
        Parameters
        ------------
        resnet_input : np.ndarray
            A numpy array of type and structure compatible
            with the TF kernel SVM

        Returns
        ------------
        predict_pb2.PredictRequest
        """

        return tfs_utils.create_predict_request(model_name=KERNEL_SVM_MODEL_NAME,
                                                data=svm_input)

    def _get_inception_request(self, inception_input):
        """
        Parameters
        ------------
        resnet_input : np.ndarray
            A numpy array of type and structure compatible
            with the Inception model

        Returns
        ------------
        predict_pb2.PredictRequest
        """

        return tfs_utils.create_predict_request(model_name=INCEPTION_FEATS_MODEL_NAME,
                                                data=inception_input)


    def _get_log_reg_request(self, log_reg_input):
        """
        Parameters
        ------------
        log_reg_input : np.ndarray
            A numpy array of type and structure compatible
            with the TF logistic regression model

        Returns
        ------------
        predict_pb2.PredictRequest
        """

        return tfs_utils.create_predict_request(model_name=LOG_REG_MODEL_NAME,
                                                data=log_reg_input)

class DriverBenchmarker(object):
    def __init__(self, trial_length, queue, configs):
        self.trial_length = trial_length
        self.queue = queue
        self.configs = configs

    def run(self, num_trials, request_delay=.01):
        logger.info("Creating clients!")
        clients = create_clients(self.configs)

        logger.info("Generating random inputs")
        base_inputs = [(self._get_resnet_input(), self._get_inception_input()) for _ in range(1000)]
        inputs = [i for _ in range(40) for i in base_inputs]
        logger.info("Starting predictions")
        start_time = datetime.now()
        predictor = Predictor(trial_length=self.trial_length, clients=clients)
        for resnet_input, inception_input in inputs:
            predictor.predict(resnet_input, inception_input)
            time.sleep(request_delay)

            if len(predictor.stats["thrus"]) > num_trials:
                break

        self.queue.put(predictor.stats)

    def _get_resnet_input(self):
        resnet_input = np.array(np.random.rand(224, 224, 3) * 255, dtype=np.float32)
        return resnet_input

    def _get_inception_input(self):
        inception_input = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
        return inception_input

class RequestDelayConfig:
    def __init__(self, request_delay):
        self.request_delay = request_delay
        
    def to_json(self):
        return json.dumps(self.__dict__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Clipper image driver 1')
    parser.add_argument('-t', '--num_trials', type=int, default=30, help='The number of trials to complete for the benchmarking process')
    parser.add_argument('-b', '--batch_sizes', type=int, nargs='+', help="The batch size configurations to benchmark for the model. Each configuration will be benchmarked separately.")
    parser.add_argument('-c', '--model_cpus', type=int, nargs='+', help="The set of cpu cores on which to run replicas of the provided model")
    parser.add_argument('-rd', '--request_delay', type=float, default=.015, help="The delay, in seconds, between requests")
    parser.add_argument('-l', '--trial_length', type=int, default=10, help="The length of each trial, in number of requests")
    parser.add_argument('-n', '--num_clients', type=int, default=1, help='number of clients')

    args = parser.parse_args()

    resnet_feats_config = get_heavy_node_config(model_name=RESNET_152_MODEL_NAME,
                                                batch_size=64,
                                                num_replicas=1,
                                                cpus_per_replica=2,
                                                allocated_cpus=[14,15,16,17],
                                                allocated_gpus=[0,1,2,3])

    kernel_svm_config = get_heavy_node_config(model_name=KERNEL_SVM_MODEL_NAME,
                                              batch_size=32,
                                              num_replicas=1,
                                              cpus_per_replica=2,
                                              allocated_cpus=[18,19])

    inception_feats_config = get_heavy_node_config(model_name=INCEPTION_FEATS_MODEL_NAME, 
                                                   batch_size=20, 
                                                   num_replicas=1, 
                                                   cpus_per_replica=2, 
                                                   allocated_cpus=[20,21,22,23], 
                                                   allocated_gpus=[4,5,6,7])

    log_reg_config = get_heavy_node_config(model_name=LOG_REG_MODEL_NAME,
                                           batch_size=1,
                                           num_replicas=1,
                                           cpus_per_replica=2,
                                           allocated_cpus=[24,25])

    model_configs = {
        RESNET_152_MODEL_NAME : resnet_feats_config,
        KERNEL_SVM_MODEL_NAME : kernel_svm_config,
        INCEPTION_FEATS_MODEL_NAME : inception_feats_config,
        LOG_REG_MODEL_NAME : log_reg_config
    }

    # Set up TFS nodes
    setup_heavy_nodes(model_configs)

    queue = Queue()

    procs = []
    for i in range(args.num_clients):
        clipper_metrics = (i == 0)
        benchmarker = DriverBenchmarker(args.trial_length, queue, model_configs)
        p = Process(target=benchmarker.run, args=(args.num_trials, args.request_delay))
        p.start()
        procs.append(p)

    all_stats = []
    for i in range(args.num_clients):
        all_stats.append(queue.get())

    # Save Results

    output_config = RequestDelayConfig(args.request_delay)
    all_configs = model_configs.values() + [output_config]

    fname = "{clients}_clients".format(clients=args.num_clients)
    tfs_utils.save_results(all_configs, all_stats, "tf_image_driver_1_exps", prefix=fname)
    sys.exit(0)