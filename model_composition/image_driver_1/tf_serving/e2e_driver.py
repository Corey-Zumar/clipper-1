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

from tf_serving_utils.config_utils import CONFIG_KEY_NUM_TRIALS, CONFIG_KEY_TRIAL_LENGTH, CONFIG_KEY_NUM_CLIENTS, 
from tf_serving_utils.config_utils import CONFIG_KEY_SLO_MILLIS, CONFIG_KEY_PROCESS_PATH, CONFIG_KEY_PROCESS_HASH
from tf_serving_utils.config_utils import CONFIG_KEY_CV, CONFIG_KEY_LAMBDA

from tf_serving_utils.config_utils import TAGGED_CONFIG_KEY_TAGGED_MACHINES, TAGGED_CONFIG_KEY_EXPERIMENT_CONFIG
from tf_serving_utils.config_utils import TAGGED_CONFIG_KEY_MACHINE_ADDRESS, TAGGED_CONFIG_KEY_CONFIG_PATH 

from e2e_configs import load_client_configs, load_server_configs

CONFIG_KEY_CLIENT_CONFIG_PATHS = "client_config_paths"
CONFIG_KEY_CLIENT_CONFIG_ITEM_PATH = "config_path"
CONFIG_KEY_CLIENT_CONFIG_HOST = "host"

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

INCEPTION_FEATS_OUTPUT_KEY = "feats"
RESNET_FEATS_OUTPUT_KEY = "feats"
KERNEL_SVM_OUTPUT_KEY = "outputs"
LOG_REG_OUTPUT_KEY = "outputs"


########## Client Setup ##########

class ClientConfig:

    def __init__(self, model_name, host, port):
        self.model_name = model_name
        self.host = host
        self.port = port

class ExperimentConfig:

    def __init__(self, 
                 num_trials, 
                 trial_length, 
                 num_clients, 
                 slo_millis,
                 cv,
                 lambda_val,
                 process_path,
                 node_configs,
                 client_configs)
        self.num_trials = num_trials
        self.trial_length = trial_length
        self.num_clients = num_clients
        self.slo_millis = slo_millis
        self.cv = cv
        self.lambda_val = lambda_val
        self.process_path = process_path
        self.node_configs = node_configs 
        self.client_configs = client_configs

def load_experiment_config(config_path):
    with open(config_path, "r") as f:
        experiment_config_json = json.load(f)

    tagged_machines = experiment_config_json[TAGGED_CONFIG_KEY_TAGGED_MACHINES]
    experiment_config_params = experiment_config_json[TAGGED_CONFIG_KEY_EXPERIMENT_CONFIG]

    num_trials = experiment_config_params[CONFIG_KEY_NUM_TRIALS]
    trial_length = experiment_config_params[CONFIG_KEY_TRIAL_LENGTH]
    num_clients = experiment_config_params[CONFIG_KEY_NUM_CLIENTS]
    slo_millis = experiment_config_params[CONFIG_KEY_SLO_MILLIS]
    cv = experiment_config_params[CONFIG_KEY_CV]
    lambda_val = experiment_config_params[CONFIG_KEY_LAMBDA]
    process_path = experiment_config_params[CONFIG_KEY_PROCESS_PATH]

    client_model_configs = {}
    all_node_configs = []
    for tagged_machine in client_config_items:
        config_path = tagged_machine[CONFIG_KEY_CLIENT_CONFIG_ITEM_PATH]
        host = tagged_machine[CONFIG_KEY_CLIENT_CONFIG_HOST]

        client_configs = load_client_configs(config_path)
        for config in client_configs:
            model_name, ports = config
            required_replicas = experiment_config_params[model_name]
            
            if model_name not in client_model_configs:
                client_model_configs[model_name] = []

            while len(client_model_configs[model_name]) < required_replicas and len(ports) > 0:
                port = ports.pop(0)
                new_config = ClientConfig(model_name, host, port)
                client_model_configs[model_name].append(new_config)

        machine_node_configs = load_server_configs(config_path)
        all_node_configs.append(machine_node_configs)

    experiment_config = ExperimentConfig(num_trials=num_trials, 
                                         trial_length=trial_length, 
                                         num_clients=num_clients, 
                                         slo_millis=slo_millis,
                                         cv=cv,
                                         lambda_val=lambda_val,
                                         process_path=process_path,
                                         node_configs=all_node_configs,
                                         client_configs=client_model_configs)

    return experiment_config 

def create_clients(configs):
    """
    Parameters
    ------------
    configs : dict
        Dictionary of ClientConfig objects,
        keyed on model names
    """
    clients = {}
    for key in configs:
        replica_addrs = [ReplicaAddress(client_config.host, int(client_config.port)) for client_config in configs[key]]
        print(key, replica_addrs)
        client = GRPCClient(replica_addrs)
        client.start()
        clients[key] = client

    return clients

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
            inception_features = inception_features[0][0][0]
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

    def run(self, num_trials, arrival_process):
        logger.info("Creating clients!")
        clients = create_clients(self.configs)

        logger.info("Generating random inputs")
        base_inputs = [(self._get_resnet_input(), self._get_inception_input()) for _ in range(1000)]
        inputs = [i for _ in range(40) for i in base_inputs]
        logger.info("Starting predictions")
        predictor = Predictor(trial_length=self.trial_length, clients=clients)

        for i in range(len(inputs)):
            resnet_input, inception_input = inputs[i]
            predictor.predict(resnet_input, inception_input)

            if len(predictor.stats["thrus"]) >= num_trials:
                break

            request_delay = arrival_process[i] * .001

            time.sleep(request_delay)

        if self.queue:
            self.queue.put(predictor.stats)

    def warm_up(self):
        request_delay = .1
        clients = create_clients(self.configs)

        logger.info("Generating random inputs")
        base_inputs = [(self._get_resnet_input(), self._get_inception_input()) for _ in range(1000)]
        inputs = [i for _ in range(40) for i in base_inputs]
        logger.info("Starting predictions")
        predictor = Predictor(trial_length=self.trial_length, clients=clients)

        for i in range(len(inputs)):
            resnet_input, inception_input = inputs[i]
            predictor.predict(resnet_input, inception_input)
            time.sleep(request_delay)

    def _get_resnet_input(self):
        resnet_input = np.array(np.random.rand(224, 224, 3) * 255, dtype=np.float32)
        return resnet_input

    def _get_inception_input(self):
        inception_input = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
        return inception_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Clipper image driver 1')
    parser.add_argument('-w', '--warmup', action='store_true')
    parser.add_argument('-e', '--experiment_config_path', type=str)

    args = parser.parse_args()

    experiment_config = load_experiment_config(args.experiment_config_path)

    if args.warmup:
        benchmarker = DriverBenchmarker(trial_length=experiment_config.trial_length, 
                                        queue=None, 
                                        configs=experiment_config.client_configs)
        benchmarker.warm_up()

    else:
        queue = Queue()

        arrival_process = tfs_utils.load_arrival_deltas(experiment_config.process_path)
        mean_throughput = tfs_utils.calculate_mean_throughput(arrival_process)
        peak_throughput = tfs_utils.calculate_peak_throughput(arrival_process)

        print("Mean throughput: {}\nPeak throughput: {}".format(mean_throughput, peak_throughput))
            
        procs = []
        for i in range(experiment_config.num_clients):
            benchmarker = DriverBenchmarker(trial_length=experiment_config.trial_length, 
                                            queue=queue, 
                                            configs=experiment_config.client_configs)
            p = Process(target=benchmarker.run, args=(experiment_config.num_trials, arrival_process))
            p.start()
            procs.append(p)

        all_stats = []
        for i in range(experiment_config.num_clients):
            all_stats.append(queue.get())

        # Save Results
        results_dir_path = "/".join(experiment_config.process_path.split("/")[:-1])
        fname = "results_{slo}_slo_bs_1.json".format(slo=experiment_config.slo_millis)
        tfs_utils.save_results(node_configs=experiment_config.node_configs, 
                               client_metrics=all_stats,
                               prefix=fname,
                               results_dir=results_dir_path,
                               slo_millis=experiment_config.slo_millis, 
                               cv=experiment_config.cv,
                               lambda_val=experiment_config.lambda_val,
                               arrival_process=experiment_config.process_path)
        sys.exit(0)
