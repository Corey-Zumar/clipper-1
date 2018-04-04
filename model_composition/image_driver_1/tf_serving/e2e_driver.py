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

from e2e_configs import get_e2e_model_configs

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

TFS_ADDRESS = "localhost"

########## Client Setup ##########

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

    def run(self, num_trials, request_delay=.01, arrival_process=None):
        logger.info("Creating clients!")
        clients = create_clients(self.configs)

        logger.info("Generating random inputs")
        base_inputs = [(self._get_resnet_input(), self._get_inception_input()) for _ in range(1000)]
        inputs = [i for _ in range(40) for i in base_inputs]
        logger.info("Starting predictions")
        start_time = datetime.now()
        predictor = Predictor(trial_length=self.trial_length, clients=clients)

        for i in range(len(inputs)):
            resnet_input, inception_input = inputs[i]
            predictor.predict(resnet_input, inception_input)

            if arrival_process is not None:
                request_delay = arrival_process[i]

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
    parser.add_argument('-rd', '--request_delay', type=float, default=.015, help="The delay, in seconds, between requests")
    parser.add_argument('-l', '--trial_length', type=int, default=100, help="The length of each trial, in number of requests")
    parser.add_argument('-n', '--num_clients', type=int, default=16, help='number of clients')
    parser.add_argument('-p', '--process_file', type=str, help='The arrival process file path')


    args = parser.parse_args()

    model_configs = get_e2e_model_configs()

    queue = Queue()

    arrival_process = None
    if args.process_file:
        f = open(args.process_file)
        arrival_lines = f.readlines()
        f.close()
        arrival_lines = np.array([float(line.rstrip()) for line in arrival_lines])
        arrival_process = np.cumsum(arrival_lines)
       
        mean_throughput = (float(arrival_process[-1] - arrival_process[0]) / len(arrival_process))

        args.request_delay = 1.0 / mean_throughput 
        print("Based on mean arrival process throughput of {} qps, initialized request delay to {} seconds".format(mean_throughput, args.request_delay))


    procs = []
    for i in range(args.num_clients):
        benchmarker = DriverBenchmarker(args.trial_length, queue, model_configs)
        p = Process(target=benchmarker.run, args=(args.num_trials, args.request_delay, None))
        p.start()
        procs.append(p)

    all_stats = []
    for i in range(args.num_clients):
        all_stats.append(queue.get())

    # Save Results

    all_configs = model_configs.values()

    fname = "{clients}_clients".format(clients=args.num_clients)
    tfs_utils.save_results(all_configs, all_stats, "tf_image_driver_1_exps", prefix=fname, arrival_process=args.process_file)
    sys.exit(0)
