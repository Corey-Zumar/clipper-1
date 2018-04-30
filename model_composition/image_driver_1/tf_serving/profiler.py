import sys
import os
import argparse
import numpy as np
import time
import base64
import logging
import json

from threading import Lock
from datetime import datetime
from multiprocessing import Process, Queue

from tf_serving_utils import GRPCClient, ReplicaAddress
from tf_serving_utils import tfs_utils

from e2e_configs import load_client_configs, load_server_configs
from tf_serving_utils.config_utils import CONFIG_KEY_RESNET, CONFIG_KEY_INCEPTION, CONFIG_KEY_KSVM, CONFIG_KEY_LOG_REG

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

# Models and applications for each heavy node
# will share the same name
INCEPTION_FEATS_MODEL_NAME = CONFIG_KEY_INCEPTION
RESNET_152_MODEL_NAME = CONFIG_KEY_RESNET
LOG_REG_MODEL_NAME = CONFIG_KEY_LOG_REG
KERNEL_SVM_MODEL_NAME = CONFIG_KEY_KSVM

PROFILING_NUM_TRIALS = 20
PROFILING_REQUEST_DELAY_SECONDS = .001

def create_client(host_name, port):
    replica_addr = ReplicaAddress(host_name, port)
    replica_addrs = [replica_addr]
    client = GRPCClient(replica_addrs)
    client.start()
    return client

class Predictor(object):

    def __init__(self, model_name, trial_length, client):
        self.model_name = model_name
        self.trial_length = trial_length
        self.client = client

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

    def predict(self, input_data):
        begin_time = datetime.now()
        classifications_lock = Lock()
        classifications = {}

        def update_perf_stats():
            end_time = datetime.now()
            latency = (end_time - begin_time).total_seconds()
            self.latencies.append(latency)
            self.total_num_complete += 1
            self.batch_num_complete += 1
            if self.batch_num_complete >= self.trial_length:
                self.print_stats()
                self.init_stats()

        def continuation(response):
            update_perf_stats()

        request = tfs_utils.create_predict_request(self.model_name, input_data)
        self.client.predict(request, continuation)

class Profiler(object):
    def __init__(self, trial_length):
        self.trial_length = trial_length

    def run(self, model_name, host_name, port, num_trials):
        logger.info("Creating clients!")
        client = create_client(host_name, port)

        if model_name == INCEPTION_FEATS_MODEL_NAME:
            inputs_gen_fn = get_inception_input
        elif model_name == RESNET_152_MODEL_NAME:
            inputs_gen_fn = get_resnet_input
        elif model_name == KERNEL_SVM_MODEL_NAME:
            inputs_gen_fn = get_ksvm_input
        elif model_name == LOG_REG_MODEL_NAME:
            inputs_gen_fn = get_log_reg_input
        else:
            raise

        logger.info("Generating random inputs")
        base_inputs = [inputs_gen_fn() for _ in range(1000)]
        inputs = [i for _ in range(40) for i in base_inputs]
        logger.info("Starting predictions")
        predictor = Predictor(model_name=model_name, trial_length=self.trial_length, client=client)

        for i in range(len(inputs)):
            input_data = inputs[i]
            predictor.predict(input_data)

            # print(len(predictor.stats["thrus"]), num_trials)

            if len(predictor.stats["thrus"]) >= num_trials:
                break

            # time.sleep(PROFILING_REQUEST_DELAY_SECONDS)
            time.sleep(.0001)

        return predictor.stats

def get_resnet_input():
    resnet_input = np.array(np.random.rand(224, 224, 3) * 255, dtype=np.float32)
    return resnet_input

def get_inception_input():
    inception_input = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
    return inception_input

def get_ksvm_input():
    ksvm_input = np.array(np.random.rand(2048), dtype=np.float32)
    return ksvm_input 

def get_log_reg_input():
    log_reg_input = np.array(np.random.rand(2048), dtype=np.float32)
    return log_reg_input 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Profile models for Clipper image driver 1 to obtain throughput estimates')
    parser.add_argument('-hn', '--host', type=str, help="Hostname of the model to profile")
    parser.add_argument('-p', '--port', type=str, help="Port of the model to profile")
    parser.add_argument('-m', '--model', type=str, help="Name of the model to profile") 
    parser.add_argument('-t', '--trial_length', type=int, help="The length of each profiling trial")
    parser.add_argument('-o', '--output_path', type=str, help="The length of each profiling trial")

    args = parser.parse_args()
    
    profiler = Profiler(args.trial_length)
    stats = profiler.run(args.model, args.host, args.port, PROFILING_NUM_TRIALS)
    wrapped_stats = { "client_metrics" : [stats] } 

    with open(args.output_path, "w") as f:
        json.dump(wrapped_stats, f, indent=4)

    logger.info("Wrote profiler results to: {op}".format(op=args.output_path))

    os._exit(0)

