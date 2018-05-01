import sys
import os
import argparse
import numpy as np
import time
import base64
import logging
import json

from threading import Thread, Lock
from datetime import datetime
from multiprocessing import Process, Pipe
from concurrent.futures import ThreadPoolExecutor

from tf_serving_utils import GRPCClient, ReplicaAddress, REQUEST_PIPE_POLLING_TIMEOUT_SECONDS
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
PROFILING_REQUEST_DELAY_SECONDS = .0001

def create_client(model_name, host_name, port):
    def launch_client(model_name, replica_addrs, outbound_handle, inbound_handle):
        client = GRPCClient(model_name, replica_addrs, outbound_handle, inbound_handle)
        client.start()
        time.sleep(60 * 60 * 24)

    replica_addr = ReplicaAddress(host_name, port)
    replica_addrs = [replica_addr]

    outbound_parent_handle, outbound_child_handle = Pipe()
    inbound_parent_handle, inbound_child_handle = Pipe()

    client_proc = Process(target=launch_client, 
                          args=(model_name, 
                                replica_addrs, 
                                outbound_child_handle, 
                                inbound_child_handle))

    client_proc.start()

    return outbound_parent_handle, inbound_parent_handle
    
class Predictor(object):

    def __init__(self, model_name, trial_length, client):
        self.model_name = model_name
        self.trial_length = trial_length
        self.outbound_handle, self.inbound_handle = client

        self.continuation_executor = ThreadPoolExecutor(max_workers=16)

        self.inflight_requests_lock = Lock()
        self.inflight_requests = {}

        logger.info("Starting async response thread...")
        self.response_thread = Thread(target=self._run_response_service, args=(self.inbound_handle, model_name,))
        self.response_thread.start()

        self.init_stats()
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "all_lats": [],
            "mean_lats": []}
        self.total_num_complete = 0

        self.last_update_time = datetime.now()

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

    def predict(self, msg_id):
        self.inflight_requests_lock.acquire()
        send_time = datetime.now()
        self.inflight_requests[msg_id] = (send_time, {}, Lock())
        self.inflight_requests_lock.release()

        self.outbound_handle.send(msg_id)

    def _update_perf_stats(self, msg_id):
        end_time = datetime.now()
        self.last_update_time = end_time

        self.inflight_requests_lock.acquire()
        begin_time = self.inflight_requests[msg_id][0]
        self.inflight_requests_lock.release()
        latency = (end_time - begin_time).total_seconds()
        self.latencies.append(latency)
        self.total_num_complete += 1
        self.batch_num_complete += 1
        if self.batch_num_complete % self.trial_length == 0:
            self.print_stats()
            self.init_stats()

    def _run_response_service(self, inbound_handle, model_name):
        try:
            while True:
                data_available = inbound_handle.poll(REQUEST_PIPE_POLLING_TIMEOUT_SECONDS)
                if not data_available:
                    continue
           
                inbound_msg_ids = [inbound_handle.recv()]
                while inbound_handle.poll(0):
                    inbound_msg_id = inbound_handle.recv()
                    inbound_msg_ids.append(inbound_msg_id)
                
                for msg_id in inbound_msg_ids:
                    self.continuation_executor.submit(self._update_perf_stats, msg_id)

        except Exception as e:
            print("Error in response service: {}".format(e))

class Profiler(object):
    def __init__(self, trial_length):
        self.trial_length = trial_length

    def run(self, model_name, host_name, port, num_trials):
        logger.info("Creating clients!")
        client = create_client(model_name, host_name, port)

        predictor = Predictor(model_name=model_name, 
                              trial_length=self.trial_length, 
                              client=client)

        time.sleep(60)

        logger.info("Starting predictions")

        for msg_id in range(10000000):
            predictor.predict(msg_id)

            if len(predictor.stats["thrus"]) >= num_trials:
                break

            time.sleep(PROFILING_REQUEST_DELAY_SECONDS)

        return predictor.stats

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

