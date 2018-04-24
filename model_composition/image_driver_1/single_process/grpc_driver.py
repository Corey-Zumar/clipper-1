import sys
import os
import argparse
import numpy as np
import logging
import Queue
import time
import json

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock

from machine_config_tagger import TAGGED_CONFIG_KEY_MACHINE_ADDRESS, TAGGED_CONFIG_KEY_CONFIG_PATH
from single_proc_utils import HeavyNodeConfig, save_results
from single_proc_utils.spd_grpc_utils.spd_client import ReplicaAddress, SPDClient  

from e2e_utils import load_tagged_arrival_deltas, load_arrival_deltas, calculate_mean_throughput

INCEPTION_FEATS_MODEL_NAME = "inception_feats"
TF_KERNEL_SVM_MODEL_NAME = "kernel_svm"
TF_LOG_REG_MODEL_NAME = "tf_log_reg"
TF_RESNET_MODEL_NAME = "tf_resnet_feats"

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

CONFIG_KEY_BATCH_SIZE = "batch_size"
CONFIG_KEY_CPU_AFFINITIES = "cpu_affinities"
CONFIG_KEY_GPU_AFFINITIES = "gpu_affinities"
CONFIG_KEY_PROCESS_PATH = "process_path"
CONFIG_KEY_REPLICA_NUMS = "replica_nums"
CONFIG_KEY_TRIAL_LENGTH = "trial_length"
CONFIG_KEY_NUM_TRIALS = "num_trials"
CONFIG_KEY_SLO_MILLIS = "slo_millis"

########## Setup ##########

def get_heavy_node_configs(num_replicas, batch_size, allocated_cpus, allocated_gpus):
    resnet_config = HeavyNodeConfig(model_name=TF_RESNET_MODEL_NAME,
                                    input_type="floats",
                                    num_replicas=num_replicas,
                                    allocated_cpus=allocated_cpus,
                                    gpus=allocated_gpus,
                                    batch_size=batch_size)

    inception_config = HeavyNodeConfig(model_name=INCEPTION_FEATS_MODEL_NAME,
                                       input_type="floats",
                                       num_replicas=num_replicas,
                                       allocated_cpus=allocated_cpus,
                                       gpus=allocated_gpus,
                                       batch_size=batch_size)

    kernel_svm_config = HeavyNodeConfig(model_name=TF_KERNEL_SVM_MODEL_NAME,
                                        input_type="floats",
                                        num_replicas=num_replicas,
                                        allocated_cpus=allocated_cpus,
                                        gpus=[],
                                        batch_size=batch_size)

    log_reg_config = HeavyNodeConfig(model_name=TF_LOG_REG_MODEL_NAME,
                                     input_type="floats",
                                     num_replicas=num_replicas,
                                     allocated_cpus=allocated_cpus,
                                     gpus=[],
                                     batch_size=batch_size)

    return [resnet_config, inception_config, kernel_svm_config, log_reg_config]

class ExperimentConfig:

    def __init__(self, machine_addrs, batch_size, process_path, trial_length, num_trials, slo_millis):
        self.machine_addrs = machine_addrs
        self.batch_size = batch_size
        self.process_path = process_path
        self.trial_length = trial_length
        self.num_trials = num_trials
        self.slo_millis = slo_millis

def parse_configs(tagged_config_path):
    with open(tagged_config_path, "r") as f:
        tagged_config_json = json.load(f)

    config_files = [item[TAGGED_CONFIG_KEY_CONFIG_PATH] for item in tagged_config_json]
    machine_addrs = [item[TAGGED_CONFIG_KEY_MACHINE_ADDRESS] for item in tagged_config_json]

    all_replica_nums = []
    all_cpu_affinities = []
    all_gpu_affinities = []

    machine_num = 0
    for config_file in config_files:
        with open(config_file, "r") as f:
            config = json.load(f)

        batch_size = config[CONFIG_KEY_BATCH_SIZE]
        process_path = config[CONFIG_KEY_PROCESS_PATH]
        trial_length = config[CONFIG_KEY_TRIAL_LENGTH]
        num_trials = config[CONFIG_KEY_NUM_TRIALS]
        slo_millis = config[CONFIG_KEY_SLO_MILLIS]
        replica_nums = config[CONFIG_KEY_REPLICA_NUMS]
        cpu_affinities = config[CONFIG_KEY_CPU_AFFINITIES]
        gpu_affinities = config[CONFIG_KEY_GPU_AFFINITIES]

        tagged_cpu_affinities = []
        for affinities_item in cpu_affinities:
            tagged_item = " ".join(["m{mn}:{itm}".format(mn=machine_num, itm=item) for item in affinities_item.split(" ")])
            tagged_cpu_affinities.append(tagged_item)

        tagged_gpu_affinities = []
        for affinities_item in gpu_affinities:
            tagged_item = " ".join(["m{mn}:{itm}".format(mn=machine_num, itm=item) for item in affinities_item.split(" ")])
            tagged_gpu_affinities.append(tagged_item)

        all_replica_nums += replica_nums
        all_cpu_affinities += tagged_cpu_affinities
        all_gpu_affinities += tagged_gpu_affinities

        machine_num += 1

    experiment_config = ExperimentConfig(machine_addrs, 
                                         batch_size, 
                                         process_path, 
                                         trial_length, 
                                         num_trials, 
                                         slo_millis)

    node_configs = get_heavy_node_configs(num_replicas=len(all_replica_nums), 
                                     batch_size=batch_size, 
                                     allocated_cpus=all_cpu_affinities, 
                                     allocated_gpus=all_gpu_affinities)

    return experiment_config, node_configs

########## Input Generation ##########

def generate_inputs():
    inception_inputs = [_get_inception_input() for _ in range(1000)]
    inception_inputs = [i for _ in range(40) for i in inception_inputs]

    return np.array(inception_inputs, dtype=np.float32)

def _get_inception_input():
    inception_input = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
    return inception_input.flatten()

########## Benchmarking ##########

class StatsManager(object):

    def __init__(self, trial_length):
        self.stats_thread_pool = ThreadPoolExecutor(max_workers=2)

        self._init_stats()
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "mean_lats": [],
            "all_lats": [],
            "p99_queue_lats": [],
            "mean_batch_sizes": [],
            "per_message_lats": {}
        }
        self.total_num_complete = 0
        self.trial_length = trial_length

        self.start_timestamp = datetime.now()

    def update_stats(self, completed_requests, end_time):
        try:
            batch_size = len(completed_requests)
            self.batch_sizes.append(batch_size)
            for msg_id, send_time in completed_requests:
                e2e_latency = (end_time - send_time).total_seconds()
                self.latencies.append(e2e_latency)
                self.stats["per_message_lats"][msg_id] = e2e_latency

            self.trial_num_complete += batch_size

            if self.trial_num_complete >= self.trial_length:
                self._print_stats()
                self._init_stats()
        except Exception as e:
            print(e)

    def _init_stats(self):
        self.latencies = []
        self.batch_sizes = []
        self.trial_num_complete = 0
        self.cur_req_id = 0
        self.start_time = datetime.now()

    def _print_stats(self):
        end_time = datetime.now()
        thru = float(self.trial_num_complete) / (end_time - self.start_time).total_seconds()
        self.start_time = end_time

        lats = np.array(self.latencies)
        p99 = np.percentile(lats, 99)
        mean_batch_size = np.mean(self.batch_sizes)
        mean = np.mean(lats)
        self.stats["thrus"].append(thru)
        self.stats["all_lats"].append(self.latencies)
        self.stats["p99_lats"].append(p99)
        self.stats["mean_lats"].append(mean)
        self.stats["mean_batch_sizes"].append(mean_batch_size)
        logger.info("p99_lat: {p99}, mean_lat: {mean}, thruput: {thru}, " 
                    "mean_batch: {mb}".format(p99=p99,
                                              mean=mean,
                                              thru=thru, 
                                              mb=mean_batch_size))

class DriverBenchmarker:
    def __init__(self, experiment_config):
        self.experiment_config = experiment_config
        self.spd_client = self._create_client(experiment_config.machine_addrs)
    
    def run_config(self):
        pass
    
    def run_fixed_batch(self, batch_size):
        self.spd_client.start()
        
        num_trials = 30
        trial_length = batch_size * 5
       
        logger.info("Generating inputs...")
        inputs = generate_inputs()

        stats_manager = StatsManager(trial_length)

        inflight_ids_lock = Lock()
        inflight_ids = {}

        def callback(replica_num, msg_ids):
            try:
                end_time = datetime.now()
                inflight_ids_lock.acquire()
                completed_msgs = []
                for msg_id in msg_ids:
                    send_time = inflight_ids[msg_id]
                    completed_msgs.append((msg_id, send_time))
                    del inflight_ids[msg_id]
                inflight_ids_lock.release()

                stats_manager.update_stats(completed_msgs, end_time)
            except Exception as e:
                print(e)

        logger.info("Starting predictions...")

        last_msg_id = 0
        while True:
            batch_idxs = np.random.randint(0, len(inputs), batch_size)
            batch_inputs = inputs[batch_idxs]
            batch_msg_ids = range(last_msg_id, last_msg_id + batch_size)
            last_msg_id = batch_msg_ids[0] + batch_size
            
            inflight_ids_lock.acquire()
            send_time = datetime.now()
            for msg_id in batch_msg_ids:
                inflight_ids[msg_id] = send_time
            inflight_ids_lock.release()

            batch_inputs = [np.array(np.random.rand(299 * 299 * 3 * 60), dtype=np.float32)]

            self.spd_client.predict(batch_inputs, batch_msg_ids, callback)

            if len(stats_manager.stats["thrus"]) > num_trials:
                save_results(self.configs, 
                             [stats_manager.stats], 
                             "single_proc_bs_{}_bench".format(batch_size), 
                             slo_millis) 

                self.spd_client.stop()

            time.sleep(5)


    def _create_client(self, machine_addrs):
        replica_addrs = []
        for machine_addr in machine_addrs:
            host_name, port = machine_addr.split(":")
            replica_addr = ReplicaAddress(host_name, port)
            replica_addrs.append(replica_addr)
        
        return SPDClient(replica_addrs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Single Process Image Driver 1')
    parser.add_argument('-tc',  '--tagged_config_path', type=str, help="Path to the machine-tagged benchmark config")
    parser.add_argument('-b',   '--fixed_batch_size', type=int, help="(Optional) The fixed batch size to use for profiling, instead of the specified arrival process")

    args = parser.parse_args()


    experiment_config, node_configs = parse_configs(args.tagged_config_path)
    benchmarker = DriverBenchmarker(experiment_config)

    if args.fixed_batch_size:
        benchmarker.run_fixed_batch(args.fixed_batch_size)
    else:
        benchmarker.run_config()
