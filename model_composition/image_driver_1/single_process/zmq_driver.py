import sys
import os
import argparse
import numpy as np
import logging
import Queue
import time
import json
import copy

from datetime import datetime
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock

from single_proc_utils import HeavyNodeConfig, save_results
from single_proc_utils.spd_zmq_utils import ReplicaAddress, SPDClient, QUEUE_RATE_MEASUREMENT_WINDOW_SECONDS 

from single_proc_utils.config_utils import load_tagged_arrival_deltas, load_arrival_deltas, calculate_mean_throughput
from single_proc_utils.config_utils import TAGGED_CONFIG_KEY_MACHINE_ADDRESS, TAGGED_CONFIG_KEY_CONFIG_PATH
from single_proc_utils.config_utils import CONFIG_KEY_BATCH_SIZE, CONFIG_KEY_CPU_AFFINITIES, CONFIG_KEY_GPU_AFFINITIES
from single_proc_utils.config_utils import CONFIG_KEY_PROCESS_PATH, CONFIG_KEY_REPLICA_NUMS, CONFIG_KEY_TRIAL_LENGTH
from single_proc_utils.config_utils import CONFIG_KEY_NUM_TRIALS, CONFIG_KEY_SLO_MILLIS, CONFIG_KEY_LAMBDA, CONFIG_KEY_CV

TF_INCEPTION_FEATS_MODEL_NAME = "inception_feats"
TF_KERNEL_SVM_MODEL_NAME = "kernel_svm"
TF_LOG_REG_MODEL_NAME = "tf_log_reg"
TF_RESNET_FEATS_MODEL_NAME = "tf_resnet_feats"

TF_ALEXNET_MODEL_NAME = "tf_alexnet_feats"

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

EXPIRED_REQUEST_LATENCY = sys.maxint
SERVICE_INGEST_RATIO_DIVERGENCE_THRESHOLD = .95

########## Setup ##########

def get_heavy_node_configs(num_replicas, batch_size, allocated_cpus, allocated_gpus):
    # resnet_config = HeavyNodeConfig(model_name=TF_RESNET_FEATS_MODEL_NAME,
    #                                 input_type="floats",
    #                                 num_replicas=num_replicas,
    #                                 allocated_cpus=allocated_cpus,
    #                                 gpus=allocated_gpus,
    #                                 batch_size=batch_size)

    alexnet_config = HeavyNodeConfig(model_name=TF_ALEXNET_FEATS_MODEL_NAME,
                                     input_type="floats",
                                     num_replicas=num_replicas,
                                     allocated_cpus=allocated_cpus,
                                     gpus=allocated_gpus,
                                     batch_size=batch_size)

    inception_config = HeavyNodeConfig(model_name=TF_INCEPTION_FEATS_MODEL_NAME,
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

    # return [resnet_config, inception_config, kernel_svm_config, log_reg_config]
    return [alexnet_config, inception_config, kernel_svm_config, log_reg_config]

class ExperimentConfig:

    def __init__(self, config_path, machine_addrs, batch_size, process_path, trial_length, num_trials, slo_millis, lambda_val, cv):
        self.config_path = config_path
        self.machine_addrs = machine_addrs
        self.batch_size = batch_size
        self.process_path = process_path
        self.trial_length = trial_length
        self.num_trials = num_trials
        self.slo_millis = slo_millis
        self.lambda_val = lambda_val
        self.cv = cv

def parse_configs(tagged_config_path):
    with open(tagged_config_path, "r") as f:
        tagged_config_json = json.load(f)

    config_files = set([item[TAGGED_CONFIG_KEY_CONFIG_PATH] for item in tagged_config_json])
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
        cv = config[CONFIG_KEY_CV]
        lambda_val = config[CONFIG_KEY_LAMBDA]

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

    experiment_config = ExperimentConfig(tagged_config_path,
                                         machine_addrs, 
                                         batch_size, 
                                         process_path, 
                                         trial_length, 
                                         num_trials, 
                                         slo_millis,
                                         lambda_val,
                                         cv)

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
        self.stats_lock = Lock()
        self.start_timestamp = datetime.now()

    def get_mean_thru_for_dequeue(self):
        self.stats_lock.acquire()
        mean_thru = np.mean(self.stats["thrus"][2:-1])
        self.stats_lock.release()
        return mean_thru

    def get_stats(self):
        self.stats_lock.acquire()
        result = copy.deepcopy(self.stats)
        self.stats_lock.release()
        return result

    def update_stats(self, completed_requests, end_time):
        try:
            self.stats_lock.acquire()
            batch_size = len(completed_requests)
            self.batch_sizes.append(batch_size)
            for msg_id, send_time in completed_requests:
                e2e_latency = (end_time - send_time).total_seconds()
                self.latencies.append(e2e_latency)
                self.stats["per_message_lats"][str(msg_id)] = e2e_latency

            self.trial_num_complete += batch_size

            if self.trial_num_complete >= self.trial_length:
                self._print_stats()
                self._init_stats()

            self.stats_lock.release()
        except Exception as e:
            print("ERROR UPDATING STATS: {}".format(e))
            os._exit(1)

    def expire_requests(self, msg_ids):
        try:
            self.stats_lock.acquire()
            for msg_id in msg_ids:
                self.stats["per_message_lats"][str(msg_id)] = EXPIRED_REQUEST_LATENCY
            self.stats_lock.release()
        except Exception as e:
            print("ERROR EXPIRING REQUESTS {}".format(e))
            os._exit(1)

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
    def __init__(self, node_configs, experiment_config):
        self.node_configs = node_configs
        self.experiment_config = experiment_config
        self.spd_client = self._create_client(experiment_config.machine_addrs)
    
    def run(self, fixed_batch_size=None):
        def stats_update_callback(replica_num, msg_ids):
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
                print("ERROR IN STATS UPDATE CALLBACK: {}".format(e))
                os._exit(1)

        def expiration_callback(msg_ids):
            try:
                inflight_ids_lock.acquire()
                for msg_id in msg_ids:
                    del inflight_ids[msg_id]
                inflight_ids_lock.release()

                stats_manager.expire_requests(msg_ids)
            
            except Exception as e:
                print("ERROR IN EXPIRATION CALLBACK: {}".format(e))
                os._exit(1)

        logger.info("Loading arrival process...")
        arrival_process = load_arrival_deltas(self.experiment_config.process_path)

        stats_manager = StatsManager(self.experiment_config.trial_length)

        inflight_ids = {}
        inflight_ids_lock = Lock()

        logger.info("Starting predictions...")

        diverged = False

        if fixed_batch_size:
            batch_size = fixed_batch_size
        else:
            batch_size = experiment_config.batch_size

        self.spd_client.start((fixed_batch_size is not None),
                              batch_size, 
                              self.experiment_config.slo_millis,
                              stats_update_callback,
                              expiration_callback,
                              inflight_ids,
                              inflight_ids_lock,
                              arrival_process)

        while True:
            if len(stats_manager.stats["per_message_lats"]) < .98 * len(arrival_process):
                time.sleep(QUEUE_RATE_MEASUREMENT_WINDOW_SECONDS * 2)
            else:
                break

            if len(stats_manager.stats["thrus"]) < 6:
                continue
            
            enqueue_rate = self.spd_client.get_enqueue_rate()
            dequeue_rate = stats_manager.get_mean_thru_for_dequeue()

            # dequeue_rate = self.spd_client.get_dequeue_rate()
            #
            print(enqueue_rate, dequeue_rate)

            if dequeue_rate == 0 and enqueue_rate > 0:
                logger.info("ERROR: Dequeue rate is zero, yet enqueue rate is: {}".format(enqueue_rate))

            elif (not diverged) and enqueue_rate > 0 and dequeue_rate > 0 and (dequeue_rate / enqueue_rate) <= SERVICE_INGEST_RATIO_DIVERGENCE_THRESHOLD:
                diverged = True
                logger.info("Request queue is diverging! Stopping experiment...")
                logger.info("Enqueue rate: {}, Dequeue rate: {}".format(enqueue_rate, dequeue_rate))
                # break

        time.sleep(20)

        results_base_path = "/".join(experiment_config.config_path.split("/")[:-1])
        save_results(self.experiment_config,
                     self.node_configs, 
                     [stats_manager.get_stats()],
                     results_base_path,
                     experiment_config.slo_millis,
                     diverged,
                     arrival_process=experiment_config.process_path)

        os._exit(0)

        self.spd_client.stop()
    
    def _create_client(self, machine_addrs):
        replica_addrs = []
        for machine_addr in machine_addrs:
            host_name, port = machine_addr.split(":")
            replica_addr = ReplicaAddress(host_name, port)
            replica_addrs.append(replica_addr)
        
        return SPDClient(replica_addrs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark SPD configurations')
    parser.add_argument('-tc',  '--tagged_config_path', type=str, help="Path to the machine-tagged benchmark config")
    parser.add_argument('-b',   '--fixed_batch_size', type=int, help="(Optional) The fixed batch size to use for profiling, instead of the specified arrival process")

    args = parser.parse_args()


    experiment_config, node_configs = parse_configs(args.tagged_config_path)
    benchmarker = DriverBenchmarker(node_configs, experiment_config)

    benchmarker.run(args.fixed_batch_size)
