import sys
import os
import argparse
import numpy as np
import logging
import Queue
import time

from single_proc_utils import HeavyNodeConfig, save_results

from e2e_utils import load_tagged_arrival_deltas, load_arrival_deltas, calculate_mean_throughput

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

def get_heavy_node_configs(num_replicas, batch_size, allocated_cpus, resnet_gpus=[], inception_gpus=[]):
    resnet_config = HeavyNodeConfig(model_name=TF_RESNET_MODEL_NAME,
                                    input_type="floats",
                                    num_replicas=num_replicas,
                                    allocated_cpus=allocated_cpus,
                                    gpus=resnet_gpus,
                                    batch_size=batch_size)

    inception_config = HeavyNodeConfig(model_name=INCEPTION_FEATS_MODEL_NAME,
                                       input_type="floats",
                                       num_replicas=num_replicas,
                                       allocated_cpus=allocated_cpus,
                                       gpus=inception_gpus,
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

def parse_configs(configs_path):
    config_files = [os.path.join(configs_path, fname) for fname in os.listdir(configs_path) if "config" in fname]
    for config_file in config_files:
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Single Process Image Driver 1')
    parser.add_argument('-c',  '--configs_dir_path', type=str, help="Path to the benchmark config")

    # parser.add_argument('-n',  '--num_replicas', type=int, help="The number of SPD replicas to benchmark")
    # parser.add_argument('-b',  '--batch_size', type=int, help="The batch size to benchmark")
    # parser.add_argument('-c',  '--cpus', type=int, nargs='+', help="The set of machine-tagged VIRTUAL cpu cores on which to run SPD replicas")
    # parser.add_argument('-g',  '--gpus', type=int, nargs='+', help="The set of machine-tagged GPUs on which to run SPD replicas")
    # parser.add_argument('-t',  '--num_trials', type=int, default=15, help="The number of trials to run")
    # parser.add_argument('-tl', '--trial_length', type=int, default=200, help="The length of each trial, in requests")
    # parser.add_argument('-p',  '--process_file', type=str, help="Path to a TAGGED arrival process file")
    # parser.add_argument('-rd', '--request_delay', type=float, help="The request delay")
    # parser.add_argument('-s',  '--slo_millis', type=int, help="The latency SLO, in milliseconds")
    
    args = parser.parse_args()




