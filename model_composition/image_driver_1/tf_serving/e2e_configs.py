import sys
import os
import argparse
import numpy as np
import time
import base64
import logging
import json
import tensorflow as tf

from tf_serving_utils import GRPCClient, ReplicaAddress
from tf_serving_utils import tfs_utils

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

INCEPTION_PORTS = range(9500,9508)
RESNET_152_PORTS = range(9508, 9516)
LOG_REG_PORTS = range(9516, 9524)
KERNEL_SVM_PORTS = range(9524, 9532)

CONFIG_KEY_MODEL_NAME = "model_name"
CONFIG_KEY_NUM_REPLICAS = "num_replicas"
CONFIG_KEY_CPUS_PER_REPLICA = "cpus_per_replica"
CONFIG_KEY_ALLOCATED_CPUS = "allocated_cpus"
CONFIG_KEY_ALLOCATED_GPUS = "allocated_gpus"
CONFIG_KEY_HOST = "host"
CONFIG_KEY_PORTS = "ports"

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

def load_server_configs(configs_dir_path):
    config_paths = [os.path.join(configs_dir_path, path) for path in os.listdir(configs_dir_path) if ".json" in path]
    server_configs = []
    for path in config_paths:
        with open(path, "r") as f:
            config_params = json.load(f)

        model_name = config_params[CONFIG_KEY_MODEL_NAME]
        num_replicas = config_params[CONFIG_KEY_NUM_REPLICAS]
        cpus_per_replica = config_params[CONFIG_KEY_CPUS_PER_REPLICA]
        allocated_cpus = config_params[CONFIG_KEY_ALLOCATED_CPUS]
        allocated_gpus = config_params[CONFIG_KEY_ALLOCATED_GPUS]

        config = get_heavy_node_config(model_name=model_name,
                                       num_replicas=num_replicas,
                                       cpus_per_replica=cpus_per_replica,
                                       allocated_cpus=allocated_cpus,
                                       allocated_gpus=allocated_gpus)

        server_configs.append(config)

    return server_configs

def load_client_configs(configs_dir_path):
    config_paths = [os.path.join(configs_dir_path, path) for path in os.listdir(configs_dir_path) if ".json" in path]
    server_configs = []
    for path in config_paths:
        with open(path, "r") as f:
            config_params = json.load(f)

        model_name = config_params[CONFIG_KEY_MODEL_NAME]
        host = config_params[CONFIG_KEY_HOST]
        ports = config_params[CONFIG_KEY_PORTS]

        server_configs.append((model_name, host, ports))

    return server_configs

