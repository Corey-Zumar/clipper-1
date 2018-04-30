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

from tf_serving_utils.config_utils import CONFIG_KEY_MODEL_NAME, CONFIG_KEY_BATCH_SIZE, CONFIG_KEY_NUM_REPLICAS, CONFIG_KEY_VCPUS_PER_REPLICA 
from tf_serving_utils.config_utils import CONFIG_KEY_ALLOCATED_GPUS, CONFIG_KEY_ALLOCATED_VCPUS, CONFIG_KEY_ALLOCATED_GPUS, CONFIG_KEY_PORTS 

# Models and applications for each heavy node
# will share the same name
INCEPTION_FEATS_MODEL_NAME = "inception"
RESNET_152_MODEL_NAME = "resnet"
LOG_REG_MODEL_NAME = "log_reg"
KERNEL_SVM_MODEL_NAME = "ksvm"

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_BASE_DIR_PATH = os.path.join(CURR_DIR, "exported_tf_models")

INCEPTION_FEATS_MODEL_BASE_PATH = os.path.join(MODEL_BASE_DIR_PATH, "inception_tfserve")
RESNET_152_MODEL_BASE_PATH = os.path.join(MODEL_BASE_DIR_PATH, "resnet_tfserve")
LOG_REG_MODEL_BASE_PATH = os.path.join(MODEL_BASE_DIR_PATH, "log_reg_tfserve")
KERNEL_SVM_MODEL_BASE_PATH = os.path.join(MODEL_BASE_DIR_PATH, "kernel_svm_tfserve")

def get_heavy_node_config(model_name, batch_size, num_replicas, ports, allocated_vcpus, vcpus_per_replica=2, allocated_gpus=[]):
    if model_name == INCEPTION_FEATS_MODEL_NAME:
        return tfs_utils.TFSHeavyNodeConfig(name=INCEPTION_FEATS_MODEL_NAME,
                                            model_base_path=INCEPTION_FEATS_MODEL_BASE_PATH,
                                            ports=ports,
                                            input_type="floats",
                                            allocated_cpus=allocated_vcpus,
                                            cpus_per_replica=vcpus_per_replica,
                                            num_replicas=num_replicas,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size)

    elif model_name == LOG_REG_MODEL_NAME:
        return tfs_utils.TFSHeavyNodeConfig(name=LOG_REG_MODEL_NAME,
                                            model_base_path=LOG_REG_MODEL_BASE_PATH,
                                            ports=ports,
                                            input_type="floats",
                                            allocated_cpus=allocated_vcpus,
                                            cpus_per_replica=vcpus_per_replica,
                                            num_replicas=num_replicas,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size)


    elif model_name == RESNET_152_MODEL_NAME:
        return tfs_utils.TFSHeavyNodeConfig(name=RESNET_152_MODEL_NAME,
                                            model_base_path=RESNET_152_MODEL_BASE_PATH,
                                            ports=ports,
                                            input_type="floats",
                                            allocated_cpus=allocated_vcpus,
                                            cpus_per_replica=vcpus_per_replica,
                                            num_replicas=num_replicas,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size)

    elif model_name == KERNEL_SVM_MODEL_NAME:
        return tfs_utils.TFSHeavyNodeConfig(name=KERNEL_SVM_MODEL_NAME,
                                            model_base_path=KERNEL_SVM_MODEL_BASE_PATH,
                                            ports=ports,
                                            input_type="floats",
                                            allocated_cpus=allocated_vcpus,
                                            cpus_per_replica=vcpus_per_replica,
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
        batch_size = config_params[CONFIG_KEY_BATCH_SIZE]
        num_replicas = config_params[CONFIG_KEY_NUM_REPLICAS]
        vcpus_per_replica = config_params[CONFIG_KEY_VCPUS_PER_REPLICA]
        allocated_vcpus = config_params[CONFIG_KEY_ALLOCATED_VCPUS]
        allocated_gpus = config_params[CONFIG_KEY_ALLOCATED_GPUS]
        ports = config_params[CONFIG_KEY_PORTS]

        config = get_heavy_node_config(model_name=model_name,
                                       batch_size=batch_size,
                                       num_replicas=num_replicas,
                                       ports=ports,
                                       vcpus_per_replica=vcpus_per_replica,
                                       allocated_vcpus=allocated_vcpus,
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
        ports = config_params[CONFIG_KEY_PORTS]

        server_configs.append((model_name, ports))

    return server_configs

