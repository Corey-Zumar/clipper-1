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

def get_setup_model_configs():
    resnet_feats_config = get_heavy_node_config(model_name=RESNET_152_MODEL_NAME,
                                                batch_size=1,
                                                num_replicas=5,
                                                cpus_per_replica=2,
                                                allocated_cpus=[0,16,1,17,2,18,3,19,4,20],
                                                allocated_gpus=[0,1,2,3,4])

    inception_feats_config = get_heavy_node_config(model_name=INCEPTION_FEATS_MODEL_NAME,
                                                   batch_size=1,
                                                   num_replicas=3,
                                                   cpus_per_replica=2,
                                                   allocated_cpus=[5,21,6,22,7,23],
                                                   allocated_gpus=[5,6,7])
    
    kernel_svm_config = get_heavy_node_config(model_name=KERNEL_SVM_MODEL_NAME,
                                              batch_size=1,
                                              num_replicas=2,
                                              cpus_per_replica=2,
                                              allocated_cpus=[8,24,9,25])


    log_reg_config = get_heavy_node_config(model_name=LOG_REG_MODEL_NAME,
                                           batch_size=1,
                                           num_replicas=2,
                                           cpus_per_replica=2,
                                           allocated_cpus=[10,26,11,27])

    model_configs = {
        RESNET_152_MODEL_NAME : resnet_feats_config,
        KERNEL_SVM_MODEL_NAME : kernel_svm_config,
        INCEPTION_FEATS_MODEL_NAME : inception_feats_config,
        LOG_REG_MODEL_NAME : log_reg_config
    }

    return model_configs

def get_benchmark_model_configs():
    resnet_feats_config = get_heavy_node_config(model_name=RESNET_152_MODEL_NAME,
                                                batch_size=1,
                                                num_replicas=3,
                                                cpus_per_replica=2,
                                                allocated_cpus=[0,16,1,17,2,18,3,19,4,20],
                                                allocated_gpus=[0,1,2,3,4])

    inception_feats_config = get_heavy_node_config(model_name=INCEPTION_FEATS_MODEL_NAME,
                                                   batch_size=1,
                                                   num_replicas=2,
                                                   cpus_per_replica=2,
                                                   allocated_cpus=[5,21,6,22,7,23],
                                                   allocated_gpus=[5,6,7])
    
    kernel_svm_config = get_heavy_node_config(model_name=KERNEL_SVM_MODEL_NAME,
                                              batch_size=1,
                                              num_replicas=1,
                                              cpus_per_replica=2,
                                              allocated_cpus=[8,24,9,25])


    log_reg_config = get_heavy_node_config(model_name=LOG_REG_MODEL_NAME,
                                           batch_size=1,
                                           num_replicas=1,
                                           cpus_per_replica=2,
                                           allocated_cpus=[10,26,11,27])

    model_configs = {
        RESNET_152_MODEL_NAME : resnet_feats_config,
        KERNEL_SVM_MODEL_NAME : kernel_svm_config,
        INCEPTION_FEATS_MODEL_NAME : inception_feats_config,
        LOG_REG_MODEL_NAME : log_reg_config
    }

    return model_configs
