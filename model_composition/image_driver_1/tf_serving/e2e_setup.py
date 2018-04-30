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

from e2e_configs import load_server_configs   

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

########## Setup ##########

def setup_heavy_nodes(configs_dir):
    """
    Parameters
    ------------
    config_dir : str
        Path to a directory containing json-formatted
        node configurations 
    """

    node_configs = load_server_configs(configs_dir)

    for config in node_configs:
        tfs_utils.setup_heavy_node(config)

    time.sleep(5)

if __name__ == "__main__":
    configs_dir_path = sys.argv[1]

    logger.info("Setting up pipeline nodes...")

    setup_heavy_nodes(configs_dir_path)