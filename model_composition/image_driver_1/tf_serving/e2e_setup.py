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

from e2e_configs import get_setup_model_configs 

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

########## Setup ##########

def setup_heavy_nodes(configs):
    """
    Parameters
    ------------
    configs : dict
        Dictionary of TFSHeavyNodeConfig objects,
        keyed on model names
    """

    for config in configs.values():
        tfs_utils.setup_heavy_node(config)

    time.sleep(5)

if __name__ == "__main__":
    model_configs = get_setup_model_configs()

    logger.info("Setting up pipeline nodes...")

    # Set up TFS nodes
    setup_heavy_nodes(model_configs)
