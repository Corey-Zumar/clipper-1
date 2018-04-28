import sys
import os
import argparse
import json 

import numpy as np

from machine_config_tagger import TAGGED_CONFIG_KEY_MACHINE_ADDRESS

def setup(tagged_config_path, port):
    with open(tagged_config_path, "r") as f:
        replica_configs = json.load(f)
        target_config = None
        for entry in replica_configs:
            replica_host, replica_port = entry[TAGGED_CONFIG_KEY_MACHINE_ADDRESS].split(":")
            if replica_port == port:
                target_config = entry

        if not target_config:
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Setup SPD replicas based on specified configurations')
    parser.add_argument('-tc',  '--tagged_config_path', type=str, help="Path to the machine-tagged benchmark config")
    parser.add_argument('-p',   '--port', type=int, help="The port to match to a machine address")

    args = parser.parse_args()



