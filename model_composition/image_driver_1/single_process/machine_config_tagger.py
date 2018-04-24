import sys
import os
import json
import shutil
import argparse

from config_creator import HIERARCHY_SUBDIR_MEAN_PROVISION, HIERARCHY_SUBDIR_PEAK_PROVISION
from config_creator import CONFIG_KEY_REPLICA_NUMS

TAGGED_CONFIG_KEY_MACHINE_ADDRESS = "machine_address"
TAGGED_CONFIG_KEY_CONFIG_PATH = "config_path"

from config_creator import CONFIG_KEY_REPLICA_NUMS

def create_tagged_config(machine_addrs, config_paths, machines_per_config):
    tagged_config = []

    machine_idx = 0
    for i in range(len(config_paths)):
        config_path = config_paths[i]
        
        with open(config_path, "r") as f:
            config_json = json.load(f)
            replica_nums = config_json[CONFIG_KEY_REPLICA_NUMS]
        

        for _ in range(len(replica_nums)):
            addr = machine_addrs[machine_idx]
            config_item = { TAGGED_CONFIG_KEY_MACHINE_ADDRESS : addr , TAGGED_CONFIG_KEY_CONFIG_PATH : config_path }
            tagged_config.append(config_item)
            machine_idx += 1

    return tagged_config

def populate_tagged_configs_directory(machine_addrs, machines_per_config, untagged_dir_path, tagged_dir_path):
    for subdir in os.listdir(untagged_dir_path):
        output_subdir_path = os.path.join(tagged_dir_path, subdir)
        os.mkdir(output_subdir_path)

        subdir_path = os.path.join(untagged_dir_path, subdir)

        # Sort config paths based on machine number, m<i> for i = 1, 2, ...
        config_subpaths = sorted([item for item in os.listdir(subdir_path) if "config" in item])
        copied_config_paths = []
        for config_subpath in config_subpaths:
            shutil.copy2(os.path.join(subdir_path, config_subpath), output_subdir_path)
            copied_config_path = os.path.join(output_subdir_path, config_subpath)
            copied_config_paths.append(copied_config_path)

        tagged_configs = create_tagged_config(machine_addrs, copied_config_paths, machines_per_config) 
        tagged_config_subpath = "{nr}_tagged_config.json".format(nr=subdir)
        tagged_config_path = os.path.join(output_subdir_path, tagged_config_subpath)
        with open(tagged_config_path, "w") as f:
            json.dump(tagged_configs, f, indent=4)


def tag_configs(machine_addrs, machines_per_config, configs_dir, output_dir):
    assert configs_dir != output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mean_provision_output_path = os.path.join(output_dir, HIERARCHY_SUBDIR_MEAN_PROVISION)
    peak_provision_output_path = os.path.join(output_dir, HIERARCHY_SUBDIR_PEAK_PROVISION)
    os.mkdir(mean_provision_output_path)
    os.mkdir(peak_provision_output_path)

    mean_provision_path = os.path.join(configs_dir, HIERARCHY_SUBDIR_MEAN_PROVISION)
    peak_provision_path = os.path.join(configs_dir, HIERARCHY_SUBDIR_PEAK_PROVISION)

    populate_tagged_configs_directory(machine_addrs, machines_per_config, mean_provision_path, mean_provision_output_path)
    populate_tagged_configs_directory(machine_addrs, machines_per_config, peak_provision_path, peak_provision_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create machine-tagged configs for querying grpc model servers on specific hosts')
    parser.add_argument('-a', '--machine_addrs', type=str, nargs='+', help="Collection of machine addresses in hostname:port format")
    parser.add_argument('-c', '--configs_dir', type=str, help="Path to directory of configurations for which to create tags")
    parser.add_argument('-o', '--output_dir', type=str, help="Path to the directory to output tagged configurations")
    parser.add_argument('-m', '--machines_per_config', type=int, default=2, help="The number of machines to assign to each config")

    args = parser.parse_args()

    tag_configs(machine_addrs=args.machine_addrs, 
                machines_per_config=args.machines_per_config, 
                configs_dir=args.configs_dir, 
                output_dir=args.output_dir)
