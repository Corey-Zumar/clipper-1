import sys
import os
import json
import shutil
import argparse

from config_creator import HIERARCHY_SUBDIR_MEAN_PROVISION, HIERARCHY_SUBDIR_PEAK_PROVISION

TAGGED_CONFIG_KEY_MACHINE_ADDRESS = "machine_address"
TAGGED_CONFIG_KEY_CONFIG_PATH = "config_path"

def create_tagged_config(machine_addrs, config_paths):
    tagged_config = []

    for i in range(len(config_paths)):
        addr = machine_addrs[i]
        config_path = config_paths[i]

        config_item = { TAGGED_CONFIG_KEY_MACHINE_ADDRESS : addr , TAGGED_CONFIG_KEY_CONFIG_PATH : config_path }
        tagged_config.append(config_item)

    return tagged_config

def populate_tagged_configs_directory(machine_addrs, untagged_dir_path, tagged_dir_path):
    for subdir in os.listdir(untagged_dir_path):
        output_subdir_path = os.path.join(tagged_dir_path, subdir)
        os.mkdir(output_subdir_path)

        subdir_path = os.path.join(untagged_dir_path, subdir)

        # Sort config paths based on machine number, m<i> for i = 1, 2, ...
        config_paths = sorted([os.path.join(subdir_path, item) for item in os.listdir(subdir_path) if "config" in item])

        tagged_configs = create_tagged_config(machine_addrs, config_paths) 
        tagged_config_subpath = "{nr}_tagged_config.json".format(nr=subdir)
        tagged_config_path = os.path.join(output_subdir_path, tagged_config_subpath)
        with open(tagged_config_path, "w") as f:
            json.dump(tagged_configs, f, indent=4)


def tag_configs(machine_addrs, configs_dir, output_dir):
    assert configs_dir != output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mean_provision_output_path = os.path.join(output_dir, HIERARCHY_SUBDIR_MEAN_PROVISION)
    peak_provision_output_path = os.path.join(output_dir, HIERARCHY_SUBDIR_PEAK_PROVISION)
    os.mkdir(mean_provision_output_path)
    os.mkdir(peak_provision_output_path)

    mean_provision_path = os.path.join(configs_dir, HIERARCHY_SUBDIR_MEAN_PROVISION)
    peak_provision_path = os.path.join(configs_dir, HIERARCHY_SUBDIR_PEAK_PROVISION)

    populate_tagged_configs_directory(machine_addrs, mean_provision_path, mean_provision_output_path)
    populate_tagged_configs_directory(machine_addrs, peak_provision_path, peak_provision_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create machine-tagged configs for querying grpc model servers on specific hosts')
    parser.add_argument('-a', '--machine_addrs', type=str, nargs='+', help="Collection of machine addresses in hostname:port format")
    parser.add_argument('-c', '--configs_dir', type=str, help="Path to directory of configurations for which to create tags")
    parser.add_argument('-o', '--output_dir', type=str, help="Path to the directory to output tagged configurations")

    args = parser.parse_args()

    tag_configs(machine_addrs=args.machine_addrs, configs_dir=args.configs_dir, output_dir=args.output_dir)
