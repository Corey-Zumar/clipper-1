import sys
import os
import json
import shutil
import argparse

from config_creator import HIERARCHY_SUBDIR_MEAN_PROVISION, HIERARCHY_SUBDIR_PEAK_PROVISION

TAGGED_CONFIG_KEY_MACHINE_ADDRESS = "machine_address"
TAGGED_CONFIG_KEY_CONFIG_PATH = "config_path"

def machine_dir_comparator(addr1, addr2):
    addr1_machine_num = int(''.join(ch for ch in addr1 if ch.isdigit()))
    addr2_machine_num = int(''.join(ch for ch in addr2 if ch.isdigit()))

    if addr2_machine_num < addr1_machine_num:
        return -1
    elif addr1_machine_num < addr2_machine_num:
        return 1
    else:
        return 0

def create_tagged_config(machine_addr_dir_mapping):
    tagged_config = []
    for machine_addr, machine_subdir_output_path in machine_addr_dir_mapping.iteritems():
        config_item = { TAGGED_CONFIG_KEY_MACHINE_ADDRESS : machine_addr, TAGGED_CONFIG_KEY_CONFIG_PATH : machine_subdir_output_path }
        tagged_config.append(config_item)

    return tagged_config

def populate_tagged_configs_directory(machine_addrs, untagged_dir_path, tagged_dir_path):
    for lambda_subdir in os.listdir(untagged_dir_path):
        output_lambda_subdir_path = os.path.join(tagged_dir_path, lambda_subdir)
        os.mkdir(output_lambda_subdir_path)

        lambda_subdir_path = os.path.join(untagged_dir_path, lambda_subdir)

        machine_subdir_subpaths = sorted([dirname for dirname in os.listdir(lambda_subdir_path) if "machine" in dirname], cmp=machine_dir_comparator, reverse=True)
        machine_subdir_paths = [os.path.join(lambda_subdir_path, dirname) for dirname in machine_subdir_subpaths] 

        machine_addr_dir_mapping = {}

        for idx in range(len(machine_subdir_subpaths)):
            machine_subdir_output_path = os.path.join(output_lambda_subdir_path, machine_subdir_subpaths[idx])
            os.mkdir(machine_subdir_output_path)

            machine_subdir_path = machine_subdir_paths[idx]

            config_subpaths = [item for item in os.listdir(machine_subdir_path) if "config" in item]
            for config_subpath in config_subpaths:
                shutil.copy2(os.path.join(machine_subdir_path, config_subpath), machine_subdir_output_path)

            machine_addr = machine_addrs[idx]
            machine_addr_dir_mapping[machine_addr] = machine_subdir_output_path

        tagged_config = create_tagged_config(machine_addr_dir_mapping)

        tagged_config_subpath = "machine_tagged_config.json"
        tagged_config_path = os.path.join(output_lambda_subdir_path, tagged_config_subpath)
        with open(tagged_config_path, "w") as f:
            json.dump(tagged_config, f, indent=4)

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

    populate_tagged_configs_directory(list(machine_addrs), mean_provision_path, mean_provision_output_path)
    populate_tagged_configs_directory(list(machine_addrs), peak_provision_path, peak_provision_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create machine-tagged configs for querying grpc model servers on specific hosts')
    parser.add_argument('-a', '--machine_addrs', type=str, nargs='+', help="Collection of machine hostnames (NO PORTS)")
    parser.add_argument('-c', '--configs_dir', type=str, help="Path to directory of configurations for which to create tags")
    parser.add_argument('-o', '--output_dir', type=str, help="Path to the directory to output tagged configurations")

    args = parser.parse_args()

    tag_configs(machine_addrs=args.machine_addrs, 
                configs_dir=args.configs_dir, 
                output_dir=args.output_dir)
