import sys
import os
import argparse
import json
import shutil
import math

import bench_utils
import e2e_utils
import numpy as np

FIXED_MIN_LAT_BATCH_SIZE = 1
FIXED_NUM_CLIENTS = 1

CONFIG_KEY_MODEL_NAME = "model_name"
CONFIG_KEY_BATCH_SIZE = "batch_size"
CONFIG_KEY_NUM_REPLICAS = "num_replicas"
CONFIG_KEY_VCPUS_PER_REPLICA = "vcpus_per_replica"
CONFIG_KEY_ALLOCATED_VCPUS = "allocated_vcpus"
CONFIG_KEY_ALLOCATED_GPUS = "allocated_gpus"
CONFIG_KEY_PORTS = "ports"

CONFIG_KEY_NUM_TRIALS = "num_trials"
CONFIG_KEY_TRIAL_LENGTH = "trial_length"
CONFIG_KEY_NUM_CLIENTS = "num_clients"
CONFIG_KEY_SLO_MILLIS = "slo_millis"
CONFIG_KEY_LAMBDA = "lambda"
CONFIG_KEY_CV = "cv"
CONFIG_KEY_PROCESS_PATH = "process_path"

CONFIG_KEY_RESNET = "resnet"
CONFIG_KEY_INCEPTION = "inception"
CONFIG_KEY_KSVM = "ksvm"
CONFIG_KEY_LOG_REG = "log_reg"

PROFILE_KEY_RESNET = CONFIG_KEY_RESNET
PROFILE_KEY_INCEPTION = CONFIG_KEY_INCEPTION
PROFILE_KEY_KSVM = CONFIG_KEY_KSVM
PROFILE_KEY_LOG_REG = CONFIG_KEY_LOG_REG

PROFILE_KEYS = [
    PROFILE_KEY_RESNET,
    PROFILE_KEY_INCEPTION,
    PROFILE_KEY_KSVM,
    PROFILE_KEY_LOG_REG
]

GPUS_PER_REPLICA = {
    PROFILE_KEY_RESNET : 1,
    PROFILE_KEY_INCEPTION : 1,
    PROFILE_KEY_KSVM : 0,
    PROFILE_KEY_LOG_REG : 0
}

PCPUS_PER_REPLICA = {
    PROFILE_KEY_RESNET : 1,
    PROFILE_KEY_INCEPTION : 1,
    PROFILE_KEY_KSVM : 1,
    PROFILE_KEY_LOG_REG : 1
}

RESNET_PORT_RANGE = range(9500,9508)
INCEPTION_PORT_RANGE = range(9508, 9516)
KSVM_PORT_RANGE = range(9516, 9524)
LOG_REG_PORT_RANGE = range(9524, 9532)

MODEL_PORT_RANGES = {
    PROFILE_KEY_RESNET : RESNET_PORT_RANGE,
    PROFILE_KEY_INCEPTION : INCEPTION_PORT_RANGE,
    PROFILE_KEY_KSVM : KSVM_PORT_RANGE,
    PROFILE_KEY_LOG_REG : LOG_REG_PORT_RANGE
}

HIERARCHY_KEY_MEAN_PATHS = "mean"
HIERARCHY_KEY_PEAK_PATHS = "peak"

HIERARCHY_SUBDIR_MEAN_PROVISION = "mean_provision"
HIERARCHY_SUBDIR_PEAK_PROVISION = "peak_provision"

CONFIG_NUM_TRIALS = 30

def get_arrival_process_path(procs_dir_path, cv, lambda_val, tagged_num_replicas=None):
    if cv == 1:
        if tagged_num_replicas:
            fname = "{lv}_tagged_{nr}.deltas".format(lv=lambda_val, nr=tagged_num_replicas) 
        else:
            fname = "{lv}.deltas".format(lv=lambda_val)
    else:
        if tagged_num_replicas:
            fname = "{lv}_{cv}_tagged_{nr}.deltas".format(lv=lambda_val, cv=cv, nr=tagged_num_replicas) 
        else:
            fname = "{lv}_{cv}.deltas".format(lv=lambda_val, cv=cv)

    return os.path.join(procs_dir_path, fname)

def create_configs_hierarchy_lambda_vals(configs_base_dir_path, lambda_vals, cv, slo_millis):
    if not os.path.exists(configs_base_dir_path):
        try: 
            os.makedirs(configs_base_dir_path)
        except OSError:
            print("Failed to create outputs base directory with path: {}".format(configs_base_dir_path))
            raise

    slo_subpath = "slo_{slo_val}".format(slo_val=slo_millis)
    slo_path = os.path.join(configs_base_dir_path, slo_subpath)

    cv_subpath = "cv_{cv_val}".format(cv_val=cv)
    cv_path = os.path.join(slo_path, cv_subpath)

    mean_path = os.path.join(cv_path, HIERARCHY_SUBDIR_MEAN_PROVISION)
    peak_path = os.path.join(cv_path, HIERARCHY_SUBDIR_PEAK_PROVISION)

    try:
        os.makedirs(mean_path)
    except OSError:
        print("Failed to create mean provision directory with path: {}".format(mean_path))
        raise

    try:
        os.makedirs(peak_path)
    except OSError:
        print("Failed to create peak provision directory with path: {}".format(peak_path))
        raise

    path_outputs = { HIERARCHY_KEY_MEAN_PATHS : {}, HIERARCHY_KEY_PEAK_PATHS : {} }

    for lambda_val in lambda_vals:
        lambda_subpath = "lambda_{lv}".format(lv=lambda_val)
        mean_lambda_path = os.path.join(mean_path, lambda_subpath)
        peak_lambda_path = os.path.join(peak_path, lambda_subpath)

        try:
            os.mkdir(mean_lambda_path)
        except OSError:
            print("Failed to create mean provision lambda directory with path: {}".format(mean_lambda_path))
            raise

        try:
            os.mkdir(peak_lambda_path)
        except OSError:
            print("Failed to create peak provision lambda directory with path: {}".format(peak_lambda_path))
            raise

        path_outputs[HIERARCHY_KEY_MEAN_PATHS][lambda_val] = mean_lambda_path
        path_outputs[HIERARCHY_KEY_PEAK_PATHS][lambda_val] = peak_lambda_path

    return path_outputs

def create_per_model_json_configs(machine_config, gpus_per_machine, pcpus_per_machine):
    per_model_configs = {}

    available_vcpus_numa_one = range(pcpus_per_machine)
    available_vcpus_numa_two = [item + pcpus_per_machine for item in range(pcpus_per_machine)]
    available_gpus = range(gpus_per_machine)
    for model_key in machine_config:
        num_model_replicas = machine_config[model_key]

        if num_model_replicas <= 0:
            continue

        allocated_ports = MODEL_PORT_RANGES[model_key][:num_model_replicas]

        vcpus_per_replica = PCPUS_PER_REPLICA[model_key] * 2 
        num_allocated_gpus = GPUS_PER_REPLICA[model_key] * num_model_replicas
        num_allocated_pcpus = PCPUS_PER_REPLICA[model_key] * num_model_replicas

        allocated_gpus = []
        for _ in range(num_allocated_gpus):
            allocated_gpu = available_gpus.pop()
            allocated_gpus.append(allocated_gpu)

        allocated_vcpus = []
        for i in range(num_allocated_pcpus):
            allocated_vcpu_numa_one = available_vcpus_numa_one.pop()
            allocated_vcpu_numa_two = available_vcpus_numa_two.pop()
            allocated_vcpus.append(allocated_vcpu_numa_one)
            allocated_vcpus.append(allocated_vcpu_numa_two)

        model_json_config = {
            CONFIG_KEY_MODEL_NAME : model_key,
            CONFIG_KEY_BATCH_SIZE : FIXED_MIN_LAT_BATCH_SIZE,
            CONFIG_KEY_NUM_REPLICAS : num_model_replicas,
            CONFIG_KEY_VCPUS_PER_REPLICA : 2,
            CONFIG_KEY_ALLOCATED_GPUS : allocated_gpus,
            CONFIG_KEY_ALLOCATED_VCPUS : allocated_vcpus,
            CONFIG_KEY_PORTS: allocated_ports
        }

        per_model_configs[model_key] = model_json_config

    return per_model_configs

def populate_configs_directory(hierarchy_path,
                               process_path_output,
                               required_replica_config,
                               gpus_per_machine,
                               pcpus_per_machine,
                               lambda_val,
                               cv,
                               slo_millis):

    ### CREATE MACHINE CONFIGS ###

    curr_machine_num = 0
    curr_total_replica_config = {}
    curr_machine_replica_config = {}
    for key in required_replica_config:
        curr_total_replica_config[key] = 0
        curr_machine_replica_config[key] = 0

    remaining_machine_gpus = gpus_per_machine
    remaining_machine_pcpus = pcpus_per_machine
    while True:
        # Whether or not a new replica was added during the current iteration
        added_new_replica = False 
        for key in curr_total_replica_config:
            num_configured_replicas = curr_total_replica_config[key]
            num_required_replicas, _ = required_replica_config[key]

            if num_configured_replicas < num_required_replicas:
                required_additional_gpus = GPUS_PER_REPLICA[key]
                required_additional_pcpus = PCPUS_PER_REPLICA[key]

                if required_additional_gpus <= remaining_machine_gpus and required_additional_pcpus <= remaining_machine_pcpus:
                    curr_total_replica_config[key] += 1
                    curr_machine_replica_config[key] += 1
                        
                    remaining_machine_gpus -= required_additional_gpus
                    remaining_machine_pcpus -= required_additional_pcpus
                    added_new_replica = True

        if not added_new_replica:
            # The fact that we did not add a a replica means that either:
            # A) the current machine cannot support the hardware requirements of any 
            # remaining, required replicas. We should save the current machine configuration and move
            # to the next machine.
            #
            # B) We have assigned all specified replicas to available machines

            machine_subpath = "machine_{mach_num}".format(mach_num=curr_machine_num)
            machine_path = os.path.join(hierarchy_path, machine_subpath)
            os.mkdir(machine_path)

            model_json_configs = create_per_model_json_configs(curr_machine_replica_config, gpus_per_machine, pcpus_per_machine)
            for model_key, model_json_config in model_json_configs.iteritems():
                config_subpath = "lambda_{lv}_{mod_name}_server_{mach_num}_config.json".format(lv=lambda_val,
                                                                                               mod_name=model_key, 
                                                                                               mach_num=curr_machine_num)

                config_path = os.path.join(machine_path, config_subpath)
                with open(config_path, "w") as f:
                    json.dump(model_json_config, f, indent=4)

            done = True 
            for key in curr_total_replica_config:
                num_configured_replicas = curr_total_replica_config[key]
                num_required_replicas, _ = required_replica_config[key]

                if num_configured_replicas < num_required_replicas:
                    done = False

            if not done:
                curr_machine_num += 1
                curr_machine_replica_config = {}
                for key in required_replica_config:
                    curr_machine_replica_config[key] = 0

                remaining_machine_gpus = gpus_per_machine
                remaining_machine_pcpus = pcpus_per_machine

            else:
                break

    ### CREATE EXPERIMENT CONFIG ###

    num_trials = CONFIG_NUM_TRIALS
    trial_length = max(30, lambda_val * 5)
    
    experiment_config_json = {
        CONFIG_KEY_NUM_TRIALS : num_trials,
        CONFIG_KEY_TRIAL_LENGTH : trial_length,
        CONFIG_KEY_NUM_CLIENTS : FIXED_NUM_CLIENTS,
        CONFIG_KEY_SLO_MILLIS : slo_millis,
        CONFIG_KEY_CV : cv,
        CONFIG_KEY_LAMBDA : lambda_val,
        CONFIG_KEY_PROCESS_PATH : process_path_output
    }

    for model_key in required_replica_config:
        num_replicas, _ = required_replica_config[model_key]
        experiment_config_json[model_key] = num_replicas

    experiment_config_subpath = "{lv}_experiment_config.json".format(lv=lambda_val)
    experiment_config_path = os.path.join(hierarchy_path, experiment_config_subpath)
    with open(experiment_config_path, "w") as f:
        json.dump(experiment_config_json, f)

def parse_profiles(profiles_path):
    with open(profiles_path, "r") as f:
        profile_json = json.load(f)

    parsed_thrus = {}

    for profile_key in PROFILE_KEYS:
        profile_path = profile_json[profile_key]
        mean_thru = bench_utils.get_mean_throughput(profile_path)
        parsed_thrus[profile_key] = mean_thru

    return parsed_thrus

def find_replica_configuration(parsed_profiles, target_thru):
    replica_configurations = {}
    pipeline_thru = sys.maxint
    min_key = None
    for key in parsed_profiles:
        model_mean_thru = parsed_profiles[key]
        if model_mean_thru < pipeline_thru:
            min_key = key
            pipeline_thru = model_mean_thru

        replica_configurations[key] = (1, model_mean_thru)

    while pipeline_thru < target_thru:
        num_replicas, model_replicated_thru = replica_configurations[min_key]
        model_mean_thru = parsed_profiles[min_key]
        new_num_replicas = num_replicas + 1
        new_model_replicated_thru = model_replicated_thru + model_mean_thru
        replica_configurations[min_key] = (new_num_replicas, new_model_replicated_thru)

        pipeline_thru = sys.maxint
        for key in replica_configurations:
            num_replicas, model_replicated_thru = replica_configurations[key]
            if model_replicated_thru < pipeline_thru:
                pipeline_thru = model_replicated_thru
                min_key = key 

    return replica_configurations, pipeline_thru

def create_configs_find_min_cost(arrival_procs_path, 
                                 profiles_path,
                                 gpus_per_machine,
                                 pcpus_per_machine,
                                 slo_millis,
                                 lambda_vals,
                                 cv, 
                                 utilization_factor, 
                                 configs_base_dir):

    configs_hierarchy = create_configs_hierarchy_lambda_vals(configs_base_dir, lambda_vals, cv, slo_millis)
    arrival_procs, fpaths = bench_utils.load_relevant_arrival_procs(arrival_procs_path, cv)

    parsed_profiles = parse_profiles(profiles_path)

    for lambda_val in lambda_vals:
        hierarchy_mean_path = configs_hierarchy[HIERARCHY_KEY_MEAN_PATHS][lambda_val] 
        hierarchy_peak_path = configs_hierarchy[HIERARCHY_KEY_PEAK_PATHS][lambda_val] 

        lambda_proc = arrival_procs[lambda_val]
        lambda_fpath = fpaths[lambda_val]

        mean_thru = bench_utils.calculate_mean_throughput(lambda_proc)
        peak_thru = bench_utils.calculate_peak_throughput(lambda_proc, slo_millis)

        mean_replica_config, mean_pipeline_thru = find_replica_configuration(parsed_profiles, mean_thru)
        peak_replica_config, peak_pipeline_thru = find_replica_configuration(parsed_profiles, peak_thru)

        print("MEAN CONFIG", mean_replica_config)
        print("PEAK CONFIG", peak_replica_config) 

        mean_process_path_output = get_arrival_process_path(hierarchy_mean_path, 
                                                            cv=cv, 
                                                            lambda_val=lambda_val, 
                                                            tagged_num_replicas=None)


        peak_process_path_output = get_arrival_process_path(hierarchy_peak_path,                                                            
                                                            cv=cv, 
                                                            lambda_val=lambda_val, 
                                                            tagged_num_replicas=None)


        shutil.copyfile(lambda_fpath, mean_process_path_output) 
        shutil.copyfile(lambda_fpath, peak_process_path_output)

        populate_configs_directory(hierarchy_mean_path,
                                   mean_process_path_output,
                                   mean_replica_config,
                                   gpus_per_machine,
                                   pcpus_per_machine,
                                   lambda_val,
                                   cv,
                                   slo_millis)

        populate_configs_directory(hierarchy_peak_path,
                                   peak_process_path_output,
                                   peak_replica_config,
                                   gpus_per_machine,
                                   pcpus_per_machine,
                                   lambda_val,
                                   cv,
                                   slo_millis)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create configurations for SPD experiments')
    parser.add_argument('-cv',  '--cv', type=float, help="The CV for which to generate configurations")
    parser.add_argument('-l', '--lambda_vals', type=int, nargs="+", help="If specified, generate configs by finding the minimum cost configuration to support workloads with mean throughputs specified by each lambda")
    parser.add_argument('-lp', '--lambdas_path', type=str, help="If specified, a path to an SLO-keyed json file containing lambda values")
    
    parser.add_argument('-p',  '--arrival_procs_path', type=str, help="The path to the arrival processes directory")
    parser.add_argument('-s',  '--slo_profiles_path', type=str, help="The path to JSON-formatted profiles for all pipeline models")
    parser.add_argument('-u', '--utilization_factor', type=float, help="The utilization (decay) factor used to scale target thruputs when selecting lambda values") 
    parser.add_argument('-c', '--configs_base_dir', type=str, help="The output base directory to which to write configurations")
    parser.add_argument('-gm', '--gpus_per_machine', type=int, default=4, help="The number of gpus available on search server machine")
    parser.add_argument('-pcm', '--pcpus_per_machine', type=int, default=16, help="The number of cpus available on each server machine")
    parser.add_argument('-sm', '--slo_millis', type=int, help="The latency SLO, in milliseconds, that will be recorded when running experiments")
    parser.add_argument('-t', '--tag_procs', action="store_true", help="If specified, tags arrival processes in accordance with replica configurations")

    args = parser.parse_args()

    if args.lambda_vals or args.lambdas_path:
        if args.lambdas_path:
            slo_key_seconds = str(args.slo_millis * .001)
            cv_key = str(args.cv)
            with open(args.lambdas_path, "r") as f:
                lambdas_json = json.load(f)
                args.lambda_vals = lambdas_json[slo_key_seconds][cv_key]

    create_configs_find_min_cost(arrival_procs_path=args.arrival_procs_path, 
                                 profiles_path=args.slo_profiles_path,
                                 gpus_per_machine=args.gpus_per_machine,
                                 pcpus_per_machine=args.pcpus_per_machine,
                                 slo_millis=args.slo_millis,
                                 lambda_vals=args.lambda_vals,
                                 cv=args.cv,
                                 utilization_factor=args.utilization_factor,
                                 configs_base_dir=args.configs_base_dir)
