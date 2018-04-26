import sys
import os
import argparse
import json
import shutil

import bench_utils
import e2e_utils

import numpy as np

CONFIG_KEY_BATCH_SIZE = "batch_size"
CONFIG_KEY_CPU_AFFINITIES = "cpu_affinities"
CONFIG_KEY_GPU_AFFINITIES = "gpu_affinities"
CONFIG_KEY_PROCESS_PATH = "process_path"
CONFIG_KEY_REPLICA_NUMS = "replica_nums"
CONFIG_KEY_TRIAL_LENGTH = "trial_length"
CONFIG_KEY_NUM_TRIALS = "num_trials"
CONFIG_KEY_SLO_MILLIS = "slo_millis"
CONFIG_KEY_LAMBDA = "lambda"
CONFIG_KEY_CV = "cv"

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

def create_configs_hierarchy(configs_base_dir_path, max_num_replicas, cv):
    if not os.path.exists(configs_base_dir_path):
        try: 
            os.makedirs(configs_base_dir_path)
        except OSError:
            print("Failed to create outputs base directory with path: {}".format(configs_base_dir_path))
            raise

    cv_subpath = "cv_{cv_val}".format(cv_val=cv)
    cv_path = os.path.join(configs_base_dir_path, cv_subpath)

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

    for replica_num in range(1, max_num_replicas + 1):
        replica_subpath = "{rn}_rep".format(rn=replica_num)
        mean_replica_path = os.path.join(mean_path, replica_subpath)
        peak_replica_path = os.path.join(peak_path, replica_subpath)

        try:
            os.mkdir(mean_replica_path)
        except OSError:
            print("Failed to create mean provision replica directory with path: {}".format(mean_replica_path))
            raise

        try:
            os.mkdir(peak_replica_path)
        except OSError:
            print("Failed to create peak provision replica directory with path: {}".format(peak_replica_path))
            raise

        path_outputs[HIERARCHY_KEY_MEAN_PATHS][replica_num] = mean_replica_path
        path_outputs[HIERARCHY_KEY_PEAK_PATHS][replica_num] = peak_replica_path

    return path_outputs
            
def create_config_json(process_path,
                       replicas_per_machine,
                       gpus_per_replica,
                       pcpus_per_replica,
                       replica_nums,
                       batch_size,
                       lambda_val,
                       cv,
                       slo_millis):

    """
    replica_num : [int]
        The ZERO-INDEXED replica numbers for which to create a json config
    """

    trial_length = max(30, lambda_val * 5)
    num_trials = CONFIG_NUM_TRIALS

    config_json = {
        CONFIG_KEY_BATCH_SIZE : batch_size,
        CONFIG_KEY_PROCESS_PATH : process_path,
        CONFIG_KEY_REPLICA_NUMS : replica_nums,
        CONFIG_KEY_TRIAL_LENGTH : trial_length,
        CONFIG_KEY_NUM_TRIALS : CONFIG_NUM_TRIALS,
        CONFIG_KEY_SLO_MILLIS : slo_millis,
        CONFIG_KEY_GPU_AFFINITIES : [],
        CONFIG_KEY_CPU_AFFINITIES : [],
        CONFIG_KEY_LAMBDA : lambda_val,
        CONFIG_KEY_CV : cv
    }

    for replica_num in replica_nums:
        machine_replica_num = replica_num % replicas_per_machine

        gpu_affinities = range(machine_replica_num * gpus_per_replica, (machine_replica_num + 1) * gpus_per_replica)
        cpu_affinities = range(machine_replica_num * pcpus_per_replica, (machine_replica_num + 1) * pcpus_per_replica)
        cpu_affinities = cpu_affinities + [16 + item for item in cpu_affinities]

        gpu_affinities_str = " ".join([str(aff_item) for aff_item in gpu_affinities])
        cpu_affinities_str = " ".join([str(aff_item) for aff_item in cpu_affinities])

        config_json[CONFIG_KEY_GPU_AFFINITIES].append(gpu_affinities_str)
        config_json[CONFIG_KEY_CPU_AFFINITIES].append(cpu_affinities_str)

    return config_json

def create_configs(arrival_procs_path, 
                   profile_path, 
                   max_num_replicas,
                   replicas_per_machine,
                   gpus_per_replica,
                   pcpus_per_replica,
                   batch_size,
                   slo_millis,
                   cv, 
                   utilization_factor, 
                   configs_base_dir,
                   tag_procs):

    configs_hierarchy = create_configs_hierarchy(configs_base_dir, max_num_replicas, cv)
    arrival_procs = bench_utils.load_relevant_arrival_procs(arrival_procs_path, cv)

    mean_profile_thruput = bench_utils.get_mean_throughput(profile_path)
    for num_replicas in xrange(1, max_num_replicas + 1):
        target_thruput = num_replicas * mean_profile_thruput * utilization_factor
        target_thrus = [target_thruput]

        peak_lambda, peak_thru = bench_utils.find_peak_arrival_proc(arrival_procs, target_thrus, slo_millis)[target_thruput]
        mean_lambda, mean_thru = bench_utils.find_mean_arrival_proc(arrival_procs, target_thrus)[target_thruput] 

        peak_lambda = int(peak_lambda)
        mean_lambda = int(mean_lambda)

        mean_process_path = get_arrival_process_path(arrival_procs_path, 
                                                     cv=cv, 
                                                     lambda_val=mean_lambda, 
                                                     tagged_num_replicas=None)

        peak_process_path = get_arrival_process_path(arrival_procs_path, 
                                                     cv=cv, 
                                                     lambda_val=peak_lambda, 
                                                     tagged_num_replicas=None)

        
        hierarchy_mean_path = configs_hierarchy[HIERARCHY_KEY_MEAN_PATHS][num_replicas] 
        hierarchy_peak_path = configs_hierarchy[HIERARCHY_KEY_PEAK_PATHS][num_replicas] 

        if tag_procs:
            mean_process_path_output = get_arrival_process_path(hierarchy_mean_path, 
                                                                cv=cv, 
                                                                lambda_val=mean_lambda, 
                                                                tagged_num_replicas=num_replicas)


            peak_process_path_output = get_arrival_process_path(hierarchy_peak_path,                                                            cv=cv, 
                                                                lambda_val=peak_lambda, 
                                                                tagged_num_replicas=num_replicas)

            e2e_utils.tag_arrival_process(mean_process_path, mean_process_path_output, num_replicas) 
            e2e_utils.tag_arrival_process(peak_process_path, peak_process_path_output, num_replicas)

        else:
            mean_process_path_output = get_arrival_process_path(hierarchy_mean_path, 
                                                                  cv=cv, 
                                                                  lambda_val=mean_lambda, 
                                                                  tagged_num_replicas=None)


            peak_process_path_output = get_arrival_process_path(hierarchy_peak_path,                                                            
                                                                  cv=cv, 
                                                                  lambda_val=peak_lambda, 
                                                                  tagged_num_replicas=None)


            shutil.copyfile(mean_process_path, mean_process_path_output) 
            shutil.copyfile(peak_process_path, peak_process_path_output) 

        i = 0
        j = 0
        while i < num_replicas:
            replica_nums = range(i, min(num_replicas, i + replicas_per_machine))
            mean_config_json = create_config_json(mean_process_path_output,
                                                  replicas_per_machine,
                                                  gpus_per_replica, 
                                                  pcpus_per_replica, 
                                                  replica_nums, 
                                                  batch_size,
                                                  mean_lambda,
                                                  cv,
                                                  slo_millis)

            mean_config_subpath = "{lv}_{nr}_rep_config_{tag}.json".format(lv=mean_lambda,
                                                                           nr=num_replicas,
                                                                           tag="m{}".format(j))

            mean_config_path = os.path.join(hierarchy_mean_path, mean_config_subpath) 

            with open(mean_config_path, "w") as f:
                json.dump(mean_config_json, f, indent=4)

            peak_config_json = create_config_json(peak_process_path_output,
                                                  replicas_per_machine,
                                                  gpus_per_replica, 
                                                  pcpus_per_replica, 
                                                  replica_nums, 
                                                  batch_size,
                                                  peak_lambda,
                                                  cv,
                                                  slo_millis)

            peak_config_subpath = "{lv}_{nr}_rep_config_{tag}.json".format(lv=peak_lambda,
                                                                           nr=num_replicas,
                                                                           tag="m{}".format(j))

            peak_config_path = os.path.join(hierarchy_peak_path, peak_config_subpath)
            
            with open(peak_config_path, "w") as f:
                json.dump(peak_config_json, f, indent=4)

            i += replicas_per_machine
            j += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create configurations for SPD experiments')
    parser.add_argument('-cv',  '--cv', type=float, help="The CV for which to generate configurations")
    parser.add_argument('-p',  '--arrival_procs_path', type=str, help="The path to the arrival processes directory")
    parser.add_argument('-s',  '--slo_profile_path', type=str, help="The path to a JSON profile for a fixed batch size corresponding to some SLO")
    parser.add_argument('-m', '--max_num_replicas', type=int, help="The maximum number of replicas for which to generate configs. Configs will be generated in the range (1, max]")
    parser.add_argument('-u', '--utilization_factor', type=float, help="The utilization (decay) factor used to scale target thruputs when selecting lambda values") 
    parser.add_argument('-c', '--configs_base_dir', type=str, help="The output base directory to which to write configurations")
    parser.add_argument('-r', '--replicas_per_machine', type=int, default=1, help="The number of replicas of SPD that can be launched per machine")
    parser.add_argument('-b', '--batch_size', type=int, help="The batch size that will be used when running experiments")
    parser.add_argument('-sm', '--slo_millis', type=int, help="The latency SLO, in milliseconds, that will be recorded when running experiments")
    parser.add_argument('-gr', '--gpus_per_replica', type=int, default=2, help="The number of GPUs required to run a single replica of SPD")
    parser.add_argument('-cr', '--cpus_per_replica', type=int, default=4, help="The number of PHYSICAL CPUs required to run a single replica of SPD")
    parser.add_argument('-t', '--tag_procs', action="store_true", help="If specified, tags arrival processes in accordance with replica configurations")

    args = parser.parse_args()

    create_configs(arrival_procs_path=args.arrival_procs_path, 
                   profile_path=args.slo_profile_path, 
                   max_num_replicas=args.max_num_replicas,
                   replicas_per_machine=args.replicas_per_machine,
                   gpus_per_replica=args.gpus_per_replica,
                   pcpus_per_replica=args.cpus_per_replica,
                   batch_size=args.batch_size,
                   slo_millis=args.slo_millis,
                   cv=args.cv, 
                   utilization_factor=args.utilization_factor,
                   configs_base_dir=args.configs_base_dir,
                   tag_procs=args.tag_procs)

