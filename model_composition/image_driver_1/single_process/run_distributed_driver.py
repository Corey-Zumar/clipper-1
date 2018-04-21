import sys
import os
import json

from subprocess import Popen

CONFIG_KEY_BATCH_SIZE = "batch_size"
CONFIG_KEY_CPU_AFFINITIES = "cpu_affinities"
CONFIG_KEY_GPU_AFFINITIES = "gpu_affinities"
CONFIG_KEY_PROCESS_PATH = "process_path"
CONFIG_KEY_REPLICA_NUMS = "replica_nums"
CONFIG_KEY_TRIAL_LENGTH = "trial_length"
CONFIG_KEY_NUM_TRIALS = "num_trials"
CONFIG_KEY_SLO_MILLIS = "slo_millis"

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    return config

def launch_processes(config):
    batch_size = config[CONFIG_KEY_BATCH_SIZE]
    cpu_affinities = config[CONFIG_KEY_CPU_AFFINITIES]
    process_path = config[CONFIG_KEY_PROCESS_PATH]
    replica_nums = config[CONFIG_KEY_REPLICA_NUMS]
    trial_length = config[CONFIG_KEY_TRIAL_LENGTH]
    num_trials = config[CONFIG_KEY_NUM_TRIALS]
    slo_millis = config[CONFIG_KEY_SLO_MILLIS]
    gpu_affinities = config[CONFIG_KEY_GPU_AFFINITIES]

    for idx in range(len(replica_nums)):
        replica_num = replica_nums[idx]
        
        cpu_affinity = cpu_affinities[idx]
        cpu_aff_list = cpu_affinity.split(" ")
        comma_delimited_cpu_aff = ",".join(cpu_aff_list)

        gpu_affinity = gpu_affinities[idx]
        resnet_gpu, inception_gpu = gpu_affinity.split(" ")

        process_cmd = "(export CUDA_VISIBLE_DEVICES=\"{res_gpu},{incep_gpu}\";" \
                      " numactl -C {cd_cpu_aff} python driver.py -b {bs} -c {cpu_aff}" \
                      " -t {trials} -tl {length} -n {rep_num} -p {proc_file} -s {slo})".format(
                              res_gpu=resnet_gpu,
                              incep_gpu=inception_gpu,
                              cd_cpu_aff=comma_delimited_cpu_aff,
                              bs=batch_size,
                              cpu_aff=cpu_affinity,
                              trials=num_trials,
                              length=trial_length,
                              rep_num=replica_num,
                              proc_file=process_path,
                              slo=slo_millis)

        print("Running: {}".format(process_cmd))
        Popen(process_cmd, shell=True)

if __name__ == "__main__":
    config_path = sys.argv[1]
    config = load_config(config_path)
    launch_processes(config)
    
