import json
import sys
import os
import numpy as np
import math

from collections import OrderedDict

from e2e_utils import load_arrival_deltas, calculate_mean_throughput, calculate_peak_throughput

ARRIVAL_PROCS_DIR = "cached_arrival_processes"

THROUGHPUT_UTILIZATION_DECAY_FACTOR = .8

def get_mean_throughput(config_path):
    with open(config_path, "r") as f:
        config_json = json.load(f)

    thrus = [float(thru) for thru in config_json["client_metrics"][0]["thrus"]]
    print(np.mean(thrus) * THROUGHPUT_UTILIZATION_DECAY_FACTOR)
    return np.mean(thrus)

def load_arrival_procs(cv):
    deltas_dict = {}
    if cv == 1:
        fnames = [os.path.join(ARRIVAL_PROCS_DIR, fname) for fname in os.listdir(ARRIVAL_PROCS_DIR) if ("deltas" in fname) and "_" not in fname]
    else:
        fnames = [os.path.join(ARRIVAL_PROCS_DIR, fname) for fname in os.listdir(ARRIVAL_PROCS_DIR) if ("deltas" in fname) and str(cv) in fname]
    for fname in fnames:
        print(fname)
        deltas_subname = fname.split("/")[1]
        if "_" in deltas_subname:
            delta = int(deltas_subname.split("_")[0])
        else:
            delta = int(deltas_subname.split(".")[0])

        deltas_dict[delta] = load_arrival_deltas(fname)

    return OrderedDict(sorted(deltas_dict.items()))

def probe_throughputs(eval_fn, arrival_process):
    min = 0
    max = len(arrival_process)
    highest_successful_config = None
    while True:
        if max == min:
            break
        middle = min + math.ceil((max - min) / 2)
        print("PROBING. min: {}, max: {}, middle: {}".format(
            min, max, middle))

        result = eval_fn(int(middle))

        if result:
            min = middle
            highest_successful_config = result
        else:
            max = middle - 1
    return highest_successful_config
        

def find_peak_arrival_proc(arrival_procs, target_thru):
    
    def eval_fn(middle):
        key = arrival_procs.keys()[middle]
        peak_thru = calculate_peak_throughput(arrival_procs[key], slo_window_millis=350)
        if peak_thru <= target_thru:
            return (key, peak_thru)
        else:
            return None

    return probe_throughputs(eval_fn, arrival_procs)

def find_mean_arrival_proc(arrival_procs, target_thru):
    
    def eval_fn(middle):
        key = arrival_procs.keys()[middle]
        mean_thru = calculate_mean_throughput(arrival_procs[key])
        if mean_thru <= target_thru:
            return (key, mean_thru)
        else:
            return None

    return probe_throughputs(eval_fn, arrival_procs)

if __name__ == "__main__":
    path = sys.argv[1]
    cv = float(sys.argv[2])
    num_replicas = int(sys.argv[3])

    arrival_procs = load_arrival_procs(cv)
    mean_thruput = get_mean_throughput(path)
    target_thruput = num_replicas * mean_thruput

    # IMPORTANT: DECAY THROUGHPUT BY THE SPECIFIED DECAY FACTOR
    target_thruput = target_thruput * THROUGHPUT_UTILIZATION_DECAY_FACTOR

    peak_delta = find_peak_arrival_proc(arrival_procs, target_thruput)
    print(peak_delta)

    mean_delta = find_mean_arrival_proc(arrival_procs, target_thruput)
    print(mean_delta)
