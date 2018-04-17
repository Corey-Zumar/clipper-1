import json
import sys
import os
import numpy as np

from collections import OrderedDict

from e2e_utils import load_arrival_deltas, calculate_mean_throughput, calculate_peak_throughput

ARRIVAL_PROCS_DIR = "cached_arrival_processes"

def get_mean_throughput(config_path):
    with open(config_path, "r") as f:
        config_json = json.load(f)

    thrus = [float(thru) for thru in config_json["client_metrics"][0]["thrus"]]
    print(np.mean(thrus))
    return np.mean(thrus)

def load_arrival_procs():
    deltas_dict = {}
    fnames = [os.path.join(ARRIVAL_PROCS_DIR, fname) for fname in os.listdir(ARRIVAL_PROCS_DIR) if "deltas" in fname]
    for fname in fnames:
        print(fname)
        deltas_subname = fname.split("/")[1]
        delta = int(deltas_subname.split(".")[0])
        deltas_dict[-1 * delta] = load_arrival_deltas(fname)

    return OrderedDict(sorted(deltas_dict.items()))
        

def find_peak_arrival_proc(arrival_procs, target_thru):
    for mean_delta, proc in arrival_procs.iteritems():
        mean_thru = calculate_peak_throughput(proc, slo_window_millis=350)
        print(abs(mean_delta), mean_thru)
        if mean_thru <= target_thru:
            return mean_delta


def find_mean_arrival_proc(arrival_procs, target_thru):
    for mean_delta, proc in arrival_procs.iteritems():
        mean_thru = calculate_mean_throughput(proc)
        print(abs(mean_delta), mean_thru)
        if mean_thru <= target_thru:
            return mean_delta

if __name__ == "__main__":
    path = sys.argv[1]
    num_replicas = int(sys.argv[2])

    arrival_procs = load_arrival_procs()
    mean_thruput = get_mean_throughput(path)
    target_thruput = num_replicas * mean_thruput

    peak_delta = find_peak_arrival_proc(arrival_procs, target_thruput)
    print(abs(peak_delta))

    mean_delta = find_mean_arrival_proc(arrival_procs, target_thruput)
    print(abs(mean_delta))

