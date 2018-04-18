import sys
import os
import numpy as np
import argparse

MILLISECONDS_PER_SECOND = 1000
REPLICA_NUM_GENERATOR_SEED = 1200

def seed_generator():
    np.random.seed(REPLICA_NUM_GENERATOR_SEED)

def tag_arrival_process(process_path, output_path, num_replicas):
    def validate(tagged_process):
        replica_request_counts = {}
        for _, replica_num in tagged_process:
            if replica_num in replica_request_counts:
                replica_request_counts[replica_num] += 1
            else:
                replica_request_counts[replica_num] = 1

        for replica_num, request_count in replica_request_counts.iteritems():
            count_proportion = float(request_count) / len(tagged_process)
            print(replica_num, count_proportion)

    seed_generator()

    arrival_process = load_arrival_deltas(process_path)
    tagged_process = []
    for delta in arrival_process:
        replica_num = np.random.randint(num_replicas)
        tagged_process.append((delta, replica_num))

    validate(tagged_process)

    with open(output_path, "w+") as f:
        for delta, replica_num in tagged_process:
            f.write("{},{}\n".format(delta, replica_num))

def load_tagged_arrival_deltas(path):
    with open(path, "r") as f:
        arrival_lines = f.readlines()
        raw_tagged_deltas = [line.rstrip() for line in arrival_lines]
        tagged_deltas = []
        for delta_line in raw_tagged_deltas:
            delta, replica_num = delta_line.split(",")
            delta = float(delta)
            replica_num = int(replica_num)
            tagged_deltas.append((delta, replica_num))

        return tagged_deltas

def load_arrival_deltas(path):
    with open(path, "r") as f:
        arrival_lines = f.readlines()
        arrival_deltas = [float(line.rstrip()) for line in arrival_lines]

    return arrival_deltas

def calculate_mean_throughput(arrival_deltas_millis):
    cumulative = np.cumsum(arrival_deltas_millis)
    return MILLISECONDS_PER_SECOND * (len(cumulative) / (cumulative[-1] - cumulative[0]))

def calculate_peak_throughput(arrival_deltas_millis, slo_window_millis=250):
    cumulative = np.cumsum(arrival_deltas_millis)
    front = 0
    back = 1

    max_window_length_queries = 0

    while back < len(cumulative):
        if front == back:
            back += 1
            continue

        window_length_millis = cumulative[back] - cumulative[front]
        window_length_queries = back - front

        if window_length_millis <= slo_window_millis:
            max_window_length_queries = max(max_window_length_queries, window_length_queries)
            back += 1
        else:
            front += 1
    
    last_idx = len(cumulative) - 1
    while front < last_idx:
        window_length_millis = cumulative[last_idx] - cumulative[front]
        window_length_queries = last_idx - front

        if window_length_millis <= slo_window_millis:
            max_window_length_queries = max(max_window_length_queries, window_length_queries)
            break
        else:
            front += 1

    peak_throughput = (MILLISECONDS_PER_SECOND * float(max_window_length_queries)) / (slo_window_millis)
    return peak_throughput

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Obtain mean and peak throughputs from an arrival distribution')
    parser.add_argument('-p', '--process_file', type=str, help='The arrival process file path')

    args = parser.parse_args()

    deltas = load_arrival_deltas(args.process_file)

    print(min(np.cumsum(deltas)))
    print(len(deltas))
    
    mean_thru = calculate_mean_throughput(deltas)
    print(mean_thru)

    peak_thru = calculate_peak_throughput(deltas)
    print(peak_thru)

