import sys
import os
import numpy as np
import argparse

MILLISECONDS_PER_SECOND = 1000

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

