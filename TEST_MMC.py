import sys
import os
import logging
import argparse
import time
import numpy as np
import json
import copy

from datetime import datetime
from threading import Lock

from containerized_utils.zmq_client import Client

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

class StatsManager(object):

    def __init__(self, trial_length):
        self._init_stats()
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "mean_lats": [],
            "per_message_lats": {}
        }
        self.total_num_complete = 0
        self.trial_length = trial_length
        self.stats_lock = Lock()
        self.start_timestamp = datetime.now()

    def get_mean_thru_for_dequeue(self):
        self.stats_lock.acquire()
        mean_thru = np.mean(self.stats["thrus"][2:-1])
        self.stats_lock.release()
        return mean_thru

    def get_stats(self):
        self.stats_lock.acquire()
        result = copy.deepcopy(self.stats)
        self.stats_lock.release()
        return result

    def update_stats(self, request_id, latency):
        try:
            self.stats_lock.acquire()
            self.latencies.append(latency)
            self.stats["per_message_lats"][str(request_id)] = latency 
            self.trial_num_complete += 1 

            if self.trial_num_complete >= self.trial_length:
                self._print_stats()
                self._init_stats()

            self.stats_lock.release()
        except Exception as e:
            print("ERROR UPDATING STATS: {}".format(e))
            os._exit(1)

    def _init_stats(self):
        self.latencies = []
        self.batch_sizes = []
        self.trial_num_complete = 0
        self.cur_req_id = 0
        self.start_time = datetime.now()

    def _print_stats(self):
        end_time = datetime.now()
        thru = float(self.trial_num_complete) / (end_time - self.start_time).total_seconds()
        self.start_time = end_time

        lats = np.array(self.latencies)
        p99 = np.percentile(lats, 99)
        mean = np.mean(lats)
        self.stats["thrus"].append(thru)
        self.stats["p99_lats"].append(p99)
        self.stats["mean_lats"].append(mean)
        logger.info("p99_lat: {p99}, mean_lat: {mean}, thruput: {thru}".format(p99=p99,
                                                                               mean=mean,
                                                                               thru=thru))

def save_results(stats_manager, results_dir):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_subpath = "results_bs_50-{:%y%m%d_%H%M%S}.json".format(datetime.now())
    results_path = os.path.join(results_dir, results_subpath)
    with open(results_path, "w") as f:
        json.dump(stats_manager.get_stats(), f, indent=4)

    logger.info("Saved results to: {}".format(results_path))
    os._exit(0)

def main(app_name, num_trials, trial_length, request_delay, results_dir):
    client = Client("localhost", 4456, 4455)
    client.start()

    inflight_messages_lock = Lock()
    inflight_messages = {}

    stats_manager = StatsManager(trial_length=trial_length)

    def continuation(output_info):
        try:
            request_id, _ = output_info
            end_time = datetime.now()
            inflight_messages_lock.acquire()
            begin_time = inflight_messages[request_id]
            inflight_messages_lock.release()
            latency = (end_time - begin_time).total_seconds()
            stats_manager.update_stats(request_id, latency)
        except Exception as e:
            print("Exception in continuation: {}".format(e))

    logger.info("Generating inputs...")
    inputs = [np.random.rand(299, 299, 3).flatten() for _ in range(1000)]
    inputs = [i for _ in range(40) for i in inputs]

    logger.info("Starting predictions...")

    for i in range(len(inputs)):
        input_item = inputs[i]

        inflight_messages_lock.acquire()
        send_time = datetime.now()
        future, request_id = client.send_request(app_name, input_item)
        inflight_messages[request_id] = send_time
        inflight_messages_lock.release()

        future.then(continuation)

        time.sleep(request_delay)

        if len(stats_manager.stats["thrus"]) >= num_trials:
            break

    save_results(stats_manager, results_dir)
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MMC Experiments')
    parser.add_argument('-r',  '--results_dir', type=str, help="Path to a directory to which to save results")
    parser.add_argument('-t',   '--trial_length', type=int, default=200, help="The length of each experimental trial")
    parser.add_argument('-n',   '--num_trials', type=int, default=30, help="The number of experimental trials to run")
    parser.add_argument('-a',   '--app_name', type=str, help="The name of the application to query")
    parser.add_argument('-rd',   '--request_delay', type=float, help="The inter-request delay")

    args = parser.parse_args()

    main(args.app_name, args.num_trials, args.trial_length, args.request_delay, args.results_dir)
