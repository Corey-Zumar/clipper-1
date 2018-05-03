import sys
import os
import argparse
import numpy as np
import time
import base64
import logging
import json
import copy

from threading import Thread, Lock
from datetime import datetime
from multiprocessing import Process, Pipe, Value 
from concurrent.futures import ThreadPoolExecutor

from tf_serving_utils import GRPCClient, ReplicaAddress, REQUEST_PIPE_POLLING_TIMEOUT_SECONDS
from tf_serving_utils import tfs_utils

from tf_serving_utils.config_utils import CONFIG_KEY_NUM_TRIALS, CONFIG_KEY_TRIAL_LENGTH, CONFIG_KEY_NUM_CLIENTS
from tf_serving_utils.config_utils import CONFIG_KEY_SLO_MILLIS, CONFIG_KEY_PROCESS_PATH 
from tf_serving_utils.config_utils import CONFIG_KEY_CV, CONFIG_KEY_LAMBDA
from tf_serving_utils.config_utils import CONFIG_KEY_RESNET, CONFIG_KEY_INCEPTION, CONFIG_KEY_KSVM, CONFIG_KEY_LOG_REG

from tf_serving_utils.config_utils import TAGGED_CONFIG_KEY_TAGGED_MACHINES, TAGGED_CONFIG_KEY_EXPERIMENT_CONFIG
from tf_serving_utils.config_utils import TAGGED_CONFIG_KEY_MACHINE_ADDRESS, TAGGED_CONFIG_KEY_CONFIG_PATH 

from e2e_configs import load_client_configs, load_server_configs

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

# Models and applications for each heavy node
# will share the same name
INCEPTION_FEATS_MODEL_NAME = CONFIG_KEY_INCEPTION
RESNET_152_MODEL_NAME = CONFIG_KEY_RESNET
LOG_REG_MODEL_NAME = CONFIG_KEY_LOG_REG
KERNEL_SVM_MODEL_NAME = CONFIG_KEY_KSVM

ALL_MODEL_NAMES = [
    INCEPTION_FEATS_MODEL_NAME,
    RESNET_152_MODEL_NAME,
    KERNEL_SVM_MODEL_NAME,
    LOG_REG_MODEL_NAME
]

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

INCEPTION_FEATS_OUTPUT_KEY = "feats"
RESNET_FEATS_OUTPUT_KEY = "feats"
KERNEL_SVM_OUTPUT_KEY = "outputs"
LOG_REG_OUTPUT_KEY = "outputs"

CLIENT_KEY_OUTBOUND_HANDLE = "outbound_handle"
CLIENT_KEY_INBOUND_HANDLE = "inbound_handle"
CLIENT_KEY_REPLICA_GROUP_SIZE = "replica_group_size"
CLIENT_KEY_QUEUE_SIZE = "queue_size_val"

# We will assign at most 5 replicas to a single client process
REPLICA_GROUP_SIZE = 10

########## Client Setup ##########

class ClientConfig:

    def __init__(self, model_name, host, port):
        self.model_name = model_name
        self.host = host
        self.port = port

class ExperimentConfig:

    def __init__(self, 
                 num_trials, 
                 trial_length, 
                 num_clients, 
                 slo_millis,
                 cv,
                 lambda_val,
                 process_path,
                 node_configs,
                 client_configs):
        self.num_trials = num_trials
        self.trial_length = trial_length
        self.num_clients = num_clients
        self.slo_millis = slo_millis
        self.cv = cv
        self.lambda_val = lambda_val
        self.process_path = process_path
        self.node_configs = node_configs 
        self.client_configs = client_configs

def load_experiment_config(config_path, 
                           resnet_extras, 
                           inception_extras,
                           ksvm_extras,
                           log_reg_extras):
    with open(config_path, "r") as f:
        experiment_config_json = json.load(f)

    # Temporary hack
    nodes_path = "/home/ubuntu/clipper/model_composition/image_driver_1/tf_serving/CONFIGS_COST/tagged/500ms_cv0.1/mean_provision/lambda_426/machine_tagged_config.json"
    with open(nodes_path, "r") as f:
        nodes_json = json.load(f)

    # Machines that are actually available and will be used for client creation (up to the limits required by the experiment)
    client_tagged_machines = nodes_json[TAGGED_CONFIG_KEY_TAGGED_MACHINES]
    # Machines provisioned for the experiment (the number of "experiment_tagged_machines" sets the limit for client creation
    # from "client_tagged_machines"
    experiment_tagged_machines = experiment_config_json[TAGGED_CONFIG_KEY_TAGGED_MACHINES]

    experiment_config_params = experiment_config_json[TAGGED_CONFIG_KEY_EXPERIMENT_CONFIG]

    num_trials = experiment_config_params[CONFIG_KEY_NUM_TRIALS]
    trial_length = experiment_config_params[CONFIG_KEY_TRIAL_LENGTH]
    num_clients = experiment_config_params[CONFIG_KEY_NUM_CLIENTS]
    slo_millis = experiment_config_params[CONFIG_KEY_SLO_MILLIS]
    cv = experiment_config_params[CONFIG_KEY_CV]
    lambda_val = experiment_config_params[CONFIG_KEY_LAMBDA]
    process_path = experiment_config_params[CONFIG_KEY_PROCESS_PATH]

    client_model_configs = {}
    all_node_configs = []
    for tagged_machine in client_tagged_machines:
        config_path = tagged_machine[TAGGED_CONFIG_KEY_CONFIG_PATH]
        host = tagged_machine[TAGGED_CONFIG_KEY_MACHINE_ADDRESS]

        client_configs = load_client_configs(config_path)
        for config in client_configs:
            model_name, ports = config
            required_replicas = experiment_config_params[model_name]
            
            if model_name not in client_model_configs:
                client_model_configs[model_name] = []

            while len(client_model_configs[model_name]) < required_replicas and len(ports) > 0:
                port = ports.pop(0)
                new_config = ClientConfig(model_name, host, port)
                client_model_configs[model_name].append(new_config)

    for tagged_machine in experiment_tagged_machines:
        config_path = tagged_machine[TAGGED_CONFIG_KEY_CONFIG_PATH]

        machine_node_configs = load_server_configs(config_path)
        all_node_configs += machine_node_configs

    def add_extras(model_name, extras):
        if not extras:
            return

        for extra in extras:
            host, port = extra.split(":")
            new_config = ClientConfig(model_name, host, port)
            client_model_configs[model_name].append(new_config)

            for machine_node_config in all_node_configs:
                if machine_node_config.name == model_name:
                    all_node_configs.append(copy.copy(machine_node_config))
                    break

    add_extras(RESNET_152_MODEL_NAME, resnet_extras)
    add_extras(INCEPTION_FEATS_MODEL_NAME, inception_extras)
    add_extras(KERNEL_SVM_MODEL_NAME, ksvm_extras)
    add_extras(LOG_REG_MODEL_NAME, log_reg_extras)

    experiment_config = ExperimentConfig(num_trials=num_trials, 
                                         trial_length=trial_length, 
                                         num_clients=num_clients, 
                                         slo_millis=slo_millis,
                                         cv=cv,
                                         lambda_val=lambda_val,
                                         process_path=process_path,
                                         node_configs=all_node_configs,
                                         client_configs=client_model_configs)

    return experiment_config 

def create_clients(configs):
    """
    Parameters
    ------------
    configs : dict
        Dictionary of ClientConfig objects,
        keyed on model names
    """

    def launch_client(model_name, replica_addrs, outbound_handle, inbound_handle, queue_size_val):
        client = GRPCClient(model_name, replica_addrs, outbound_handle, inbound_handle, queue_size_val)
        client.start()
        time.sleep(60 * 60 * 24)

    client_handles = {}
    for model_name in configs:
        replica_addrs = [ReplicaAddress(client_config.host, int(client_config.port)) for client_config in configs[model_name] if not (client_config.host == "172.34.180.129" and client_config.port not in [9517, 9524])]
        addr_idx = 0
        while addr_idx < len(replica_addrs):
            replica_group_addrs = replica_addrs[addr_idx : min(addr_idx + REPLICA_GROUP_SIZE, len(replica_addrs))]

            outbound_parent_handle, outbound_child_handle = Pipe()
            inbound_parent_handle, inbound_child_handle = Pipe()

            # Initialize a multiprocess value that will store, as an unsigned long,
            # the size of the client's request queue
            queue_size_val = Value('L') 

            client_proc = Process(target=launch_client, 
                                  args=(model_name, 
                                        replica_group_addrs, 
                                        outbound_child_handle, 
                                        inbound_child_handle,
                                        queue_size_val))

            client_proc.start()

            new_handles = { 
                                CLIENT_KEY_OUTBOUND_HANDLE : outbound_parent_handle,
                                CLIENT_KEY_INBOUND_HANDLE : inbound_parent_handle,
                                CLIENT_KEY_REPLICA_GROUP_SIZE : len(replica_group_addrs),
                                CLIENT_KEY_QUEUE_SIZE : queue_size_val
                           }

            if model_name not in client_handles:
                client_handles[model_name] = []
            client_handles[model_name].append(new_handles)

            addr_idx += REPLICA_GROUP_SIZE

    return client_handles 

########## Benchmarking ##########

class Predictor(object):

    def __init__(self, trial_length, clients):
        self.trial_length = trial_length
        self.clients = clients
        
        self.response_threads = []
        for model_name in ALL_MODEL_NAMES:
            replica_group_handles = self.clients[model_name]
            for handle in replica_group_handles:
                inbound_handle = handle[CLIENT_KEY_INBOUND_HANDLE]
                response_thread = Thread(target=self._run_response_service, args=(inbound_handle, model_name, ))
                response_thread.start()
                self.response_threads.append(response_thread)

        self.continuation_executor = ThreadPoolExecutor(max_workers=16)
        
        self.inflight_requests_lock = Lock()
        self.inflight_requests = {}

        self.init_stats()
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "all_lats": [],
            "mean_lats": []}

        self.stats["model_pred_lats"] = { KERNEL_SVM_MODEL_NAME : [], LOG_REG_MODEL_NAME : [],
                                          RESNET_152_MODEL_NAME : [],  INCEPTION_FEATS_MODEL_NAME : [] }
        self.total_num_complete = 0

        self.ingest_start_time = datetime.now()
        self.num_enqueued = 0

        self.response_prop_lats = []

    def init_stats(self):
        self.latencies = []
        self.batch_num_complete = 0
        self.cur_req_id = 0
        self.start_time = datetime.now()

    def print_stats(self):
        lats = np.array(self.latencies)
        p99 = np.percentile(lats, 99)
        mean = np.mean(lats)
        end_time = datetime.now()
        thru = float(self.batch_num_complete) / (end_time - self.start_time).total_seconds()
        self.stats["thrus"].append(thru)
        self.stats["p99_lats"].append(p99)
        self.stats["all_lats"].append(self.latencies)
        self.stats["mean_lats"].append(mean)
        logger.info("p99: {p99}, mean: {mean}, thruput: {thru}".format(p99=p99,
                                                                       mean=mean,
                                                                       thru=thru))
    def predict(self, msg_id):
        send_time = datetime.now()
        self.inflight_requests[msg_id] = (send_time, {}, Lock())

        resnet_replica_group_handles = self.clients[RESNET_152_MODEL_NAME]
        inception_replica_group_handles = self.clients[INCEPTION_FEATS_MODEL_NAME]

        resnet_handle = self._weighted_select_handle(resnet_replica_group_handles)
        inception_handle = self._weighted_select_handle(inception_replica_group_handles)

        resnet_handle.send(msg_id)
        inception_handle.send(msg_id)

        self.num_enqueued += 1
        if self.num_enqueued > 100:
            trial_end_time = datetime.now()
            ingest_rate = float(self.num_enqueued) / (trial_end_time - self.ingest_start_time).total_seconds()
            # print("INGEST RATE: {} qps".format(ingest_rate))
            self.num_enqueued = 0
            self.ingest_start_time = trial_end_time


    def _weighted_select_handle(self, group_handles):
        min_size = sys.maxint
        min_handle_idx = None

        for i in range(len(group_handles)):
            handle_item = group_handles[i]
            # If this is too slow, try the power of two choices
            queue_size_val = handle_item[CLIENT_KEY_QUEUE_SIZE]
            with queue_size_val.get_lock():
                queue_size = int(queue_size_val.value)

            if queue_size < min_size:
                min_size = queue_size 
                min_handle_idx = i

        return group_handles[min_handle_idx][CLIENT_KEY_OUTBOUND_HANDLE]
        
    def _update_perf_stats(self, msg_id):
        end_time = datetime.now()
        begin_time = self.inflight_requests[msg_id][0]
        latency = (end_time - begin_time).total_seconds()
        self.latencies.append(latency)
        self.total_num_complete += 1
        self.batch_num_complete += 1
        if self.batch_num_complete % self.trial_length == 0:
            self.print_stats()
            self.init_stats()
            mean_prop_lat = np.mean(self.response_prop_lats)
            p99_prop_lat = np.mean(self.response_prop_lats)
            self.response_prop_lats = []
            
    def _resnet_feats_continuation(self, msg_ids):
        ksvm_replica_group_handles = self.clients[KERNEL_SVM_MODEL_NAME]
        model_pred_lats = self.stats["model_pred_lats"][RESNET_152_MODEL_NAME]
        for msg_id in msg_ids:
            ksvm_handle = self._weighted_select_handle(ksvm_replica_group_handles)
            ksvm_handle.send(msg_id)

    def _inception_feats_continuation(self, msg_ids):
        log_reg_replica_group_handles = self.clients[LOG_REG_MODEL_NAME]
        model_pred_lats = self.stats["model_pred_lats"][INCEPTION_FEATS_MODEL_NAME]
        for msg_id in msg_ids:
            log_reg_handle = self._weighted_select_handle(log_reg_replica_group_handles)
            log_reg_handle.send(msg_id)

    def _ksvm_continuation(self, msg_ids):
        model_pred_lats = self.stats["model_pred_lats"][KERNEL_SVM_MODEL_NAME]
        for msg_id in msg_ids:
            _, completed_dict, lock = self.inflight_requests[msg_id]
            lock.acquire()
            if LOG_REG_MODEL_NAME in completed_dict:
                lock.release()
                self._update_perf_stats(msg_id)
            else:
                completed_dict[KERNEL_SVM_MODEL_NAME] = True
                lock.release()

    def _log_reg_continuation(self, msg_ids):
        model_pred_lats = self.stats["model_pred_lats"][LOG_REG_MODEL_NAME]
        for msg_id in msg_ids:
            _, completed_dict, lock = self.inflight_requests[msg_id]
            lock.acquire()
            if KERNEL_SVM_MODEL_NAME in completed_dict:
                lock.release()
                self._update_perf_stats(msg_id)
            else:
                completed_dict[LOG_REG_MODEL_NAME] = True
                lock.release()

    def _run_response_service(self, inbound_handle, model_name):
        try:
            if model_name == RESNET_152_MODEL_NAME:
                continuation_fn = self._resnet_feats_continuation
            elif model_name == INCEPTION_FEATS_MODEL_NAME:
                continuation_fn = self._inception_feats_continuation
            elif model_name == KERNEL_SVM_MODEL_NAME:
                continuation_fn = self._ksvm_continuation
            elif model_name == LOG_REG_MODEL_NAME:
                continuation_fn = self._log_reg_continuation

            while True:
                data_available = inbound_handle.poll(REQUEST_PIPE_POLLING_TIMEOUT_SECONDS)
                if not data_available:
                    continue
           
                inbound_msg_ids = [inbound_handle.recv()]
                while inbound_handle.poll(0):
                    inbound_msg_id = inbound_handle.recv()
                    inbound_msg_ids.append(inbound_msg_id)

                # self.continuation_executor.submit(continuation_fn, inbound_msg_ids)
                continuation_fn(inbound_msg_ids)

                 # for msg_id in inbound_msg_ids:
                 #    self.continuation_executor.submit(continuation_fn, msg_id)

        except Exception as e:
            print("Error in response service: {}".format(e))

class DriverBenchmarker(object):
    def __init__(self, trial_length, configs):
        self.trial_length = trial_length
        self.configs = configs

    def run(self, lambda_val, num_trials, arrival_process):
        logger.info("Creating clients!")
        clients = create_clients(self.configs)

        time.sleep(60)

        logger.info("Starting predictions")
        predictor = Predictor(trial_length=self.trial_length, clients=clients)

        num_queries_sent = 0

        last_catchup_time = datetime.now()
        last_catchup_msg_id = 0

        msg_id = 0
        while msg_id < len(arrival_process):
        # for msg_id in range(len(arrival_process)):
            # if False:  
            if msg_id % 1000 == 0:  
                curr_time = datetime.now()
                elapsed_time = (curr_time - last_catchup_time).total_seconds()
                expected_num_queries = int(lambda_val * elapsed_time)
                last_catchup_msg_id += expected_num_queries
                while msg_id < min(last_catchup_msg_id, len(arrival_process) - 1):
                    predictor.predict(msg_id)
                    msg_id += 1
                last_catchup_time = datetime.now()
            else:
                predictor.predict(msg_id)

            request_delay = arrival_process[msg_id] * .001
            time.sleep(request_delay)
            # time.sleep(.00200)

            msg_id += 1

        return predictor.stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Clipper image driver 1')
    parser.add_argument('-w', '--warmup', action='store_true')
    parser.add_argument('-e', '--experiment_config_path', type=str)
    parser.add_argument('-r', '--resnet_extras', type=str, nargs='+', help="Addresses of extra resnet replicas to add, in host:port format")
    parser.add_argument('-i', '--inception_extras', type=str, nargs='+', help="Addresses of extra inception replicas to add, in host:port format")
    parser.add_argument('-k', '--ksvm_extras', type=str, nargs='+', help="Addresses of extra kernel svm replicas to add, in host:port format")
    parser.add_argument('-l', '--log_reg_extras', type=str, nargs='+', help="Addresses of extra log reg replicas to add, in host:port format")

    args = parser.parse_args()

    experiment_config = load_experiment_config(args.experiment_config_path, 
                                               args.resnet_extras, 
                                               args.inception_extras,
                                               args.ksvm_extras,
                                               args.log_reg_extras)

    if args.warmup:
        benchmarker = DriverBenchmarker(trial_length=experiment_config.trial_length, 
                                        configs=experiment_config.client_configs)
        benchmarker.warm_up()

    else:
        arrival_process = tfs_utils.load_arrival_deltas(experiment_config.process_path)
        mean_throughput = tfs_utils.calculate_mean_throughput(arrival_process)
        peak_throughput = tfs_utils.calculate_peak_throughput(arrival_process)

        print("Mean throughput: {}\nPeak throughput: {}".format(mean_throughput, peak_throughput))
        
        benchmarker = DriverBenchmarker(trial_length=experiment_config.trial_length, 
                                        configs=experiment_config.client_configs)
        stats = benchmarker.run(experiment_config.lambda_val, 
                                experiment_config.num_trials, 
                                arrival_process)
        all_stats = [stats]

        # Save Results
        results_dir_path = "/".join(experiment_config.process_path.split("/")[:-1])
        fname = "results_{slo}_slo_bs_1.json".format(slo=experiment_config.slo_millis)
        tfs_utils.save_results(node_configs=experiment_config.node_configs, 
                               client_metrics=all_stats,
                               prefix=fname,
                               results_dir=results_dir_path,
                               slo_millis=experiment_config.slo_millis, 
                               cv=experiment_config.cv,
                               lambda_val=experiment_config.lambda_val,
                               arrival_process=experiment_config.process_path)
        sys.exit(0)
