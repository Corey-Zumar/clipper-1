import sys
import os
import time
import logging
import tensorflow as tf
import subprocess
import requests
import numpy as np
import json

from datetime import datetime

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

# Change this if necessary
TFS_BASE_PATH = "~/tfserving"

MODEL_SERVER_RELATIVE_PATH = "bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server"

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

class TFSHeavyNodeConfig(object):
    def __init__(self,
                 name,
                 model_base_path,
                 ports,
                 input_type,
                 allocated_cpus,
                 cpus_per_replica,
                 num_replicas,
                 gpus,
                 batch_size):
        assert len(ports) >= num_replicas
        assert len(allocated_cpus) >= cpus_per_replica * num_replicas

        self.name = name
        self.model_base_path = model_base_path
        self.ports = ports
        self.input_type = input_type
        self.allocated_cpus = allocated_cpus
        self.cpus_per_replica = cpus_per_replica
        self.num_replicas = num_replicas
        self.gpus = gpus
        self.batch_size = batch_size
        self.instance_type = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-type").text
        if len(gpus) == 0:
            self.gpus_per_replica = 0
        else:
            self.gpus_per_replica = 1

    def to_json(self):
        return json.dumps(self.__dict__)

def save_results(configs, client_metrics, results_dir, prefix="results", slo_millis=250, arrival_process=None):
    """
    Parameters
    ----------
    configs : list(HeavyNodeConfig)
       The configs for any models deployed
    arrival_process : str (optional)
        Path to an arrival process file used to generate experimental request
        delays
    """

    results_dir = os.path.abspath(os.path.expanduser(results_dir))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logger.info("Created experiments directory: %s" % results_dir)

    results_obj = {
        "node_configs": [c.__dict__ for c in configs],
        "client_metrics": client_metrics,
    }

    if arrival_process is not None:
        arrival_deltas = load_arrival_deltas(arrival_process)
        mean_throughput = calculate_mean_throughput(arrival_deltas)
        peak_throughput = calculate_peak_throughput(arrival_deltas, slo_millis)

        results_obj["arrival_process"] = {
            "filename" : arrival_process,
            "mean_thru" : mean_throughput,
            "peak_thru" : peak_throughput
        }

    results_file = os.path.join(results_dir, "{prefix}-{ts:%y%m%d_%H%M%S}.json".format(
        prefix=prefix, ts=datetime.now()))
    with open(results_file, "w") as f:
        json.dump(results_obj, f, indent=4)
        logger.info("Saved results to {}".format(results_file))

def create_predict_request(model_name, data, signature_name="predict_inputs"):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name
    wrapped_data = np.array([data])
    request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(wrapped_data, shape=wrapped_data.shape))
    return request

def parse_predict_response(response, output_key):
    outputs_tensor = response.outputs[output_key]
    return tf.contrib.util.make_ndarray(outputs_tensor)

def setup_heavy_node(config):
    gpus = list(config.gpus)
    cpus = list(config.allocated_cpus)
    ports = list(config.ports)

    for _ in range(config.num_replicas):
        node_gpu = None
        if len(gpus) > 0:
            node_gpu = gpus.pop()


        node_cpus = []
        for _ in range(config.cpus_per_replica):
            node_cpus.append(cpus.pop())

        port_number = ports.pop()

        _start_serving(config, port_number, node_cpus, node_gpu)

def _start_serving(config, port_number, cpus, gpu_number=None):
    model_server_path = os.path.join(TFS_BASE_PATH, MODEL_SERVER_RELATIVE_PATH)

    batching_params = _get_batching_params(config.batch_size)
    batching_params_path = os.path.join("/tmp", "batching_params-{ts:%y%m%d_%H%M%S}.json".format(ts=datetime.now()))

    batching_params_file = open(batching_params_path, "w")
    batching_params_file.write(batching_params)
    batching_params_file.close()

    cpus_str = ",".join(["%d"] * len(cpus)) % tuple(cpus)
    cmd_filter_cpus = "numactl -C {cpus}".format(cpus=cpus_str)

    if gpu_number is not None:
        cmd_filter_gpus = "export CUDA_VISIBLE_DEVICES={gpu}".format(gpu=gpu_number)
    else:
        cmd_filter_gpus = "export CUDA_VISIBLE_DEVICES=''"

    cmd_serve = ("{cf} {msp} --enable_batching \\\n"
                "--port={pn} \\\n"
                "--model_name={mn} \\\n"
                "--model_base_path={mbp} \\\n"
                "--batching_parameters_file={bpf}").format(cf=cmd_filter_cpus,
                                                           msp=model_server_path,
                                                           pn=port_number,
                                                           mn=config.name,
                                                           mbp=config.model_base_path,
                                                           bpf=batching_params_path)

    full_cmd = "({cf};{cs}) &".format(cf=cmd_filter_gpus, cs=cmd_serve)

    print("Starting node! model name: {mn} port: {pn} gpu_num: {gpu} cpus: {cpus} \n\n\n"
        .format(mn=config.name,
                pn=port_number,
                gpu=gpu_number,
                cpus=cpus_str))

    subprocess.call(full_cmd, shell=True)


def _get_batching_params(max_batch_size, batch_timeout_micros=10000, max_enqueued_batches=200):
    batching_params_text = ("max_batch_size {{ value : {mbs} }} \n"
                            "batch_timeout_micros {{ value : {btm} }} \n"
                            "max_enqueued_batches {{ value : {meb} }} \n"
                            "num_batch_threads {{ value : {nbt} }} \n")

    num_batch_threads = max_batch_size * 2

    formatted_params = batching_params_text.format(mbs=max_batch_size,
                                                   btm=batch_timeout_micros,
                                                   meb=max_enqueued_batches,
                                                   nbt=num_batch_threads)

    return formatted_params
