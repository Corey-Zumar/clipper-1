# import sys
# import os
# import argparse
# import numpy as np
import time
# import base64
import logging

from clipper_admin import ClipperConnection, DockerContainerManager
# from datetime import datetime
# from io import BytesIO
# from PIL import Image
# from containerized_utils.zmq_client import Client
from containerized_utils import driver_utils
# from multiprocessing import Process, Queue
import json
import argparse

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

# Models and applications for each heavy node
# will share the same name
RES50 = "res50"
RES152 = "res152"
ALEXNET = "alexnet"


CLIPPER_ADDRESS = "localhost"
CLIPPER_SEND_PORT = 4456
CLIPPER_RECV_PORT = 4455

DEFAULT_OUTPUT = "TIMEOUT"


def setup_clipper(configs):
    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.stop_all()
    cl.start_clipper(
        query_frontend_image="clipper/zmq_frontend:develop",
        redis_cpu_str="4",
        mgmt_cpu_str="4",
        query_cpu_str="5-20")
    time.sleep(10)
    for config in configs:
        driver_utils.setup_heavy_node(cl, config, DEFAULT_OUTPUT)
    logger.info("Clipper is set up!")
    return config


def setup_noop(batch_size,
               num_replicas,
               cpus_per_replica,
               allocated_cpus):

    return driver_utils.HeavyNodeConfig(name="noop",
                                        input_type="floats",
                                        # model_image="model-comp/pytorch-alexnet",
                                        model_image="clipper/noop-container:develop",
                                        allocated_cpus=allocated_cpus,
                                        cpus_per_replica=cpus_per_replica,
                                        gpus=[],
                                        batch_size=batch_size,
                                        num_replicas=num_replicas,
                                        use_nvidia_docker=False)


def get_batch_sizes(metrics_json):
    hists = metrics_json["histograms"]
    mean_batch_sizes = {}
    for h in hists:
        if "batch_size" in h.keys()[0]:
            name = h.keys()[0]
            model = name.split(":")[1]
            mean = h[name]["mean"]
            mean_batch_sizes[model] = round(float(mean), 2)
    return mean_batch_sizes


def get_lock_latencies(metrics_json):
    hists = metrics_json["histograms"]
    mean_lock_latencies = {}
    for h in hists:
        if "lock_latency" in h.keys()[0]:
            name = h.keys()[0]
            model = name.split(":")[1]
            mean = h[name]["mean"]
            # mean_lock_latencies[model] = round(float(mean), 2)
            mean_lock_latencies[model] = mean
    return mean_lock_latencies


def get_queue_submit_latencies(metrics_json):
    hists = metrics_json["histograms"]
    mean_lock_latencies = {}
    for h in hists:
        if "queue_submit_latency" in h.keys()[0]:
            name = h.keys()[0]
            queue_name = name.split(":")[0]
            mean = h[name]["mean"]
            # mean_lock_latencies[model] = round(float(mean), 2)
            mean_lock_latencies[queue_name] = mean
    return mean_lock_latencies


def get_request_rate(metrics_json):
    meters = metrics_json["meters"]
    for m in meters:
        if "request_rate" in m.keys()[0]:
            name = m.keys()[0]
            return m[name]["rate"]


def get_throughput(metrics_json):
    meters = metrics_json["meters"]
    for m in meters:
        if "model:noop:1:prediction_throughput" in m.keys()[0]:
            name = m.keys()[0]
            return m[name]["rate"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--delay', type=float, help='inter-request delay')
    # parser.add_argument('-c', '--num_clients', type=int, help='number of clients')
    parser.add_argument('-n', '--num_replicas', type=int, help='number of container replicas')

    args = parser.parse_args()

    # total_cpus = list(reversed(range(12, 32)))
    # total_cpus = range(8, 16) + range(24, 32)

    # def get_cpus(num_cpus):
    #     return [total_cpus.pop() for _ in range(num_cpus)]

    total_gpus = range(8)

    def get_gpus(num_gpus):
        return [total_gpus.pop() for _ in range(num_gpus)]

    noop_reps = args.num_replicas
    noop_batch = 30

    # alexnet_reps = 4
    # res50_reps = 1
    # res152_reps = 1
    #
    # alex_batch = 30
    # res50_batch = 30
    # res152_batch = 30

    configs = [
        setup_noop(batch_size=noop_batch,
                   num_replicas=noop_reps,
                   cpus_per_replica=1,
                   allocated_cpus=range(24, 32))
        # setup_alexnet(batch_size=alex_batch,
        #               num_replicas=alexnet_reps,
        #               cpus_per_replica=1,
        #               allocated_cpus=get_cpus(8),
        #               allocated_gpus=get_gpus(res50_reps)),
        # setup_res50(batch_size=res50_batch,
        #             num_replicas=res50_reps,
        #             cpus_per_replica=1,
        #             allocated_cpus=get_cpus(4),
        #             allocated_gpus=get_gpus(res50_reps)),
        # setup_res152(batch_size=res152_batch,
        #              num_replicas=res152_reps,
        #              cpus_per_replica=1,
        #              allocated_cpus=get_cpus(4),
        #              allocated_gpus=get_gpus(res152_reps))
    ]

    setup_clipper(configs)

    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.connect()
    while True:
        time.sleep(10)
        metrics = cl.inspect_instance()
        request_rate = get_request_rate(metrics)
        throughput = get_throughput(metrics)
        batch_sizes = get_batch_sizes(metrics)
        queue_submit_lats = get_queue_submit_latencies(metrics)
        logger.info(("\n\nrequest_rate: {rr}"
                     "\nthroughput: {thru}"
                     "\nbatch_sizes: {batches}"
                     "\nsubmit_lats: {submit_lats}\n\n").format(
                         rr=request_rate,
                         thru=throughput,
                         batches=json.dumps(
                             batch_sizes, sort_keys=True),
                         submit_lats=json.dumps(
                             queue_submit_lats, sort_keys=True)
                     ))
