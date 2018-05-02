import sys
# import os
# import argparse
import numpy as np
import time
# import base64
import logging

from clipper_admin import ClipperConnection, DockerContainerManager
from datetime import datetime
# from io import BytesIO
# from PIL import Image
from containerized_utils.zmq_client import Client
from containerized_utils import driver_utils
from multiprocessing import Process, Queue
# import json
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
        redis_cpu_str="0",
        mgmt_cpu_str="0",
        query_cpu_str="4")
        # query_cpu_str="1-4")
    time.sleep(10)
    for config in configs:
        driver_utils.setup_heavy_node(cl, config, DEFAULT_OUTPUT)
    time.sleep(10)
    logger.info("Clipper is set up!")
    return config


def setup_alexnet(batch_size,
                  num_replicas,
                  cpus_per_replica,
                  allocated_cpus,
                  allocated_gpus):

    return driver_utils.HeavyNodeConfig(name="alexnet",
                                        input_type="floats",
                                        model_image="model-comp/pytorch-alexnet",
                                        allocated_cpus=allocated_cpus,
                                        cpus_per_replica=cpus_per_replica,
                                        gpus=allocated_gpus,
                                        batch_size=batch_size,
                                        num_replicas=num_replicas,
                                        use_nvidia_docker=True)


def setup_res50(batch_size,
                num_replicas,
                cpus_per_replica,
                allocated_cpus,
                allocated_gpus):
        return driver_utils.HeavyNodeConfig(name="res50",
                                            input_type="floats",
                                            model_image="model-comp/pytorch-res50",
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True)


def setup_res152(batch_size,
                 num_replicas,
                 cpus_per_replica,
                 allocated_cpus,
                 allocated_gpus):
        return driver_utils.HeavyNodeConfig(name="res152",
                                            input_type="floats",
                                            model_image="model-comp/pytorch-res152",
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True)


class Predictor(object):

    def __init__(self, config):
        self.outstanding_reqs = {}
        self.client = Client(CLIPPER_ADDRESS, CLIPPER_SEND_PORT, CLIPPER_RECV_PORT)
        self.client.start()
        self.init_stats()
        self.stats = {
            "thrus": [],
            "all_lats": [],
            "p99_lats": [],
            "p95_lats": [],
            "mean_lats": []}
        self.total_num_complete = 0
        self.config = config

    def init_stats(self):
        self.latencies = []
        self.batch_num_complete = 0
        self.cur_req_id = 0
        self.start_time = datetime.now()

    def print_stats(self):
        lats = np.array(self.latencies)
        p99 = np.percentile(lats, 99)
        p95 = np.percentile(lats, 95)
        mean = np.mean(lats)
        end_time = datetime.now()
        thru = float(self.batch_num_complete) / (end_time - self.start_time).total_seconds()
        self.stats["thrus"].append(thru)
        self.stats["all_lats"].append(lats.tolist())
        self.stats["p99_lats"].append(p99)
        self.stats["p95_lats"].append(p95)
        self.stats["mean_lats"].append(mean)
        logger.info("p99: {p99}, p95: {p95}, mean: {mean}, thruput: {thru}".format(p99=p99,
                                                                                   p95=p95,
                                                                                   mean=mean,
                                                                                   thru=thru))

    def predict(self, input_item, model_name):
        begin_time = datetime.now()

        def complete(output):
            if output == DEFAULT_OUTPUT:
                return
            else:
                end_time = datetime.now()
                latency = (end_time - begin_time).total_seconds()
                self.latencies.append(latency)
                self.total_num_complete += 1
                self.batch_num_complete += 1
                if self.batch_num_complete % (self.config.batch_size * 7) == 0:
                    self.print_stats()
                    self.init_stats()

        return self.client.send_request(model_name, input_item).then(complete)


class ModelBenchmarker(object):
    def __init__(self, queue, delay, config):
        self.queue = queue
        self.delay = delay
        self.config = config

    def run(self):
        logger.info("Generating random inputs")
        base_inputs = [np.array(np.random.rand(299*299*3), dtype=np.float32) for _ in range(1000)]
        inputs = [i for _ in range(50) for i in base_inputs]
        logger.info("Starting predictions")
        predictor = Predictor(self.config)
        i = 0
        for input_item in inputs:
            predictor.predict(input_item, self.config.name)
            # if i % 2 == 0:
            time.sleep(self.delay)
            if len(predictor.stats["thrus"]) > 30:
                break
            i += 1
        self.queue.put(predictor.stats)
        return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--delay', type=float, help='inter-request delay')
    # parser.add_argument('-m', '--model', type=str, help='Model name. alexnet, res50, or res152')

    args = parser.parse_args()

    queue = Queue()

    total_cpus = range(9, 16)

    def get_cpus(num_cpus):
        return [total_cpus.pop() for _ in range(num_cpus)]

    total_gpus = range(8)

    def get_gpus(num_gpus):
        return [total_gpus.pop() for _ in range(num_gpus)]

    alexnet_reps = 3
    res50_reps = 1
    res152_reps = 1

    alex_batch = 128
    res50_batch = 30
    res152_batch = 30

    config = setup_alexnet(batch_size=alex_batch,
                           num_replicas=alexnet_reps,
                           cpus_per_replica=1,
                           allocated_cpus=get_cpus(alexnet_reps),
                           allocated_gpus=get_gpus(alexnet_reps))
    # setup_res50(batch_size=res50_batch,
    #             num_replicas=res50_reps,
    #             cpus_per_replica=1,
    #             allocated_cpus=get_cpus(6),
    #             allocated_gpus=get_gpus(res50_reps)),
    # setup_res152(batch_size=res152_batch,
    #              num_replicas=res152_reps,
    #              cpus_per_replica=1,
    #              allocated_cpus=get_cpus(4),
    #              allocated_gpus=get_gpus(res152_reps))

    setup_clipper([config, ])
    benchmarker = ModelBenchmarker(queue, args.delay, config)

    all_stats = []
    p = Process(target=benchmarker.run)
    p.start()
    all_stats.append(queue.get())
    # p.join()

    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.connect()
    fname = "batch_{}".format(config.batch_size)
    driver_utils.save_results([config],
                              cl,
                              all_stats,
                              "single-model-prof-{}".format(config.name),
                              prefix=fname)
    sys.exit(0)
    # driver_utils.save_results(configs, cl, all_stats, "e2e_min_lat_resnet-cascade_DEBUG")
