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
    # cl.connect()
    # print(get_batch_sizes(cl.inspect_instance()))
    cl.stop_all()
    cl.start_clipper(
        query_frontend_image="clipper/zmq_frontend:develop",
        redis_cpu_str="4",
        mgmt_cpu_str="4",
        query_cpu_str="5,6,7,21,22,23")
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
                                        # model_image="model-comp/pytorch-alexnet",
                                        model_image="model-comp/pytorch-alex-sleep",
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


class Predictor(object):

    def __init__(self, clipper_metrics):
        self.outstanding_reqs = {}
        self.client = Client(CLIPPER_ADDRESS, CLIPPER_SEND_PORT, CLIPPER_RECV_PORT)
        self.client.start()
        self.init_stats()
        self.stats = {
            "thrus": [],
            "all_lats": [],
            "p99_lats": [],
            "mean_lats": [],
        }
        self.total_num_complete = 0
        self.cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        self.cl.connect()
        self.get_clipper_metrics = clipper_metrics
        if self.get_clipper_metrics:
            self.stats["all_metrics"] = []
            self.stats["mean_batch_sizes"] = []

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
        self.stats["all_lats"].append(lats.tolist())
        self.stats["p99_lats"].append(p99)
        self.stats["mean_lats"].append(mean)
        if self.get_clipper_metrics:
            metrics = self.cl.inspect_instance()
            batch_sizes = get_batch_sizes(metrics)
            self.stats["mean_batch_sizes"].append(batch_sizes)
            self.stats["all_metrics"].append(metrics)
            logger.info(("p99: {p99}, mean: {mean}, thruput: {thru}, "
                         "batch_sizes: {batches}").format(p99=p99, mean=mean, thru=thru,
                                                          batches=json.dumps(
                                                              batch_sizes, sort_keys=True)))
        else:
            logger.info("p99: {p99}, mean: {mean}, thruput: {thru}".format(p99=p99,
                                                                           mean=mean,
                                                                           thru=thru))

    def predict(self, input_item):
        begin_time = datetime.now()

        def complete():
            end_time = datetime.now()
            latency = (end_time - begin_time).total_seconds()
            self.latencies.append(latency)
            self.total_num_complete += 1
            self.batch_num_complete += 1
            if self.batch_num_complete % 500 == 0:
                self.print_stats()
                self.init_stats()

        def res152_cont(output):
            if output == DEFAULT_OUTPUT:
                return
            else:
                complete()

        def res50_cont(output):
            if output == DEFAULT_OUTPUT:
                return
            else:
                idk = np.random.random() > 0.4633
                # idk = True
                if idk:
                    self.client.send_request("res152", input_item).then(res152_cont)
                else:
                    complete()

        def alex_cont(output):
            if output == DEFAULT_OUTPUT:
                return
            else:
                # idk = np.random.random() > 0.192
                idk = False
                if idk:
                    self.client.send_request("res50", input_item).then(res50_cont)
                else:
                    complete()

        return self.client.send_request("alexnet", input_item).then(alex_cont)


class ModelBenchmarker(object):
    def __init__(self, queue, delay, client_num):
        self.queue = queue
        self.delay = delay
        self.client_num = client_num

    def run(self):
        logger.info("Generating random inputs")
        base_inputs = [np.array(np.random.rand(299*299*3), dtype=np.float32) for _ in range(1000)]
        inputs = [i for _ in range(50) for i in base_inputs]
        logger.info("Starting predictions")
        if self.client_num == 0:
            predictor = Predictor(clipper_metrics=True)
        else:
            predictor = Predictor(clipper_metrics=False)
        i = 0
        while len(predictor.stats["thrus"]) < 8:
            input_item = inputs[i % len(inputs)]
            predictor.predict(input_item=input_item)
            # if i % 2 == 0:
            time.sleep(self.delay)
            # if len(predictor.stats["thrus"]) > 20:
            #     break
            i += 1
        self.queue.put(predictor.stats)
        print("DONE")
        return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--delay', type=float, help='inter-request delay')
    parser.add_argument('-c', '--num_clients', type=int, help='number of clients')

    args = parser.parse_args()

    queue = Queue()

    # total_cpus = list(reversed(range(12, 32)))
    total_cpus = range(8, 16) + range(24, 32)

    def get_cpus(num_cpus):
        return [total_cpus.pop() for _ in range(num_cpus)]

    total_gpus = range(8)

    def get_gpus(num_gpus):
        return [total_gpus.pop() for _ in range(num_gpus)]

    alexnet_reps = 5
    res50_reps = 1
    res152_reps = 1

    alex_batch = 30
    res50_batch = 30
    res152_batch = 30

    configs = [
        setup_alexnet(batch_size=alex_batch,
                      num_replicas=alexnet_reps,
                      cpus_per_replica=1,
                      allocated_cpus=get_cpus(8),
                      allocated_gpus=get_gpus(alexnet_reps)),
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
    procs = []
    for i in range(args.num_clients):
        benchmarker = ModelBenchmarker(queue, args.delay, i)
        p = Process(target=benchmarker.run)
        p.start()
        procs.append(p)

    all_stats = []
    for i in range(args.num_clients):
        all_stats.append(queue.get())
    # p.join()

    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.connect()
    # fname = "alex_{}-r50_{}-r152_{}".format(alexnet_reps, res50_reps, res152_reps)
    # driver_utils.save_results(configs, cl, all_stats, "e2e_max_thru_resnet-cascade", prefix=fname)
    fname = "{clients}_clients-{alex_reps}_reps".format(clients=args.num_clients,
                                                        alex_reps=alexnet_reps)
    driver_utils.save_results(configs, cl, all_stats, "alexnet-thruput-replicas", prefix=fname)
    sys.exit(0)
