import sys
# import os
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
from containerized_utils.driver_utils import INCREASING, DECREASING, CONVERGED_HIGH, CONVERGED, UNKNOWN
from multiprocessing import Process, Queue
import json
# import argparse

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
        redis_cpu_str="0",
        mgmt_cpu_str="0",
        query_cpu_str="1-8")
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
                                        use_nvidia_docker=True,
                                        no_diverge=True,
                                        )


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
                                        use_nvidia_docker=True,
                                        no_diverge=True,
                                        )


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
                                        use_nvidia_docker=True,
                                        no_diverge=True,
                                        )


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
                idk = np.random.random() > 0.192
                if idk:
                    self.client.send_request("res50", input_item).then(res50_cont)
                else:
                    complete()

        return self.client.send_request("alexnet", input_item).then(alex_cont)


class ModelBenchmarker(object):
    def __init__(self, configs, queue, client_num, latency_upper_bound):
        self.configs = configs
        self.queue = queue
        assert client_num == 0
        self.client_num = client_num
        logger.info("Generating random inputs")
        base_inputs = [np.array(np.random.rand(299*299*3), dtype=np.float32) for _ in range(1000)]
        self.inputs = [i for _ in range(60) for i in base_inputs]
        self.latency_upper_bound = latency_upper_bound

    # start with an overly aggressive request rate
    # then back off
    def initialize_request_rate(self):
        # initialize delay to be very small
        self.delay = 0.001
        setup_clipper(self.configs)
        time.sleep(5)
        predictor = Predictor(clipper_metrics=True)
        idx = 0
        while len(predictor.stats["thrus"]) < 6:
            predictor.predict(input_item=self.inputs[idx])
            time.sleep(self.delay)
            idx += 1
            idx = idx % len(self.inputs)

        max_thruput = np.mean(predictor.stats["thrus"][1:])
        self.delay = 1.0 / max_thruput
        logger.info("Initializing delay to {}".format(self.delay))

    def increase_delay(self):
        if self.delay < 0.005:
            self.delay += 0.0002
        elif self.delay < 0.01:
            self.delay += 0.0005
        else:
            self.delay += 0.001


    def find_steady_state(self):
        setup_clipper(self.configs)
        time.sleep(7)
        predictor = Predictor(clipper_metrics=True)
        idx = 0
        done = False
        # start checking for steady state after 7 trials
        last_checked_length = 6
        while not done:
            predictor.predict(input_item=self.inputs[idx])
            time.sleep(self.delay)
            idx += 1
            idx = idx % len(self.inputs)

            if len(predictor.stats["thrus"]) > last_checked_length:
                last_checked_length = len(predictor.stats["thrus"]) + 4
                convergence_state = driver_utils.check_convergence(predictor.stats, self.configs, self.latency_upper_bound)
                # Diverging, try again with higher
                # delay
                if convergence_state == INCREASING or convergence_state == CONVERGED_HIGH:
                    self.increase_delay()
                    logger.info("Increasing delay to {}".format(self.delay))
                    done = True
                    return self.find_steady_state()
                elif convergence_state == CONVERGED:
                    logger.info("Converged with delay of {}".format(self.delay))
                    done = True
                    self.queue.put(predictor.stats)
                    return
                elif len(predictor.stats) > 100:
                    self.increase_delay()
                    logger.info("Increasing delay to {}".format(self.delay))
                    done = True
                    return self.find_steady_state()
                elif convergence_state == DECREASING or convergence_state == UNKNOWN:
                    logger.info("Not converged yet. Still waiting")
                else:
                    logger.error("Unknown convergence state: {}".format(convergence_state))
                    sys.exit(1)

    def run(self):
        self.initialize_request_rate()
        self.find_steady_state()
        return


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--delay', type=float, help='inter-request delay')
    # parser.add_argument('-c', '--num_clients', type=int, help='number of clients')
    #
    # args = parser.parse_args()

    queue = Queue()

    alex_batch = 1
    res50_batch = 1
    res152_batch = 1

    latency_upper_bound = 0.300

    # alexnet_reps = 1
    # res50_reps = 1
    # res152_reps = 1
    reps = [(1, 1, 1),
            (1, 1, 2),
            (1, 2, 2),
            (2, 2, 2),
            (2, 2, 3),
            (2, 3, 3)]

    for alexnet_reps, res50_reps, res152_reps in reps:
        total_cpus = range(9, 29)

        def get_cpus(num_cpus):
            return [total_cpus.pop() for _ in range(num_cpus)]

        total_gpus = range(8)

        def get_gpus(num_gpus):
            return [total_gpus.pop() for _ in range(num_gpus)]

        configs = [
            setup_alexnet(batch_size=alex_batch,
                          num_replicas=alexnet_reps,
                          cpus_per_replica=1,
                          allocated_cpus=get_cpus(alexnet_reps),
                          allocated_gpus=get_gpus(alexnet_reps)),
            setup_res50(batch_size=res50_batch,
                        num_replicas=res50_reps,
                        cpus_per_replica=1,
                        allocated_cpus=get_cpus(res50_reps),
                        allocated_gpus=get_gpus(res50_reps)),
            setup_res152(batch_size=res152_batch,
                         num_replicas=res152_reps,
                         cpus_per_replica=1,
                         allocated_cpus=get_cpus(res152_reps),
                         allocated_gpus=get_gpus(res152_reps))
        ]

        client_num = 0
        benchmarker = ModelBenchmarker(configs, queue, client_num, latency_upper_bound)
        p = Process(target=benchmarker.run)
        p.start()

        all_stats = []
        all_stats.append(queue.get())
        p.join()

        cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        cl.connect()
        fname = "alex_{}-r50_{}-r152_{}".format(alexnet_reps, res50_reps, res152_reps)
        # driver_utils.save_results(configs, cl, all_stats, "e2e_max_thru_resnet-cascade", prefix=fname)
        driver_utils.save_results(configs, cl, all_stats, "e2e_min_lat_resnet-cascade", prefix=fname)
        # driver_utils.save_results(configs, cl, all_stats, "e2e_500_slo_resnet-cascade", prefix=fname)
        # driver_utils.save_results(configs, cl, all_stats,
        #                           "e2e_alex_no_gpu_resnet-cascade", prefix=fname)
    sys.exit(0)
