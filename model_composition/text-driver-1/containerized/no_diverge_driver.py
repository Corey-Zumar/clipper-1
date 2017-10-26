import sys
import os
import numpy as np
import time
import logging

from clipper_admin import ClipperConnection, DockerContainerManager
from datetime import datetime
from containerized_utils.zmq_client import Client
from containerized_utils import driver_utils
from containerized_utils.driver_utils import (INCREASING, DECREASING, CONVERGED_HIGH,
                                              CONVERGED, UNKNOWN)
from multiprocessing import Process, Queue
import json

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

# Models and applications for each heavy node
# will share the same name
AUTOCOMPLETION_MODEL_APP_NAME = "autocompletion"
LSTM_MODEL_APP_NAME = "lstm"
NMT_MODEL_APP_NAME = "nmt"

AUTOCOMPLETION_IMAGE_NAME = "model-comp/tf-autocomplete"
LSTM_IMAGE_NAME = "model-comp/theano-lstm"
NMT_IMAGE_NAME = "model-comp/nmt"


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


def setup_nmt(batch_size,
              num_replicas,
              cpus_per_replica,
              allocated_cpus,
              allocated_gpus,
              input_size):

        return driver_utils.HeavyNodeConfig(name=NMT_MODEL_APP_NAME,
                                            input_type="bytes",
                                            model_image=NMT_IMAGE_NAME,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            input_size=input_size,
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
            if self.batch_num_complete % 100 == 0:
                self.print_stats()
                self.init_stats()

        def continuation(output):
            if output == DEFAULT_OUTPUT:
                return
            else:
                complete()

        return self.client.send_request("nmt", np.frombuffer(
            bytearray(input_item), dtype=np.int8)).then(continuation)


def load_german_docs():
    german_data_path = os.path.join(CURR_DIR, "nmt_workload", "german_text.de")
    german_data_file = open(german_data_path, "rb")
    german_text = german_data_file.readlines()
    np.random.shuffle(german_text)
    return german_text


def gen_german_inputs(num_inputs=5000, input_size=20):
    german_text = load_german_docs()

    inputs = []
    num_gen_inputs = 0
    while num_gen_inputs < num_inputs:
        idx = np.random.randint(len(german_text))
        text = german_text[idx]
        words = text.split()
        if len(words) > input_size:
            words = words[:input_size]
            inputs.append(" ".join(words))
            num_gen_inputs += 1

    return inputs


class ModelBenchmarker(object):
    def __init__(self, configs, queue, client_num, latency_upper_bound, input_size):
        self.configs = configs
        self.queue = queue
        assert client_num == 0
        self.client_num = client_num
        logger.info("Generating random inputs")
        base_inputs = gen_german_inputs(input_size=input_size)
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
        while len(predictor.stats["thrus"]) < 4:
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
                convergence_state = driver_utils.check_convergence(predictor.stats,
                                                                   self.configs,
                                                                   self.latency_upper_bound)
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
    queue = Queue()
    input_size = 20
    latency_upper_bound = None
    for batch_size in [1, 2, 4, 6, 8, 10, 15, 20, 25]:
        total_cpus = range(9, 32)

        def get_cpus(num_cpus):
            return [total_cpus.pop() for _ in range(num_cpus)]

        total_gpus = range(8)

        def get_gpus(num_gpus):
            return [total_gpus.pop() for _ in range(num_gpus)]

        configs = [
            setup_nmt(batch_size=batch_size,
                      num_replicas=1,
                      cpus_per_replica=1,
                      allocated_cpus=get_cpus(1),
                      allocated_gpus=get_gpus(1),
                      input_size=input_size),
        ]

        client_num = 0
        benchmarker = ModelBenchmarker(configs, queue, client_num, latency_upper_bound, input_size)
        p = Process(target=benchmarker.run)
        p.start()

        all_stats = []
        all_stats.append(queue.get())
        p.join()

        cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        cl.connect()
        fname = "input-size_{}".format(input_size)
        driver_utils.save_results(configs, cl, all_stats, "no-diverge-nmt-profile", prefix=fname)
    sys.exit(0)
