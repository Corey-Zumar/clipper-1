import sys
import os
import argparse
import numpy as np
import time
import base64
import logging
import json
import math

from clipper_admin import ClipperConnection, DockerContainerManager
from datetime import datetime
from containerized_utils.zmq_client import Client
from containerized_utils import driver_utils
from multiprocessing import Process, Queue

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

# Models and applications for each heavy node
# will share the same name
LANG_DETECT_MODEL_APP_NAME = "tf-lang-detect"
NMT_MODEL_APP_NAME = "tf-nmt"
LSTM_MODEL_APP_NAME = "tf-lstm"

LANG_DETECT_IMAGE_NAME = "model-comp/tf-lang-detect"
NMT_IMAGE_NAME = "model-comp/tf-nmt"
LSTM_IMAGE_NAME = "model-comp/tf-lstm"

VALID_MODEL_NAMES = [
    LANG_DETECT_MODEL_APP_NAME,
    NMT_MODEL_APP_NAME,
    LSTM_MODEL_APP_NAME
]

LANG_DETECT_WORKLOAD_RELATIVE_PATH = "lang_detect_workload"
NMT_WORKLOAD_RELATIVE_PATH = "nmt_workload"
LSTM_WORKLOAD_RELATIVE_PATH = "lstm_workload"


CLIPPER_ADDRESS = "localhost"
CLIPPER_SEND_PORT = 4456
CLIPPER_RECV_PORT = 4455

DEFAULT_OUTPUT = "TIMEOUT"

########## Setup ##########

def setup_clipper(config):
    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.stop_all()
    cl.start_clipper(
        query_frontend_image="clipper/zmq_frontend:develop",
        redis_cpu_str="0",
        mgmt_cpu_str="0",
        query_cpu_str="1-4")
    time.sleep(10)
    driver_utils.setup_heavy_node(cl, config, DEFAULT_OUTPUT)
    time.sleep(10)
    logger.info("Clipper is set up!")
    return config

def get_heavy_node_config(model_name,
                          batch_size,
                          num_replicas,
                          cpus_per_replica,
                          allocated_cpus,
                          allocated_gpus,
                          input_size):
    if model_name == LANG_DETECT_MODEL_APP_NAME:
        return driver_utils.HeavyNodeConfig(name=LANG_DETECT_MODEL_APP_NAME,
                                            input_type="bytes",
                                            model_image=LANG_DETECT_IMAGE_NAME,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=[0],
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            input_size=input_size,
                                            no_diverge=True
                                            )

    elif model_name == NMT_MODEL_APP_NAME:
        return driver_utils.HeavyNodeConfig(name=NMT_MODEL_APP_NAME,
                                            input_type="bytes",
                                            model_image=NMT_IMAGE_NAME,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=[0],
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            input_size=input_size,
                                            no_diverge=True
                                            )

    elif LSTM_MODEL_APP_NAME == LSTM_MODEL_APP_NAME:
        return driver_utils.HeavyNodeConfig(name=LSTM_MODEL_APP_NAME,
                                    input_type="bytes",
                                    model_image=LSTM_IMAGE_NAME,
                                    allocated_cpus=allocated_cpus,
                                    cpus_per_replica=cpus_per_replica,
                                    gpus=[0],
                                    batch_size=batch_size,
                                    num_replicas=num_replicas,
                                    use_nvidia_docker=True,
                                    input_size=input_size,
                                    no_diverge=True
                                    )

########## Benchmarking ##########

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
            "mean_lats": []}
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
        self.stats["all_lats"] = self.stats["all_lats"] + self.latencies
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

    def predict(self, model_app_name, input_item):
        begin_time = datetime.now()
        def continuation(output):
            if output == DEFAULT_OUTPUT:
                return
            end_time = datetime.now()
            latency = (end_time - begin_time).total_seconds()
            self.latencies.append(latency)
            self.total_num_complete += 1
            self.batch_num_complete += 1
            if self.batch_num_complete % 200 == 0:
                self.print_stats()
                self.init_stats()

        return self.client.send_request(model_app_name, input_item).then(continuation)

class ModelBenchmarker(object):
    def __init__(self, config, queue, input_length=20):
        self.config = config
        self.queue = queue
        self.load_text_fn = self._get_load_text_fn(model_app_name=self.config.name)
        self.loaded_text = False
        base_inputs = self._gen_inputs(num_inputs=1000, input_length=input_length)
        self.inputs = [i for _ in range(40) for i in base_inputs]

    def run(self, client_num=0):
        assert client_num == 0
        self.initialize_request_rate()
        self.find_steady_state()
        return

    # start with an overly aggressive request rate
    # then back off
    def initialize_request_rate(self):
        # initialize delay to be very small
        self.delay = 0.001
        setup_clipper(self.config)
        time.sleep(5)
        predictor = Predictor(clipper_metrics=True)
        idx = 0
        while len(predictor.stats["thrus"]) < 5:
            predictor.predict(model_app_name=self.config.name, input_item=self.inputs[idx])
            time.sleep(self.delay)
            idx += 1
            idx = idx % len(self.inputs)

        max_thruput = np.mean(predictor.stats["thrus"][1:])
        self.delay = 1.0 / max_thruput
        logger.info("Initializing delay to {}".format(self.delay))

    def increase_delay(self):
        if self.delay < 0.005:
            self.delay += 0.0002
        else:
            self.delay += 0.0005

    def find_steady_state(self):
        setup_clipper(self.config)
        time.sleep(7)
        predictor = Predictor(clipper_metrics=True)
        idx = 0
        done = False
        # start checking for steady state after 7 trials
        last_checked_length = 6
        while not done:
            predictor.predict(model_app_name=self.config.name, input_item=self.inputs[idx])
            time.sleep(self.delay)
            idx += 1
            idx = idx % len(self.inputs)

            if len(predictor.stats["thrus"]) > last_checked_length:
                last_checked_length = len(predictor.stats["thrus"]) + 4
                convergence_state = driver_utils.check_convergence(predictor.stats, [self.config])
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

    def _gen_inputs(self, num_inputs=1000, input_length=20):
        if not self.loaded_text:
            self.text = self.load_text_fn()
            self.loaded_text = True

        inputs = []
        num_gen_inputs = 0
        while num_gen_inputs < num_inputs:
            idx = np.random.randint(len(self.text))
            text = self.text[idx]
            words = text.split()
            if len(words) < input_length:
                expansion_factor = int(math.ceil(float(input_length)/len(text)))
                for i in range(expansion_factor):
                    words = words + words
            words = words[:input_length]
            inputs.append(" ".join(words))
            num_gen_inputs += 1

        byte_inputs = [np.frombuffer(bytearray(input_item), dtype=np.uint8) for input_item in inputs]

        return byte_inputs

    def _get_load_text_fn(self, model_app_name):
        if model_app_name == NMT_MODEL_APP_NAME:
            return self._load_nmt_text

        elif model_app_name == LANG_DETECT_MODEL_APP_NAME:
            return self._load_detect_text

        elif model_app_name == LSTM_MODEL_APP_NAME:
            return self._load_lstm_text

    def _load_nmt_text(self):
        nmt_data_path = os.path.join(CURR_DIR, NMT_WORKLOAD_RELATIVE_PATH, "workload.txt")
        nmt_data_file = open(nmt_data_path, "rb")
        nmt_text = nmt_data_file.readlines()
        np.random.shuffle(nmt_text)
        return nmt_text

    def _load_detect_text(self):
        detect_data_path = os.path.join(CURR_DIR, LANG_DETECT_WORKLOAD_RELATIVE_PATH, "workload.txt")
        detect_data_file = open(detect_data_path, "rb")
        detect_text = detect_data_file.readlines()
        np.random.shuffle(detect_text)
        return detect_text

    def _load_lstm_text(self):
        lstm_data_path = os.path.join(CURR_DIR, LSTM_WORKLOAD_RELATIVE_PATH, "workload.txt")
        lstm_data_file = open(lstm_data_path, "rb")
        lstm_text = lstm_data_file.readlines()
        np.random.shuffle(lstm_text)
        return lstm_text

class InputLengthConfig:
    def __init__(self, input_length):
        self.input_length_words = input_length

    def to_json(self):
        return json.dumps(self.__dict__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Clipper image driver 1')
    parser.add_argument('-m', '--model_name', type=str, help="The name of the model to benchmark. One of: 'gensim-lda', 'gensim-docsim'")
    parser.add_argument('-b', '--batch_sizes', type=int, nargs='+', help="The batch size configurations to benchmark for the model. Each configuration will be benchmarked separately.")
    parser.add_argument('-r', '--num_replicas', type=int, nargs='+', help="The replica number configurations to benchmark for the model. Each configuration will be benchmarked separately.")
    parser.add_argument('-c', '--model_cpus', type=int, nargs='+', help="The set of cpu cores on which to run replicas of the provided model")
    parser.add_argument('-p', '--cpus_per_replica_nums', type=int, nargs='+', help="Configurations for the number of cpu cores allocated to each replica of the model")
    parser.add_argument('-g', '--model_gpus', type=int, nargs='+', help="The set of gpus on which to run replicas of the provided model. Each replica of a gpu model must have its own gpu!")
    parser.add_argument('-n', '--num_clients', type=int, default=1, help="The number of concurrent client processes. This can help increase the request rate in order to saturate high throughput models.")
    parser.add_argument('-l', '--input_lengths', type=int, nargs='+', help="Input length configurations to benchmark")
    
    args = parser.parse_args()

    if args.model_name not in VALID_MODEL_NAMES:
        raise Exception("Model name must be one of: {}".format(VALID_MODEL_NAMES))

    default_batch_size_confs = [2]
    default_replica_num_confs = [1]
    default_cpus_per_replica_confs = [1]
    default_input_length_confs = [20]

    batch_size_confs = args.batch_sizes if args.batch_sizes else default_batch_size_confs
    replica_num_confs = args.num_replicas if args.num_replicas else default_replica_num_confs
    cpus_per_replica_confs = args.cpus_per_replica_nums if args.cpus_per_replica_nums else default_cpus_per_replica_confs
    input_length_confs = args.input_lengths if args.input_lengths else default_input_length_confs

    for input_length in input_length_confs:
        for num_replicas in replica_num_confs:
            for cpus_per_replica in cpus_per_replica_confs:
                for batch_size in batch_size_confs:
                    input_length_config = InputLengthConfig(input_length)

                    model_config = get_heavy_node_config(model_name=args.model_name, 
                                                         batch_size=batch_size, 
                                                         num_replicas=num_replicas,
                                                         cpus_per_replica=cpus_per_replica,
                                                         allocated_cpus=args.model_cpus,                               
                                                         allocated_gpus=args.model_gpus,
                                                         input_size=input_length)
                    setup_clipper(model_config)
                    queue = Queue()
                    benchmarker = ModelBenchmarker(model_config, queue, input_length)

                    processes = []
                    all_stats = []
                    for client_num in range(args.num_clients):
                        p = Process(target=benchmarker.run, args=(client_num,))
                        p.start()
                        processes.append(p)
                    for p in processes:
                        all_stats.append(queue.get())
                        p.join()

                    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
                    cl.connect()
                    driver_utils.save_results([input_length_config, model_config], cl, all_stats, "gpu_and_batch_size_experiments")