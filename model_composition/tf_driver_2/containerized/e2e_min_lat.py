import sys
import os
import numpy as np
import time
import base64
import logging
import json
import math

from clipper_admin import ClipperConnection, DockerContainerManager
from threading import Lock
from datetime import datetime
from io import BytesIO
from PIL import Image
from containerized_utils.zmq_client import Client
from containerized_utils import driver_utils
from containerized_utils.driver_utils import INCREASING, DECREASING, CONVERGED_HIGH, CONVERGED, UNKNOWN
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

LANG_DETECT_WORKLOAD_RELATIVE_PATH = "lang_detect_workload"
NMT_WORKLOAD_RELATIVE_PATH = "nmt_workload"
LSTM_WORKLOAD_RELATIVE_PATH = "lstm_workload"

CLIPPER_ADDRESS = "localhost"
CLIPPER_SEND_PORT = 4456
CLIPPER_RECV_PORT = 4455

DEFAULT_OUTPUT = "TIMEOUT"

LANG_CLASSIFICATION_ENGLISH = "en"
LANG_CLASSIFICATION_GERMAN = "de"

########## Setup ##########

def setup_clipper(configs):
    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.stop_all()
    cl.start_clipper(
        query_frontend_image="clipper/zmq_frontend:develop",
        redis_cpu_str="0",
        mgmt_cpu_str="0",
        query_cpu_str="0,16,1,17,2,18,3,19")
    time.sleep(10)
    for config in configs:
        driver_utils.setup_heavy_node(cl, config, DEFAULT_OUTPUT)
    time.sleep(20)
    logger.info("Clipper is set up!")
    return config

def setup_lang_detect(batch_size,
                      num_replicas,
                      cpus_per_replica,
                      allocated_cpus,
                      allocated_gpus,
                      input_size):
    return driver_utils.HeavyNodeConfig(name=LANG_DETECT_MODEL_APP_NAME,
                                        input_type="bytes",
                                        model_image=LANG_DETECT_IMAGE_NAME,
                                        allocated_cpus=allocated_cpus,
                                        cpus_per_replica=cpus_per_replica,
                                        gpus=allocated_gpus,
                                        batch_size=batch_size,
                                        num_replicas=num_replicas,
                                        use_nvidia_docker=True,
                                        input_size=input_size,
                                        no_diverge=True
                                        )

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
                                        no_diverge=True
                                        )

def setup_lstm(batch_size,
               num_replicas,
               cpus_per_replica,
               allocated_cpus,
               allocated_gpus,
               input_size):
    return driver_utils.HeavyNodeConfig(name=LSTM_MODEL_APP_NAME,
                                        input_type="bytes",
                                        model_image=LSTM_IMAGE_NAME,
                                        allocated_cpus=allocated_cpus,
                                        cpus_per_replica=cpus_per_replica,
                                        gpus=allocated_gpus,
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

    def __init__(self, clipper_metrics, batch_size):
        self.outstanding_reqs = {}
        self.client = Client(CLIPPER_ADDRESS, CLIPPER_SEND_PORT, CLIPPER_RECV_PORT)
        self.client.start()
        self.init_stats()
        self.stats = {
            "probabilities": [],
            "thrus": [],
            "p99_lats": [],
            "all_lats": [],
            "mean_lats": []}
        self.total_num_complete = 0
        self.cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        self.cl.connect()
        self.batch_size = batch_size
        self.get_clipper_metrics = clipper_metrics
        if self.get_clipper_metrics:
            self.stats["all_metrics"] = []
            self.stats["mean_batch_sizes"] = []

    def init_stats(self):
        self.latencies = []
        self.batch_num_complete = 0
        self.cur_req_id = 0
        self.start_time = datetime.now()
        self.nmt_count = 0
        self.lstm_count = 0
        self.lang_detect_count = 0
        self.total_count = 0

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

        nmt_pct = float(self.nmt_count) / self.total_count
        lstm_pct = float(self.lstm_count) / self.total_count
        ld_pct = float(self.lang_detect_count) / self.total_count

        self.stats["probabilities"].append((nmt_pct, lstm_pct, ld_pct))

        logger.info("NMT: {}, LSTM: {}, LANG DETECT: {}".format(nmt_pct, lstm_pct, ld_pct))

    def predict(self, lang_input):
        begin_time = datetime.now()

        def update_perf_stats():
            end_time = datetime.now()
            latency = (end_time - begin_time).total_seconds()
            self.latencies.append(latency)
            self.total_num_complete += 1
            self.batch_num_complete += 1

            trial_length = max(300, 50 * self.batch_size)
            if self.batch_num_complete % trial_length == 0:
                self.print_stats()
                self.init_stats()

        def lang_detect_continuation(lang_classification):
            self.total_count += 1
            self.lang_detect_count += 1
            if lang_classification == DEFAULT_OUTPUT:
                return
            elif lang_classification == LANG_CLASSIFICATION_GERMAN:
                return self.client.send_request(NMT_MODEL_APP_NAME, lang_input) \
                                  .then(nmt_continuation) \
                                  .then(lstm_continuation)
            elif lang_classification == LANG_CLASSIFICATION_ENGLISH:
                return self.client.send_request(LSTM_MODEL_APP_NAME, lang_input) \
                                  .then(lstm_continuation)
            else:
                update_perf_stats()

        def nmt_continuation(translation):
            self.nmt_count += 1
            if translation == DEFAULT_OUTPUT:
                return
            else:
                translation_bytes = np.frombuffer(bytearray(translation), dtype=np.uint8)
                return self.client.send_request(LSTM_MODEL_APP_NAME, translation_bytes)

        def lstm_continuation(classification):
            self.lstm_count += 1
            if classification == DEFAULT_OUTPUT:
                return
            else:
                update_perf_stats()


        self.client.send_request(LANG_DETECT_MODEL_APP_NAME, lang_input) \
            .then(lang_detect_continuation)

class DriverBenchmarker(object):
    def __init__(self, configs, queue, client_num, latency_upper_bound, input_size, delay=None):
        self.delay = delay
        self.loaded_text = False
        self.configs = configs
        self.max_batch_size = np.max([config.batch_size for config in configs])
        self.queue = queue
        assert client_num == 0
        self.client_num = client_num
        logger.info("Generating random inputs")
        base_inputs = self._gen_inputs(num_inputs=1000, input_length=input_size)
        self.inputs = [i for _ in range(40) for i in base_inputs]
        self.latency_upper_bound = latency_upper_bound
    def run(self):
        if not self.delay:
            self.initialize_request_rate()
        self.find_steady_state()
        return

    # start with an overly aggressive request rate
    # then back off
    def initialize_request_rate(self):
        # initialize delay to be very small
        self.delay = 0.001
        setup_clipper(self.configs)
        time.sleep(5)
        predictor = Predictor(clipper_metrics=True, batch_size=self.max_batch_size)
        idx = 0
        while len(predictor.stats["thrus"]) < 10:
            lang_input = self.inputs[idx]
            predictor.predict(lang_input)
            time.sleep(self.delay)
            idx += 1
            idx = idx % len(self.inputs)

        max_thruput = np.mean(predictor.stats["thrus"][1:])
        self.delay = 1.0 / max_thruput
        logger.info("Initializing delay to {}".format(self.delay))

    def increase_delay(self):
        if self.delay < 0.005:
            self.delay += 0.00005
        elif self.delay < 0.01:
            self.delay += 0.0001
        else:
            self.delay += 0.00025


    def find_steady_state(self):
        setup_clipper(self.configs)
        time.sleep(7)
        predictor = Predictor(clipper_metrics=True, batch_size=self.max_batch_size)
        idx = 0
        done = False
        # start checking for steady state after 7 trials
        last_checked_length = 10
        while not done:
            lang_input = self.inputs[idx]
            predictor.predict(lang_input)
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

    def _gen_inputs(self, num_inputs=1000, input_length=20):
        if not self.loaded_text:
            self.text = self._load_detect_text()
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

    def _load_detect_text(self):
        detect_data_path = os.path.join(CURR_DIR, LANG_DETECT_WORKLOAD_RELATIVE_PATH, "workload.txt")
        detect_data_file = open(detect_data_path, "rb")
        detect_text = detect_data_file.readlines()
        np.random.shuffle(detect_text)
        return detect_text

class RequestDelayConfig:
    def __init__(self, request_delay):
        self.request_delay = request_delay
        
    def to_json(self):
        return json.dumps(self.__dict__)

if __name__ == "__main__":
    queue = Queue()

    initial_request_delay = None

    input_size = 20

    min_lat_est_thrus = [16, 32, 48, 
                         64, 76, 80, 
                         94, 104, 
                         110, 126]

    ## THIS IS FOR MIN_LAT
    ## FORMAT IS (LANG_DETECT, NMT, LSTM)
    min_lat_reps = [(1,1,1),
                    (1,2,1),
                    (1,3,1),
                    (1,4,1),
                    (1,5,1),
                    (2,5,1),
                    (2,6,1),
                    (2,7,1),
                    (2,7,2),
                    (2,8,2)]

    min_lat_batches = (1,1,1)

    min_lat_latency_upper_bound = 7.0

    lang_detect_batch_idx = 0
    nmt_batch_idx = 1
    lstm_batch_idx = 2

    for i in range(len(min_lat_reps)):
        lang_detect_reps, nmt_reps, lstm_reps = min_lat_reps[i]

        request_delay = 1.0 / (min_lat_est_thrus[i] * 1.1)

        total_cpus = range(4,14)

        def get_cpus(num_cpus):
            return [total_cpus.pop() for _ in range(num_cpus)]

        total_gpus = range(8)

        def get_gpus(num_gpus):
            return [total_gpus.pop() for _ in range(num_gpus)]

        configs = [
            setup_lang_detect(batch_size=min_lat_batches[lang_detect_batch_idx],
                              num_replicas=lang_detect_reps,
                              cpus_per_replica=1,
                              allocated_cpus=get_cpus(lang_detect_reps),
                              allocated_gpus=[],
                              input_size=input_size),
            setup_nmt(batch_size=min_lat_batches[nmt_batch_idx],
                      num_replicas=nmt_reps,
                      cpus_per_replica=1,
                      allocated_cpus=get_cpus(nmt_reps),
                      allocated_gpus=get_gpus(nmt_reps),
                      input_size=input_size),
            setup_lstm(batch_size=min_lat_batches[lstm_batch_idx],
                       num_replicas=lstm_reps,
                       cpus_per_replica=1,
                       allocated_cpus=get_cpus(lstm_reps),
                       allocated_gpus=[],
                       input_size=input_size)
        ]

        client_num = 0

        benchmarker = DriverBenchmarker(configs, queue, client_num, min_lat_latency_upper_bound, input_size, request_delay)

        p = Process(target=benchmarker.run)
        p.start()

        all_stats = []
        all_stats.append(queue.get())

        cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        cl.connect()

        fname = "langdetect_{}-nmt_{}-lstm_{}".format(lang_detect_reps, nmt_reps, lstm_reps)
        driver_utils.save_results(configs, cl, all_stats, "e2e_min_lat_tf_text_driver", prefix=fname)
    
    sys.exit(0)