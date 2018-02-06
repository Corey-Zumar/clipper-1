import sys
import os
import argparse
import numpy as np
import time
import base64
import logging
import json

from clipper_admin import ClipperConnection, DockerContainerManager
from threading import Lock
from datetime import datetime
from io import BytesIO
from PIL import Image
from containerized_utils.zmq_client import Client
from containerized_utils import driver_utils
from containerized_utils.driver_utils import INCREASING, DECREASING, CONVERGED_HIGH, CONVERGED, UNKNOWN, CONVERGED_LOW
from multiprocessing import Process, Queue

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

# Models and applications for each heavy node
# will share the same name
INCEPTION_FEATS_MODEL_APP_NAME = "inception"
TF_KERNEL_SVM_MODEL_APP_NAME = "tf-kernel-svm"
TF_LOG_REG_MODEL_APP_NAME = "tf-log-reg"
TF_RESNET_MODEL_APP_NAME = "tf-resnet-feats"

INCEPTION_FEATS_IMAGE_NAME = "model-comp/inception-feats"
TF_KERNEL_SVM_IMAGE_NAME = "model-comp/tf-kernel-svm"
TF_LOG_REG_IMAGE_NAME = "model-comp/tf-log-reg"
TF_RESNET_IMAGE_NAME = "model-comp/tf-resnet-feats"

CLIPPER_ADDRESS = "localhost"
CLIPPER_SEND_PORT = 4456
CLIPPER_RECV_PORT = 4455

DEFAULT_OUTPUT = "TIMEOUT"

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

def setup_inception(batch_size,
                    num_replicas,
                    cpus_per_replica,
                    allocated_cpus,
                    allocated_gpus):

    return driver_utils.HeavyNodeConfig(name=INCEPTION_FEATS_MODEL_APP_NAME,
                                        input_type="floats",
                                        model_image=INCEPTION_FEATS_IMAGE_NAME,
                                        allocated_cpus=allocated_cpus,
                                        cpus_per_replica=cpus_per_replica,
                                        gpus=allocated_gpus,
                                        batch_size=batch_size,
                                        num_replicas=num_replicas,
                                        use_nvidia_docker=True,
                                        no_diverge=True,
                                        )

def setup_log_reg(batch_size,
                  num_replicas,
                  cpus_per_replica,
                  allocated_cpus,
                  allocated_gpus):

    return driver_utils.HeavyNodeConfig(name=TF_LOG_REG_MODEL_APP_NAME,
                                        input_type="floats",
                                        model_image=TF_LOG_REG_IMAGE_NAME,
                                        allocated_cpus=allocated_cpus,
                                        cpus_per_replica=cpus_per_replica,
                                        gpus=allocated_gpus,
                                        batch_size=batch_size,
                                        num_replicas=num_replicas,
                                        use_nvidia_docker=True,
                                        no_diverge=True,
                                        )

def setup_kernel_svm(batch_size,
                    num_replicas,
                    cpus_per_replica,
                    allocated_cpus,
                    allocated_gpus):

    return driver_utils.HeavyNodeConfig(name=TF_KERNEL_SVM_MODEL_APP_NAME,
                                        input_type="floats",
                                        model_image=TF_KERNEL_SVM_IMAGE_NAME,
                                        allocated_cpus=allocated_cpus,
                                        cpus_per_replica=cpus_per_replica,
                                        gpus=allocated_gpus,
                                        batch_size=batch_size,
                                        num_replicas=num_replicas,
                                        use_nvidia_docker=True,
                                        no_diverge=True,
                                        )

def setup_resnet(batch_size,
                 num_replicas,
                 cpus_per_replica,
                 allocated_cpus,
                 allocated_gpus):

    return driver_utils.HeavyNodeConfig(name=TF_RESNET_MODEL_APP_NAME,
                                        input_type="floats",
                                        model_image=TF_RESNET_IMAGE_NAME,
                                        allocated_cpus=allocated_cpus,
                                        cpus_per_replica=cpus_per_replica,
                                        gpus=allocated_gpus,
                                        batch_size=batch_size,
                                        num_replicas=num_replicas,
                                        use_nvidia_docker=True,
                                        no_diverge=True,
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

def get_queue_sizes(metrics_json):
    hists = metrics_json["histograms"]
    mean_queue_sizes = {}
    for h in hists:
        if "queue_size" in h.keys()[0]:
            name = h.keys()[0]
            model = name.split(":")[0]
            mean = h[name]["mean"]
            mean_queue_sizes[model] = round(float(mean), 2)
    return mean_queue_sizes

class Predictor(object):

    def __init__(self, clipper_metrics, batch_size):
        self.outstanding_reqs = {}
        self.client = Client(CLIPPER_ADDRESS, CLIPPER_SEND_PORT, CLIPPER_RECV_PORT)
        self.client.start()
        self.init_stats()
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "mean_lats": [],
            "all_lats": [],
        }
        self.total_num_complete = 0
        self.cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        self.cl.connect()
        self.batch_size = batch_size
        self.get_clipper_metrics = clipper_metrics
        if self.get_clipper_metrics:
            self.stats["all_metrics"] = []
            self.stats["mean_batch_sizes"] = []
            self.stats["mean_queue_sizes"] = []

    def init_stats(self):
        self.latencies = []
        self.trial_num_complete = 0
        self.cur_req_id = 0
        self.start_time = datetime.now()

    def print_stats(self):
        lats = np.array(self.latencies)
        p99 = np.percentile(lats, 99)
        mean = np.mean(lats)
        end_time = datetime.now()
        thru = float(self.trial_num_complete) / (end_time - self.start_time).total_seconds()
        self.stats["thrus"].append(thru)
        self.stats["p99_lats"].append(p99)
        self.stats["mean_lats"].append(mean)
        self.stats["all_lats"].append(self.latencies)
        if self.get_clipper_metrics:
            metrics = self.cl.inspect_instance()
            batch_sizes = get_batch_sizes(metrics)
            queue_sizes = get_queue_sizes(metrics)
            self.stats["mean_batch_sizes"].append(batch_sizes)
            self.stats["mean_queue_sizes"].append(queue_sizes)
            self.stats["all_metrics"].append(metrics)
            logger.info(("p99: {p99}, mean: {mean}, thruput: {thru}, "
                         "batch_sizes: {batches}").format(p99=p99, mean=mean, thru=thru,
                                                          batches=json.dumps(
                                                              batch_sizes, sort_keys=True)))
        else:
            logger.info("p99: {p99}, mean: {mean}, thruput: {thru}".format(p99=p99,
                                                                           mean=mean,
                                                                           thru=thru))

    def predict(self, resnet_input, inception_input):
        begin_time = datetime.now()
        classifications_lock = Lock()
        classifications = {}

        def update_perf_stats():
            end_time = datetime.now()
            latency = (end_time - begin_time).total_seconds()
            self.latencies.append(latency)
            self.total_num_complete += 1
            self.trial_num_complete += 1

            trial_length = max(300, 10 * self.batch_size)
            if self.trial_num_complete % trial_length == 0:
                self.print_stats()
                self.init_stats()

        def resnet_feats_continuation(resnet_features):
            if resnet_features == DEFAULT_OUTPUT:
                return
            return self.client.send_request(TF_KERNEL_SVM_MODEL_APP_NAME, resnet_features)

        def svm_continuation(svm_classification):
            if svm_classification == DEFAULT_OUTPUT:
                return
            else:
                classifications_lock.acquire()
                if TF_LOG_REG_MODEL_APP_NAME not in classifications:
                    classifications[TF_KERNEL_SVM_MODEL_APP_NAME] = svm_classification
                else:
                    update_perf_stats()
                classifications_lock.release()

        def inception_feats_continuation(inception_features):
            if inception_features == DEFAULT_OUTPUT:
                return
            return self.client.send_request(TF_LOG_REG_MODEL_APP_NAME, inception_features)


        def log_reg_continuation(log_reg_vals):
            if log_reg_vals == DEFAULT_OUTPUT:
                return
            else:
                classifications_lock.acquire()
                if TF_KERNEL_SVM_MODEL_APP_NAME not in classifications:
                    classifications[TF_LOG_REG_MODEL_APP_NAME] = log_reg_vals
                else:
                    update_perf_stats()
                classifications_lock.release()

        self.client.send_request(TF_RESNET_MODEL_APP_NAME, resnet_input) \
            .then(resnet_feats_continuation) \
            .then(svm_continuation)

        self.client.send_request(INCEPTION_FEATS_MODEL_APP_NAME, inception_input) \
            .then(inception_feats_continuation) \
            .then(log_reg_continuation)

class DriverBenchmarker(object):
    def __init__(self, configs, queue, client_num, latency_upper_bound):
        self.configs = configs
        self.max_batch_size = np.max([config.batch_size for config in configs])
        self.queue = queue
        assert client_num == 0
        self.client_num = client_num
        logger.info("Generating random inputs")
        base_inputs = [(self._get_resnet_input(), self._get_inception_input()) for _ in range(1000)]
        self.inputs = [i for _ in range(40) for i in base_inputs]
        self.latency_upper_bound = latency_upper_bound

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
        self.queries_per_sleep = 1
        self.clipper_address = setup_clipper(self.configs)
        self.cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        self.cl.connect()
        time.sleep(30)
        predictor = Predictor(clipper_metrics=True, batch_size=self.max_batch_size)
        idx = 0

        # First warm up the model.
        # NOTE: The length of time the model needs to warm up for
        # seems to be both framework and hardware dependent. 27 seems to work
        # well for PyTorch resnet, but we don't need as many warmup
        # iterations for the JIT-free models in this pipeline
        while len(predictor.stats["thrus"]) < 8:
            resnet_input, inception_input = self.inputs[idx]
            predictor.predict(resnet_input, inception_input)
            idx += 1
            if idx % self.queries_per_sleep == 0:
                time.sleep(self.delay)
            idx = idx % len(self.inputs)

        # Now let the queue drain
        logger.info("Draining queue")

        self.cl.drain_queues()
        predictor.client.stop()
        logger.info("ZMQ client stopped")
        del predictor
        time.sleep(10)
        predictor = Predictor(clipper_metrics=True, batch_size=self.max_batch_size)
        # while predictor.stats["mean_queue_sizes"][-1] > 0:
        #     sleep_time_secs = 5
        #     logger.info("Queue has {q_len} queries. Sleeping {sleep}".format(
        #         q_len=predictor.stats["mean_queue_sizes"][-1],
        #         sleep=sleep_time_secs))
        #     time.sleep(sleep_time_secs)

        # Now initialize request rate
        while len(predictor.stats["thrus"]) < 10:
            resnet_input, inception_input = self.inputs[idx]
            predictor.predict(resnet_input, inception_input)
            idx += 1
            if idx % self.queries_per_sleep == 0:
                time.sleep(self.delay)
            idx = idx % len(self.inputs)

        max_thruput = np.mean(predictor.stats["thrus"][1:])
        self.delay = 1.0 / max_thruput
        if self.delay < 0.01:
            self.queries_per_sleep = 2
            self.delay = self.delay*2.0 - 0.001
        logger.info("Initializing delay to {}".format(self.delay))
        predictor.client.stop()
        logger.info("ZMQ client stopped")
        del predictor

    def increase_delay(self, multiple=1.0):
        if self.delay < 0.01:
            self.delay += 0.00005*multiple
        elif self.delay < 0.02:
            self.delay += 0.0002*multiple
        else:
            self.delay += 0.001*multiple

    def find_steady_state(self):
        # self.cl.cm.reset()
        self.cl.drain_queues()
        time.sleep(10)
        logger.info("Queue is drained")
        predictor = Predictor(clipper_metrics=True, batch_size=self.max_batch_size)
        time.sleep(10)

        idx = 0
        done = False
        # start checking for steady state after 10 trials
        last_checked_length = 10
        while not done:
            resnet_input, inception_input = self.inputs[idx]
            predictor.predict(resnet_input, inception_input)
            if idx % self.queries_per_sleep == 0:
                time.sleep(self.delay)
            idx += 1
            idx = idx % len(self.inputs)

            if len(predictor.stats["thrus"]) > last_checked_length:
                last_checked_length = len(predictor.stats["thrus"]) + 1
                convergence_state = driver_utils.check_convergence_via_queue(
                    predictor.stats, self.configs, self.latency_upper_bound)
                # Diverging, try again with higher
                # delay
                if convergence_state == INCREASING or convergence_state == CONVERGED_HIGH:
                    self.increase_delay()
                    logger.info("Increasing delay to {}".format(self.delay))
                    done = True
                    del predictor
                    return self.find_steady_state()
                elif convergence_state == CONVERGED:
                    logger.info("Converged with delay of {}".format(self.delay))
                    done = True
                    self.queue.put(predictor.stats)
                    return
                elif len(predictor.stats) > 40:
                    self.increase_delay()
                    logger.info("Increasing delay to {}".format(self.delay))
                    done = True
                    del predictor
                    return self.find_steady_state()
                elif convergence_state == DECREASING or convergence_state == UNKNOWN:
                    logger.info("Not converged yet. Still waiting")
                elif convergence_state == CONVERGED_LOW:
                    logger.info("Converged LOW with delay of {}".format(self.delay))
                    logger.info("Consider re-running with smaller request delay augmentations")
                    done = True
                    self.queue.put(predictor.stats)
                    return
                else:
                    logger.error("Unknown convergence state: {}".format(convergence_state))
                    sys.exit(1)

    def _get_resnet_input(self):
        resnet_input = np.array(np.random.rand(224, 224, 3) * 255, dtype=np.float32)
        return resnet_input.flatten()

    def _get_inception_input(self):
        inception_input = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
        return inception_input.flatten()

class RequestDelayConfig:
    def __init__(self, request_delay):
        self.request_delay = request_delay
        
    def to_json(self):
        return json.dumps(self.__dict__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Clipper image driver 1')
    parser.add_argument('-g', '--num_gpus', type=int, default=4, help="The number of GPUs available for use")

    args = parser.parse_args()

    queue = Queue()

    ## FORMAT IS (INCEPTION, LOG REG, RESNET, KSVM)
    min_lat_reps = [(1, 1, 1, 1),
                    (1, 1, 2, 1),
                    (1, 1, 3, 1),
                    (2, 1, 3, 1),
                    (2, 1, 4, 1),
                    (2, 1, 5, 1),
                    (3, 1, 5, 1)]


    ## FORMAT IS (INCEPTION, LOG REG, RESNET, KSVM)
    min_lat_batches = (1, 1, 1, 1)

    min_lat_latency_upper_bound = 0.2

    # ## THIS IS FOR 500MS
    # ## FORMAT IS (INCEPTION, LOG_REG, RESNET, KSVM)
    # five_hundred_ms_batches = (

    # five_hundred_ms_latency_upper_bound = 1.500

    # ## THIS IS FOR 375MS
    # ## FORMAT IS (INCEPTION, LOG_REG, KSVM, RESNET)
    # three_seven_five_ms_reps = [(1, 1, 1, 1),
    #                             (1, 1, 1, 2),
    #                             (1, 1, 1, 3),
    #                             (1, 1, 1, 4),
    #                             (1, 1, 1, 5),
    #                             (2, 1, 1, 5),
    #                             (2, 1, 1, 6)]

    # ## THIS IS FOR 375MS
    # ## FORMAT IS (INCEPTION, LOG_REG, KSVM, RESNET)
    # three_seven_five_ms_batches = (7, 2, 9, 2)

    # three_seven_five_ms_latency_upper_bound = 1.000

    # ## THIS IS FOR 375MS
    # ## FORMAT IS (INCEPTION, LOG_REG, KSVM, RESNET)
    # thousand_ms_reps = [(1, 1, 1, 1),
    #                     (1, 1, 1, 2),
    #                     (2, 1, 1, 2),
    #                     (2, 1, 1, 3),
    #                     (2, 1, 1, 4),
    #                     (3, 1, 1, 4),
    #                     (3, 1, 1, 5)]

    # ## THIS IS FOR 1000MS
    # ## FORMAT IS (INCEPTION, LOG_REG, KSVM, RESNET)
    # thousand_ms_batches = (16, 2, 16, 15)

    # thousand_ms_latency_upper_bound = 3.000

    inception_batch_idx = 0
    log_reg_batch_idx = 1
    resnet_batch_idx = 2
    ksvm_batch_idx = 3

    for inception_reps, log_reg_reps, resnet_reps, ksvm_reps in min_lat_reps:
        # Note: These are PHYSICAL CPU numbers
        total_cpus = range(4,14)

        def get_cpus(num_cpus):
            return [total_cpus.pop() for _ in range(num_cpus)]

        total_gpus = range(args.num_gpus)

        def get_gpus(num_gpus):
            return [total_gpus.pop() for _ in range(num_gpus)]

        # Note: cpus_per_replica refers to PHYSICAL CPUs per replica

        inception_cpus_per_replica = 1
        resnet_cpus_per_replica = 1


        configs = [
            setup_inception(batch_size=min_lat_batches[inception_batch_idx],
                            num_replicas=inception_reps,
                            cpus_per_replica=inception_cpus_per_replica,
                            allocated_cpus=get_cpus(inception_cpus_per_replica * inception_reps),
                            allocated_gpus=get_gpus(inception_reps)),
            setup_log_reg(batch_size=min_lat_batches[log_reg_batch_idx],
                          num_replicas=log_reg_reps,
                          cpus_per_replica=1,
                          allocated_cpus=get_cpus(log_reg_reps),
                          allocated_gpus=[]),
            setup_kernel_svm(batch_size=min_lat_batches[ksvm_batch_idx],
                             num_replicas=ksvm_reps,
                             cpus_per_replica=1,
                             allocated_cpus=get_cpus(ksvm_reps),
                             allocated_gpus=[]),
            setup_resnet(batch_size=min_lat_batches[resnet_batch_idx],
                         num_replicas=resnet_reps,
                         cpus_per_replica=resnet_cpus_per_replica,
                         allocated_cpus=get_cpus(resnet_reps * resnet_cpus_per_replica),
                         allocated_gpus=get_gpus(resnet_reps))
        ]

        client_num = 0

        benchmarker = DriverBenchmarker(configs, queue, client_num, min_lat_latency_upper_bound, args.request_delay)

        p = Process(target=benchmarker.run)
        p.start()

        all_stats = []
        all_stats.append(queue.get())

        cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        cl.connect()

        fname = "incep_{}-logreg_{}-ksvm_{}-resnet_{}".format(inception_reps, log_reg_reps, ksvm_reps, resnet_reps)
        driver_utils.save_results(configs, cl, all_stats, "e2e_min_lat_img_driver_1", prefix=fname)
    
    sys.exit(0)
