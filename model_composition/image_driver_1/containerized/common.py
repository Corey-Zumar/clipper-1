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
VGG_FEATS_MODEL_APP_NAME = "vgg"
VGG_KPCA_SVM_MODEL_APP_NAME = "kpca-svm"
VGG_KERNEL_SVM_MODEL_APP_NAME = "kernel-svm"
VGG_ELASTIC_NET_MODEL_APP_NAME = "elastic-net"
LGBM_MODEL_APP_NAME = "lgbm"
PYTORCH_RESNET_MODEL_APP_NAME = "pytorch-resnet-feats"

INCEPTION_FEATS_MODEL_APP_NAME = "inception"
TF_KERNEL_SVM_MODEL_APP_NAME = "tf-kernel-svm"
TF_LOG_REG_MODEL_APP_NAME = "tf-log-reg"
TF_RESNET_MODEL_APP_NAME = "tf-resnet-feats"
ID1_APP_NAME = "ID1"

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

    def __init__(self, app_name, clipper_metrics, batch_size):
        # The client through which the predictions are sent
        self.app_name = app_name
        self.predict = self.chosen_prediction_function(app_name)
        self.client = Client(CLIPPER_ADDRESS, CLIPPER_SEND_PORT, CLIPPER_RECV_PORT)
        self.client.start()
        self.init_stats()
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "mean_lats": [],
            "all_lats": [],
            "send_timestamps": [],
            "receive_timestamps": []
        }
        self.cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        self.cl.connect()
        self.batch_size = batch_size
        self.get_clipper_metrics = clipper_metrics
        self.inception_time = datetime.now()
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
                         "batch_sizes: {batches}, queue_sizes: {queues}").format(p99=p99, mean=mean, thru=thru,
                                                          batches=json.dumps(
                                                              batch_sizes, sort_keys=True),
                                                          queues=json.dumps(
                                                              queue_sizes, sort_keys=True)))
        else:
            logger.info("p99: {p99}, mean: {mean}, thruput: {thru}".format(p99=p99,
                                                                           mean=mean,
                                                                           thru=thru))

    def update_perf_stats(self, begin_time):
        end_time = datetime.now()
        latency = (end_time - begin_time).total_seconds()
        self.latencies.append(latency)
        self.trial_num_complete += 1
        self.stats["receive_timestamps"].append((end_time - self.inception_time).total_seconds())

        trial_length = max(300, 10 * self.batch_size)
        if self.trial_num_complete % trial_length == 0:
            self.print_stats()
            self.init_stats()

    def chosen_prediction_function(self, model_app_name):
        if model_app_name in [VGG_FEATS_MODEL_APP_NAME,VGG_KPCA_SVM_MODEL_APP_NAME,
                                VGG_KERNEL_SVM_MODEL_APP_NAME, VGG_ELASTIC_NET_MODEL_APP_NAME, 
                                INCEPTION_FEATS_MODEL_APP_NAME, LGBM_MODEL_APP_NAME, 
                                TF_KERNEL_SVM_MODEL_APP_NAME, TF_LOG_REG_MODEL_APP_NAME, 
                                TF_RESNET_MODEL_APP_NAME, PYTORCH_RESNET_MODEL_APP_NAME]:
            return self.single_model_predict
        elif model_app_name == ID1_APP_NAME:
            return self.ID1_predict
        else:
            raise Exception("Given model app name was not found")

    def single_model_predict(self, input_item):
        begin_time = datetime.now()
        self.stats["send_timestamps"].append((begin_time - self.inception_time).total_seconds())
        def continuation(output):
            if output == DEFAULT_OUTPUT:
                return
            self.update_perf_stats(begin_time)

        return self.client.send_request(self.app_name, input_item).then(continuation)

    def ID1_predict(self, pipeline_input):
        resnet_input, inception_input = pipeline_input
        begin_time = datetime.now()
        classifications_lock = Lock()
        classifications = {}

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
                    self.update_perf_stats(begin_time)
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
                    self.update_perf_stats(begin_time)
                classifications_lock.release()

        def try_catch(fun):
            def try_fun(x):
                try:
                    return fun(x)
                except Exception, e:
                    import traceback
                    logging.exception("Something bad happened!")
            return try_fun

        self.client.send_request(TF_RESNET_MODEL_APP_NAME, resnet_input) \
            .then(try_catch(resnet_feats_continuation)) \
            .then(try_catch(svm_continuation))

        self.client.send_request(INCEPTION_FEATS_MODEL_APP_NAME, inception_input) \
            .then(try_catch(inception_feats_continuation)) \
            .then(try_catch(log_reg_continuation))

class ArrivalProcess(object):
    def __init__(self, process, batchsize, arg):
        self.batchsize = batchsize
        self._arg = arg
        self.process = process
        self.iteration = 0
        self.base = self._init_random_base()
        self.base_length = len(self.base)
        self.iteration = 0
    
    def __str__(self):
        if self.process == "constant":
            process_str = "C"
        elif self.process == "exponential":
            process_str = "E"
        else:
            raise Exception("Unknown process type")
        return process_str+"|"+str(self.batchsize)+"|"+str(self._arg)

    def input_arg():
        return self._arg

    def _init_random_base(self):
        if self.process == "constant":
            return [self._arg]
        elif self.process == "exponential":
            return np.random.exponential(scale=self._arg, size=2000)
        else:
            raise Exception("Unknown process type")

    # reduces the number of sleep calls, slightly batches results in two's though
    def scale_delay(self):
        if self._arg < 0.01:
            self.batchsize*=2
            self._arg = self._arg*2.0
            self.base = self._init_random_base()

    def increase_delay(self, multiple=1.0):
        if self._arg < 0.01:
            self._arg += 0.00005*multiple
        elif self._arg < 0.02:
            self._arg += 0.0002*multiple
        else:
            self._arg += 0.001*multiple
        self.base = self._init_random_base()

    def wait(self):
        next_delay = self.base[self.iteration]
        self.iteration+=1
        self.iteration%=self.base_length
        time.sleep(next_delay)
        return self.batchsize

class DriverBenchmarker(object):
    def __init__(self, app_name, configs, queue, latency_upper_bound, input_delay_process=None):
        # provided delay to being with. If None attempts to converge to find a value.
        self.input_delay_process = input_delay_process
        self.configs = configs
        self.max_batch_size = np.max([config.batch_size for config in configs])
        self.queue = queue
        logger.info("Generating random inputs")
        self.app_name = app_name
        self.inputs = self.generate_inputs(self._get_inputs_for_model(app_name), 1000, 1000)
        self.latency_upper_bound = latency_upper_bound
        self.predictor = None
        self.iteration = 0
        self.total_iterations = 0

    def _get_vgg_feats_input(self):
        return np.array(np.random.rand(224, 224, 3) * 255, dtype=np.float32).flatten()

    def _get_vgg_classifier_input(self):
        return np.array(np.random.rand(4096), dtype=np.float32)

    def _get_inception_input(self):
        return np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32).flatten()

    def _get_lgbm_input(self):
        return np.array(np.random.rand(2048), dtype=np.float32)

    def _get_tf_kernel_svm_input(self):
        return np.array(np.random.rand(2048), dtype=np.float32)

    def _get_tf_log_reg_input(self):
        return np.array(np.random.rand(2048), dtype=np.float32)

    def _get_tf_resnet_input(self):
        return np.array(np.random.rand(224, 224, 3) * 255, dtype=np.float32).flatten()

    def _get_ID1_input(self):
        return [self._get_tf_resnet_input(), self._get_inception_input()]

    def _get_inputs_for_model(self, model_app_name):
        if model_app_name == VGG_FEATS_MODEL_APP_NAME:
            return _get_vgg_feats_input
        elif model_app_name in [VGG_KPCA_SVM_MODEL_APP_NAME, VGG_KERNEL_SVM_MODEL_APP_NAME, VGG_ELASTIC_NET_MODEL_APP_NAME]:
            return self._get_vgg_classifier_input
        elif model_app_name == INCEPTION_FEATS_MODEL_APP_NAME:
            return self._get_inception_input
        elif model_app_name == LGBM_MODEL_APP_NAME:
            return self._get_lgbm_input
        elif model_app_name == TF_KERNEL_SVM_MODEL_APP_NAME:
            return self._get_tf_kernel_svm_input
        elif model_app_name == TF_LOG_REG_MODEL_APP_NAME:
            return self._get_tf_log_reg_input
        elif model_app_name == TF_RESNET_MODEL_APP_NAME:
            return self._get_tf_resnet_input
        elif model_app_name == PYTORCH_RESNET_MODEL_APP_NAME:
            return self._get_tf_resnet_input
        elif model_app_name == ID1_APP_NAME:
            return self._get_ID1_input
        else:
            raise Exception("Given model app name was not found")

    def generate_inputs(self, seed_fun, base, size):
        input_base = np.asarray([seed_fun() for _ in range(base)])
        base_size = input_base.shape[0]
        quotient = size // base_size
        remainder = size % base_size
        return np.concatenate([np.tile(input_base, (quotient,1)), input_base[:remainder]], axis=0)

    #### Mini-interface for using the Predictor in a decoupled way, just call these methods when needed ####
    def init_predictor(self):
        if self.predictor != None:
            raise Exception("Tried to intialize predictor twice")
        self.predictor = Predictor(self.app_name, clipper_metrics=True, batch_size=self.max_batch_size)

    def run_predictor(self):
        batchsize = self.delay.wait()
        for i in range(batchsize):
            self.predictor.predict(self.inputs[self.iteration])
        self.iteration+=1
        self.total_iterations+=1
        self.iteration = self.iteration % len(self.inputs)
        if self.total_iterations%1000 == 0:
            logger.info("Sent for a total of {} prediction batches".format(self.total_iterations))

    def reset_predictor(self, wait = True):
        logger.info("Draining and resetting system...")
        self.cl.drain_queues()
        self.predictor.client.stop()
        del self.predictor
        self.predictor = None
        if wait:
            time.sleep(10)
            self.cl.drain_queues()
            time.sleep(10)
    ########################################################################################################

    def warm_up_system(self):
        logger.info("Warming up system...")
        self.init_predictor()
        idx = 0
        # First warm up the model.
        # NOTE: The length of time the model needs to warm up for
        # seems to be both framework and hardware dependent. 27 seems to work
        # well for PyTorch resnet, but we don't need as many warmup
        # iterations for the JIT-free models in this pipeline

        while len(self.predictor.stats["thrus"]) < 8:
            self.run_predictor()
        self.reset_predictor()

    # start with an overly aggressive request rate
    # then back off
    def initialize_request_rate(self):
        self.init_predictor()
        initialization_iterations = 50
        while len(self.predictor.stats["thrus"]) < initialization_iterations:
            self.run_predictor()
        # Now initialize request rate
        max_thruput = np.max(self.predictor.stats["thrus"])
        logger.info("mean_throughput measured to be {}".format(max_thruput))
        self.delay = ArrivalProcess("constant", 1, 1.0 / max_thruput)
        self.reset_predictor()

    def find_steady_state(self):

        self.init_predictor()
        time.sleep(10) # just to be extra sure I guess

        idx = 0
        done = False
        # start checking for steady state after 10 trials
        last_checked_length = 10
        while not done:
            self.run_predictor()
            if len(self.predictor.stats["thrus"]) > last_checked_length:
                last_checked_length = len(self.predictor.stats["thrus"]) + 1
                convergence_state = driver_utils.check_convergence_via_queue(
                    self.predictor.stats, self.configs, self.latency_upper_bound)
                # Diverging, try again with higher
                # delay
                if convergence_state == INCREASING or convergence_state == CONVERGED_HIGH:
                    self.delay.increase_delay()
                    logger.info("Increasing delay to {}".format(self.delay))
                    done = True
                    self.reset_predictor(wait=False)
                    return self.find_steady_state()
                elif convergence_state == CONVERGED:
                    logger.info("Converged with delay of {}".format(self.delay))
                    done = True
                    self.reset_predictor()
                    return
                elif len(self.predictor.stats) > 40:
                    self.delay.increase_delay()
                    logger.info("Increasing delay to {}".format(self.delay))
                    done = True
                    self.reset_predictor(wait=False)
                    return self.find_steady_state()
                elif convergence_state == DECREASING or convergence_state == UNKNOWN:
                    logger.info("Not converged yet. Still waiting")
                elif convergence_state == CONVERGED_LOW:
                    logger.info("Converged LOW with delay of {}".format(self.delay))
                    logger.info("Consider re-running with smaller request delay augmentations")
                    raise Exception("LOW_CONVERGE, Consider re-running with smaller request delay augmentations")
                    done = True
                    self.reset_predictor(wait=False)
                    return
                else:
                    logger.error("Unknown convergence state: {}".format(convergence_state))
                    sys.exit(1)

    def run_more_predictions(self):
        self.init_predictor()
        for i in xrange(40000):
            if ((i+1) % 100) == 0:
                logger.info("Iteration "+str(i)+"...")
            self.run_predictor()
        time.sleep(10)
        predictor_stats = self.predictor.stats
        predictor_stats["used_delay"] = str(self.delay)
        return predictor_stats

    # new process
    def run(self):
        # initialize delay to be very small
        self.delay = ArrivalProcess("constant", 1, 0.001)
        self.queries_per_sleep = 1
        self.clipper_address = setup_clipper(self.configs)
        self.cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        self.cl.connect()
        # The observed amount of time needed to wait until the TF graphs are loaded on the GPU
        # This is time where progress isn't happening anyway, and we want to avoid making excessive prediction calls to clipper
        # and flood the frontend queue before the models are set up.
        time.sleep(230)
        logger.info("Waited 30 seconds to connect to clipper")
        self.warm_up_system()
        if self.input_delay_process != None:
            self.delay = self.input_delay_process
            self.delay.scale_delay()
        else:
            logger.info("Initializing request rate (AKA delay)...")
            self.initialize_request_rate()
            logger.info("Initializing delay {delay}".format(delay=self.delay))
            self.delay.scale_delay()
            logger.info("Scaling delay {delay}".format(delay=self.delay))
            self.find_steady_state()
            logger.info("Found steady state...")
        logger.info("Initializing delay {delay}".format(delay=self.delay))
        final_stats = self.run_more_predictions()
        self.queue.put(final_stats)
        return
