import sys
import os
import argparse
import numpy as np
import time
import base64
import logging

from clipper_admin import ClipperConnection, DockerContainerManager
from datetime import datetime
from PIL import Image
from containerized_utils.zmq_client import Client
from containerized_utils import driver_utils
from containerized_utils.driver_utils import INCREASING, DECREASING, CONVERGED_HIGH, CONVERGED, UNKNOWN
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
INCEPTION_FEATS_MODEL_APP_NAME = "inception"
LGBM_MODEL_APP_NAME = "lgbm"

TF_KERNEL_SVM_MODEL_APP_NAME = "tf-kernel-svm"
TF_LOG_REG_MODEL_APP_NAME = "tf-log-reg"
TF_RESNET_MODEL_APP_NAME = "tf-resnet-feats"

VGG_FEATS_IMAGE_NAME = "model-comp/vgg-feats"
VGG_KPCA_SVM_IMAGE_NAME = "model-comp/kpca-svm"
VGG_KERNEL_SVM_IMAGE_NAME = "model-comp/kernel-svm"
VGG_ELASTIC_NET_IMAGE_NAME = "model-comp/elastic-net"
INCEPTION_FEATS_IMAGE_NAME = "model-comp/inception-feats"
LGBM_IMAGE_NAME = "model-comp/lgbm"

TF_KERNEL_SVM_IMAGE_NAME = "model-comp/tf-kernel-svm"
TF_LOG_REG_IMAGE_NAME = "model-comp/tf-log-reg"
TF_RESNET_IMAGE_NAME = "model-comp/tf-resnet-feats"

VALID_MODEL_NAMES = [
    VGG_FEATS_MODEL_APP_NAME,
    VGG_KPCA_SVM_MODEL_APP_NAME,
    VGG_KERNEL_SVM_MODEL_APP_NAME,
    VGG_ELASTIC_NET_MODEL_APP_NAME,
    INCEPTION_FEATS_MODEL_APP_NAME,
    LGBM_MODEL_APP_NAME,
    TF_KERNEL_SVM_MODEL_APP_NAME,
    TF_LOG_REG_MODEL_APP_NAME,
    TF_RESNET_MODEL_APP_NAME
]

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

def get_heavy_node_config(model_name, batch_size, num_replicas, cpus_per_replica=None, allocated_cpus=None, allocated_gpus=None):
    if model_name == VGG_FEATS_MODEL_APP_NAME:
        if not cpus_per_replica:
            cpus_per_replica = 2
        if not allocated_cpus:
            allocated_cpus = [6,7,14,15]
        if not allocated_gpus:
            allocated_gpus = [0]

        return driver_utils.HeavyNodeConfig(name=VGG_FEATS_MODEL_APP_NAME,
                                            input_type="floats",
                                            model_image=VGG_FEATS_IMAGE_NAME,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True)

    elif model_name == INCEPTION_FEATS_MODEL_APP_NAME:
        if not cpus_per_replica:
            cpus_per_replica = 1
        if not allocated_cpus:
            allocated_cpus = range(16,19)
        if not allocated_gpus:
            allocated_gpus = [0]

        return driver_utils.HeavyNodeConfig(name=INCEPTION_FEATS_MODEL_APP_NAME,
                                            input_type="floats",
                                            model_image=INCEPTION_FEATS_IMAGE_NAME,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True)

    elif model_name == VGG_KPCA_SVM_MODEL_APP_NAME:
        if not cpus_per_replica:
            cpus_per_replica = 2
        if not allocated_cpus:
            allocated_cpus = range(20,27)
        if not allocated_gpus:
            allocated_gpus = []

        return driver_utils.HeavyNodeConfig(name=VGG_KPCA_SVM_MODEL_APP_NAME,
                                            input_type="floats",
                                            model_image=VGG_KPCA_SVM_IMAGE_NAME,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            no_diverge=True)

    elif model_name == VGG_KERNEL_SVM_MODEL_APP_NAME:
        if not cpus_per_replica:
            cpus_per_replica = 1
        if not allocated_cpus:
            allocated_cpus = range(20,27)
        if not allocated_gpus:
            allocated_gpus = []
        return driver_utils.HeavyNodeConfig(name=VGG_KERNEL_SVM_MODEL_APP_NAME,
                                            input_type="floats",
                                            model_image=VGG_KERNEL_SVM_IMAGE_NAME,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=False,
                                            no_diverge=True)

    elif model_name == VGG_ELASTIC_NET_MODEL_APP_NAME:
        if not cpus_per_replica:
            cpus_per_replica = 1
        if not allocated_cpus:
            allocated_cpus = range(20,27)
        if not allocated_gpus:
            allocated_gpus = []
        return driver_utils.HeavyNodeConfig(name=VGG_ELASTIC_NET_MODEL_APP_NAME,
                                            input_type="floats",
                                            model_image=VGG_ELASTIC_NET_IMAGE_NAME,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=False,
                                            no_diverge=True)     


    elif model_name == LGBM_MODEL_APP_NAME:
        if not cpus_per_replica:
            cpus_per_replica = 1
        if not allocated_cpus:
            allocated_cpus = [28,29]
        if not allocated_gpus:
            allocated_gpus = []

        return driver_utils.HeavyNodeConfig(name=LGBM_MODEL_APP_NAME,
                                            input_type="floats",
                                            model_image=LGBM_IMAGE_NAME,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=False,
                                            no_diverge=True)

    elif model_name == TF_KERNEL_SVM_MODEL_APP_NAME:
        if not cpus_per_replica:
            cpus_per_replica = 1
        if not allocated_cpus:
            allocated_cpus = [20]
        if not allocated_gpus:
            allocated_gpus = []

        return driver_utils.HeavyNodeConfig(name=TF_KERNEL_SVM_MODEL_APP_NAME,
                                            input_type="floats",
                                            model_image=TF_KERNEL_SVM_IMAGE_NAME,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True)

    elif model_name == TF_LOG_REG_MODEL_APP_NAME:
        if not cpus_per_replica:
            cpus_per_replica = 1
        if not allocated_cpus:
            allocated_cpus = [20]
        if not allocated_gpus:
            allocated_gpus = []

        return driver_utils.HeavyNodeConfig(name=TF_LOG_REG_MODEL_APP_NAME,
                                            input_type="floats",
                                            model_image=TF_LOG_REG_IMAGE_NAME,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True)

    elif model_name == TF_RESNET_MODEL_APP_NAME:
        if not cpus_per_replica:
            cpus_per_replica = 1
        if not allocated_cpus:
            allocated_cpus = [20]
        if not allocated_gpus:
            allocated_gpus = [1]

        return driver_utils.HeavyNodeConfig(name=TF_RESNET_MODEL_APP_NAME,
                                            input_type="floats",
                                            model_image=TF_RESNET_IMAGE_NAME,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True)


########## Benchmarking ##########

class Predictor(object):

    def __init__(self, clipper_metrics):
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
        self.stats["p99_lats"].append(p99)
        self.stats["mean_lats"].append(mean)
        self.stats["all_lats"].append(self.latencies)
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
            if self.batch_num_complete % 50 == 0:
                self.print_stats()
                self.init_stats()

        return self.client.send_request(model_app_name, input_item).then(continuation)

class ModelBenchmarker(object):
    def __init__(self, config, queue):
        self.config = config
        self.queue = queue
        self.input_generator_fn = self._get_input_generator_fn(model_app_name=self.config.name)
        logger.info("Generating random inputs")
        base_inputs = [self.input_generator_fn() for _ in range(1000)]
        self.inputs = [i for _ in range(60) for i in base_inputs]

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

    def _get_vgg_feats_input(self):
        input_img = np.array(np.random.rand(224, 224, 3) * 255, dtype=np.float32)
        return input_img.flatten()

    def _get_vgg_classifier_input(self):
        return np.array(np.random.rand(4096), dtype=np.float32)

    def _get_inception_input(self):
        input_img = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
        return input_img.flatten()

    def _get_lgbm_input(self):
        return np.array(np.random.rand(2048), dtype=np.float32)

    def _get_tf_kernel_svm_input(self):
        return np.array(np.random.rand(2048), dtype=np.float32)

    def _get_tf_log_reg_input(self):
        return np.array(np.random.rand(2048), dtype=np.float32)

    def _get_tf_resnet_input(self):
        input_img = np.array(np.random.rand(224, 224, 3) * 255, dtype=np.float32)
        return input_img.flatten()

    def _get_input_generator_fn(self, model_app_name):
        if model_app_name == VGG_FEATS_MODEL_APP_NAME:
            return self._get_vgg_feats_input
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Clipper image driver 1')
    parser.add_argument('-m', '--model_name', type=str, help="The name of the model to benchmark. One of: 'vgg', 'kpca-svm', 'kernel-svm', 'elastic-net', 'inception', 'lgbm', 'tf-kernel-svm', 'tf-log-reg', 'tf-resnet-feats'")
    parser.add_argument('-b', '--batch_sizes', type=int, nargs='+', help="The batch size configurations to benchmark for the model. Each configuration will be benchmarked separately.")
    parser.add_argument('-r', '--num_replicas', type=int, nargs='+', help="The replica number configurations to benchmark for the model. Each configuration will be benchmarked separately.")
    parser.add_argument('-c', '--model_cpus', type=int, nargs='+', help="The set of cpu cores on which to run replicas of the provided model")
    parser.add_argument('-p', '--cpus_per_replica_nums', type=int, nargs='+', help="Configurations for the number of cpu cores allocated to each replica of the model")
    parser.add_argument('-g', '--model_gpus', type=int, nargs='+', help="The set of gpus on which to run replicas of the provided model. Each replica of a gpu model must have its own gpu!")
    parser.add_argument('-n', '--num_clients', type=int, default=1, help="The number of concurrent client processes. This can help increase the request rate in order to saturate high throughput models.")
    
    args = parser.parse_args()

    if args.model_name not in VALID_MODEL_NAMES:
        raise Exception("Model name must be one of: {}".format(VALID_MODEL_NAMES))

    default_batch_size_confs = [2]
    default_replica_num_confs = [1]
    default_cpus_per_replica_confs = [None]

    batch_size_confs = args.batch_sizes if args.batch_sizes else default_batch_size_confs
    replica_num_confs = args.num_replicas if args.num_replicas else default_replica_num_confs
    cpus_per_replica_confs = args.cpus_per_replica_nums if args.cpus_per_replica_nums else default_cpus_per_replica_confs


    for num_replicas in replica_num_confs:
        for cpus_per_replica in cpus_per_replica_confs:
            for batch_size in batch_size_confs:
                model_config = get_heavy_node_config(model_name=args.model_name, 
                                                     batch_size=batch_size, 
                                                     num_replicas=num_replicas,
                                                     cpus_per_replica=cpus_per_replica,
                                                     allocated_cpus=args.model_cpus,                               
                                                     allocated_gpus=args.model_gpus)
                queue = Queue()
                benchmarker = ModelBenchmarker(model_config, queue)

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
                driver_utils.save_results([model_config], cl, all_stats, "gpu_and_batch_size_experiments")


