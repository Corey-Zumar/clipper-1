import sys
import os
import argparse
import numpy as np
import time
import base64
import logging

from clipper_admin import ClipperConnection, DockerContainerManager
from threading import Lock
from datetime import datetime
from io import BytesIO
from PIL import Image
from containerized_utils.zmq_client import Client
from containerized_utils import driver_utils
from multiprocessing import Process

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

VGG_FEATS_IMAGE_NAME = "model-comp/vgg-feats"
VGG_KPCA_SVM_IMAGE_NAME = "model-comp/kpca-svm"
VGG_KERNEL_SVM_IMAGE_NAME = "model-comp/kernel-svm"
VGG_ELASTIC_NET_IMAGE_NAME = "model-comp/elastic-net"
INCEPTION_FEATS_IMAGE_NAME = "model-comp/inception-feats"
LGBM_IMAGE_NAME = "model-comp/lgbm"

VALID_MODEL_NAMES = [
    VGG_FEATS_MODEL_APP_NAME,
    VGG_KPCA_SVM_MODEL_APP_NAME,
    VGG_KERNEL_SVM_MODEL_APP_NAME,
    VGG_ELASTIC_NET_MODEL_APP_NAME,
    INCEPTION_FEATS_MODEL_APP_NAME,
    LGBM_MODEL_APP_NAME
]

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
        mgmt_cpu_str="8",
        query_cpu_str="1-5,9-13")
    time.sleep(10)
    for config in configs:
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
                                            num_replicas=num_replicas)

    elif model_name == INCEPTION_FEATS_MODEL_APP_NAME:
        if not cpus_per_replica:
            cpus_per_replica = 1
        if not allocated_cpus:
            allocated_cpus = range(16,19)
        if not allocated_gpus:
            allocated_gpus = [0]

        return driver_utils.HeavyNodeConfig(name=INCEPTION_FEATS_MODEL_APP_NAME,
                                            input_type="strings",
                                            model_image=INCEPTION_FEATS_IMAGE_NAME,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas)

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
                                            num_replicas=num_replicas)

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
                                            num_replicas=num_replicas)

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
                                            num_replicas=num_replicas)        


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
                                            num_replicas=num_replicas)


########## Benchmarking ##########

class Predictor(object):

    def __init__(self):
        self.outstanding_reqs = {}
        self.client = Client(CLIPPER_ADDRESS, CLIPPER_SEND_PORT, CLIPPER_RECV_PORT)
        self.client.start()
        self.init_stats()
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "mean_lats": []}
        self.total_num_complete = 0

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
        logger.info("p99: {p99}, mean: {mean}, thruput: {thru}".format(p99=p99,
                                                                       mean=mean,
                                                                       thru=thru))

    def predict(self, vgg_input, inception_input):
        begin_time = datetime.now()
        classifications_lock = Lock()
        classifications = {}

        def update_perf_stats():
            end_time = datetime.now()
            latency = (end_time - begin_time).total_seconds()
            self.latencies.append(latency)
            self.total_num_complete += 1
            self.batch_num_complete += 1
            if self.batch_num_complete % 50 == 0:
                self.print_stats()
                self.init_stats()

        def vgg_feats_continuation(vgg_features):
            if vgg_features == DEFAULT_OUTPUT:
                return
            return self.client.send_request(VGG_KERNEL_SVM_MODEL_APP_NAME, vgg_features)

        def svm_continuation(svm_classification):
            if svm_classification == DEFAULT_OUTPUT:
                return
            else:
                classifications_lock.acquire()
                if LGBM_MODEL_APP_NAME not in classifications:
                    classifications[VGG_KERNEL_SVM_MODEL_APP_NAME] = svm_classification
                else:
                    update_perf_stats()
                classifications_lock.release()

        def inception_feats_continuation(inception_features);
            if inception_features == DEFAULT_OUTPUT:
                return
            return self.client.send_request(LGBM_MODEL_APP_NAME, inception_features)

        def lgbm_continuation(lgbm_classification):
            if lgbm_classification == DEFAULT_OUTPUT:
                return
            else:
                classifications_lock.acquire()
                if VGG_KERNEL_SVM_MODEL_APP_NAME not in classifications:
                    classifications[LGBM_MODEL_APP_NAME] = lgbm_classification
                else:
                    update_perf_stats()
                classifications_lock.release()

        return self.client.send_request(model_app_name, input_item).then(continuation)

class DriverBenchmarker(object):
    def __init__(self, config):
        self.config = config

    def run(self, duration_seconds=120, request_delay=.01):
        logger.info("Generating random inputs")
        inputs = [(self._get_vgg_feats_input(), self._get_inception_input()) for _ in range(10000)]
        logger.info("Starting predictions")
        start_time = datetime.now()
        predictor = Predictor()
        for vgg_feats_input, inception_input in inputs:
            predictor.predict(vgg_feats_input, inception_input)
            time.sleep(request_delay)
        while True:
            curr_time = datetime.now()
            if ((curr_time - start_time).total_seconds() > duration_seconds) or (predictor.total_num_complete == 10000):
                break
            time.sleep(1)

        cl = ClipperConnection(DockerContainerManager(redis_port=6380))
        cl.connect()
        driver_utils.save_results([self.config], cl, predictor.stats, "gpu_and_batch_size_experiments")

    def _get_vgg_feats_input(self):
        input_img = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
        input_img = Image.fromarray(input_img.astype(np.uint8))
        vgg_img = input_img.resize((224, 224)).convert('RGB')
        vgg_input = np.array(vgg_img, dtype=np.float32)
        return vgg_input.flatten()

    def _get_inception_input(self):
        input_img = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
        input_img = Image.fromarray(input_img.astype(np.uint8))
        inmem_inception_jpeg = BytesIO()
        resized_inception = input_img.resize((299,299)).convert('RGB')
        resized_inception.save(inmem_inception_jpeg, format="JPEG")
        inmem_inception_jpeg.seek(0)
        inception_input = inmem_inception_jpeg.read()
        return base64.b64encode(inception_input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Clipper image driver 1')
    parser.add_argument('-d', '--duration', type=int, default=120, help='The maximum duration of the benchmarking process in seconds, per iteration')
    parser.add_argument('-b', '--batch_sizes', type=int, nargs='+', help="The batch size configurations to benchmark for the model. Each configuration will be benchmarked separately.")
    parser.add_argument('-r', '--num_replicas', type=int, nargs='+', help="The replica number configurations to benchmark for the model. Each configuration will be benchmarked separately.")
    parser.add_argument('-c', '--model_cpus', type=int, nargs='+', help="The set of cpu cores on which to run replicas of the provided model")

    args = parser.parse_args()

    vgg_feats_config = get_heavy_node_config(model_name=VGG_FEATS_MODEL_APP_NAME, 
                                             batch_size=32, 
                                             num_replicas=1, 
                                             cpus_per_replica=1, 
                                             allocated_cpus=[14], 
                                             allocated_gpus=[0])

    vgg_svm_config = get_heavy_node_config(model_name=VGG_KERNEL_SVM_MODEL_APP_NAME, 
                                           batch_size=32, 
                                           num_replicas=1, 
                                           cpus_per_replica=1, 
                                           allocated_cpus=[15], 
                                           allocated_gpus=[])

    inception_feats_config = get_heavy_node_config(model_name=INCEPTION_FEATS_MODEL_APP_NAME, 
                                                   batch_size=32, 
                                                   num_replicas=1, 
                                                   cpus_per_replica=1, 
                                                   allocated_cpus=[16], 
                                                   allocated_gpus=[1])

    lgbm_config = get_heavy_node_config(model_name=INCEPTION_FEATS_MODEL_APP_NAME, 
                                        batch_size=32, 
                                        num_replicas=1, 
                                        cpus_per_replica=1, 
                                        allocated_cpus=[17], 
                                        allocated_gpus=[])

    model_configs = [vgg_feats_config, vgg_svm_config, inception_feats_config, lgbm_config]


    #for request_delay in range(.01, .1, .01):
    setup_clipper(configs)
    benchmarker = DriverBenchmarker(model_configs)
    p = Process(target=benchmarker.run, args=(args.duration, .01))
    p.start()
    p.join()

