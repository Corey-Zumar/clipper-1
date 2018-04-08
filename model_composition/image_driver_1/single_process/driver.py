import sys
import os
import argparse
import numpy as np
import json
import logging
import Queue

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Thread, Lock

from single_proc_utils import HeavyNodeConfig, save_results
from models import tf_resnet_model, inception_feats_model, tf_kernel_svm_model, tf_log_reg_model

from e2e_utils import load_arrival_deltas, calculate_mean_throughput, calculate_peak_throughput

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(CURR_DIR, "models")

INCEPTION_FEATS_MODEL_NAME = "inception_feats"
TF_KERNEL_SVM_MODEL_NAME = "kernel_svm"
TF_LOG_REG_MODEL_NAME = "tf_log_reg"
TF_RESNET_MODEL_NAME = "tf_resnet_feats"

RESULTS_DIR = "/results"

INCEPTION_MODEL_PATH = os.path.join(MODELS_DIR, "inception_model_data", "inception_feats_graph_def.pb")
RESNET_MODEL_PATH = os.path.join(MODELS_DIR, "tf_resnet_model_data")

########## Setup ##########

def get_heavy_node_configs(batch_size, allocated_cpus, resnet_gpus=[], inception_gpus=[]):
    resnet_config = HeavyNodeConfig(model_name=TF_RESNET_MODEL_NAME,
                                    input_type="floats",
                                    allocated_cpus=allocated_cpus,
                                    gpus=resnet_gpus,
                                    batch_size=batch_size)

    inception_config = HeavyNodeConfig(model_name=INCEPTION_FEATS_MODEL_NAME,
                                       input_type="floats",
                                       allocated_cpus=allocated_cpus,
                                       gpus=inception_gpus,
                                       batch_size=batch_size)

    kernel_svm_config = HeavyNodeConfig(model_name=TF_KERNEL_SVM_MODEL_NAME,
                                        input_type="floats",
                                        allocated_cpus=allocated_cpus,
                                        gpus=[],
                                        batch_size=batch_size)

    log_reg_config = HeavyNodeConfig(model_name=TF_LOG_REG_MODEL_NAME,
                                     input_type="floats",
                                     allocated_cpus=allocated_cpus,
                                     gpus=[],
                                     batch_size=batch_size)

    return [resnet_config, inception_config, kernel_svm_config, log_reg_config]

def create_resnet_model(model_path, gpu_num):
    return tf_resnet_model.TfResNetModel(model_path, gpu_num)

def create_kernel_svm_model():
    return tf_kernel_svm_model.TFKernelSVM()

def create_inception_model(model_path, gpu_num):
    return inception_feats_model.InceptionFeaturizationModel(model_path, gpu_num=gpu_num)

def create_log_reg_model():
    return tf_log_reg_model.TfLogRegModel()

def load_models(resnet_gpu, inception_gpu):
    models_dict = {
        TF_RESNET_MODEL_NAME : create_resnet_model(RESNET_MODEL_PATH, gpu_num=resnet_gpu),
        TF_KERNEL_SVM_MODEL_NAME : create_kernel_svm_model(),
        INCEPTION_FEATS_MODEL_NAME : create_inception_model(INCEPTION_MODEL_PATH, gpu_num=inception_gpu),
        TF_LOG_REG_MODEL_NAME : create_log_reg_model()
    }
    return models_dict

########## Benchmarking ##########

class Predictor(object):

    def __init__(self, models_dict, trial_length):
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        # Stats
        self.init_stats()
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "mean_lats": []
        }
        self.total_num_complete = 0
        self.trial_length = trial_length

        # Models
        self.resnet_model = models_dict[TF_RESNET_MODEL_NAME]
        self.kernel_svm_model = models_dict[TF_KERNEL_SVM_MODEL_NAME]
        self.inception_model = models_dict[INCEPTION_FEATS_MODEL_NAME]
        self.log_reg_model = models_dict[TF_LOG_REG_MODEL_NAME]

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
        logger.info("p99: {p99}, mean: {mean}, thruput: {thru}".format(p99=p99,
                                                                       mean=mean,
                                                                       thru=thru))

    def predict(self, send_times, resnet_inputs, inception_inputs):
        """
        Parameters
        ------------
        send_times : [datetime]
            A list of timestamps at which each input was sent
        resnet_inputs : [np.ndarray]
            A list of image inputs, each represented as a numpy array
            of shape 224 x 224 x 3
        inception_inputs : [np.ndarray]
            A list of image inputs, each represented as a numpy array
            of shape 299 x 299 x 3
        """
        assert len(send_times) == len(resnet_inputs) == len(inception_inputs)

        batch_size = len(resnet_inputs)

        resnet_svm_future = self.thread_pool.submit(
            lambda inputs : self.kernel_svm_model.predict(self.resnet_model.predict(inputs)), resnet_inputs)
        
        inception_log_reg_future = self.thread_pool.submit(
            lambda inputs : self.log_reg_model.predict(self.inception_model.predict(inputs)), inception_inputs)

        # resnet_svm_classes = resnet_svm_future.result()
        inception_log_reg_classes = inception_log_reg_future.result()

        end_time = datetime.now()

        for send_time in send_times:
            latency = (end_time - send_time).total_seconds()
            self.latencies.append(latency)
            self.total_num_complete += 1 
            self.trial_num_complete += 1 
            if self.trial_num_complete % self.trial_length == 0:
                self.print_stats()
                self.init_stats()

class DriverBenchmarker(object):
    def __init__(self, models_dict, trial_length):
        self.models_dict = models_dict
        self.trial_length = trial_length
        self.request_queue = Queue.Queue()

    def set_configs(self, configs):
        self.configs = configs

    def run(self, num_trials, batch_size, num_cpus, replica_num, process_file=None):
        if process_file: 
            self._benchmark_arrival_process(replica_num, num_trials, batch_size, process_file)
        else:
            self._benchmark_batches(replica_num, num_trials, batch_size)

    def _benchmark_batches(self, replica_num, num_trials, batch_size):
        logger.info("Generating random inputs")
        resnet_inputs, inception_inputs = self._generate_inputs()

        logger.info("Starting predictions")
        while True:
            send_time = datetime.now()
            send_times = [send_time for _ in range(batch_size)]
            
            batch_idx = np.random.randint(len(resnet_inputs) - batch_size)
            resnet_batch = resnet_inputs[batch_idx : batch_idx + batch_size]
            inception_batch = inception_inputs[batch_idx : batch_idx + batch_size]

            predictor.predict(send_times, resnet_batch, inception_batch)

            if len(predictor.stats["thrus"]) > num_trials:
                break

        save_results(self.configs, [predictor.stats], "single_proc_bs_{}_bench".format(batch_size), replica_num)

    def _benchmark_arrival_process(self, replica_num, num_trials, batch_size, process_file):
        logger.info("Parsing arrival process")
        arrival_process = load_arrival_deltas(process_file)
        mean_throughput = calculate_mean_throughput(arrival_process)
        peak_throughput = calculate_peak_throughput(arrival_process)

        print("Mean throughput: {}\nPeak throughput: {}".format(mean_throughput, peak_throughput))

        logger.info("Generating random inputs")
        resnet_inputs, inception_inputs = self._generate_inputs()

        logger.info("Initializing processor thread")
        processor_thread = Thread(target=self._run_async_query_processor, args=(replica_num, num_trials, batch_size, process_file))
        processor_thread.start()
        
        logger.info("Starting predictions")

        for idx in range(len(arrival_process)):
            send_time = datetime.now()
            input_idx = np.random.randint(len(inception_inputs))
            resnet_input = resnet_inputs[input_idx]
            inception_input = inception_inputs[input_idx]
            self.request_queue.put(send_time, resnet_input, inception_input)

            request_delay = arrival_process[idx] * .001
            time.sleep(request_delay)

        processor_thread.join()

    def _run_async_query_processor(self, replica_num, num_trials, batch_size, process_file):
        predictor = Predictor(self.models_dict, trial_length=self.trial_length)
        while True:
            curr_batch = []
            batch_item = self.request_queue.get(block=True)
            while len(cur_batch) < batch_size and (not self.request_queue.empty()):
                batch_item = self.requet_queue.get_nowait()
                curr_batch.append(batch_item)

            send_times, resnet_inputs, inception_inputs = zip(*curr_batch)
            predictor.predict(send_times, resnet_inputs, inception_inputs)

            if len(predictor.stats["thrus"]) > num_trials:
                break

        save_results(self.configs, [predictor.stats], "single_proc_arrival_procs", replica_num, process_file)
        sys.exit(0)

    def _generate_inputs(self):
        resnet_inputs = [self._get_resnet_feats_input() for _ in range(1000)]
        resnet_inputs = [i for _ in range(40) for i in resnet_inputs]

        inception_inputs = [self._get_inception_input() for _ in range(1000)]
        inception_inputs = [i for _ in range(40) for i in inception_inputs]

        assert len(inception_inputs) == len(resnet_inputs)
        return resnet_inputs, inception_inputs

    def _get_resnet_feats_input(self):
        resnet_input = np.array(np.random.rand(224, 224, 3) * 255, dtype=np.float32)
        return resnet_input.flatten()

    def _get_inception_input(self):
        inception_input = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
        return inception_input.flatten()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Single Process Image Driver 1')
    parser.add_argument('-b',  '--batch_sizes', type=int, nargs='+', help="The batch size configurations to benchmark for the driver. Each configuration will be benchmarked separately.")
    parser.add_argument('-c',  '--cpus', type=int, nargs='+', help="The set of VIRTUAL cpu cores on which to run the single process driver")
    parser.add_argument('-r',  '--resnet_gpu', type=int, default=0, help="The GPU on which to run the ResNet 152 featurization model")
    parser.add_argument('-i',  '--inception_gpu', type=int, default=0, help="The GPU on which to run the inception featurization model")
    parser.add_argument('-t',  '--num_trials', type=int, default=15, help="The number of trials to run")
    parser.add_argument('-tl', '--trial_length', type=int, default=200, help="The length of each trial, in requests")
    parser.add_argument('-n',  '--replica_num', type=int, help="The replica number corresponding to the driver")
    parser.add_argument('-p',  '--process_file', type=str, help="Path to an arrival process file")
    
    args = parser.parse_args()

    arrival_process = None

    if not args.cpus:
        raise Exception("The set of allocated cpus must be specified via the '--cpus' flag!")

    default_batch_size_confs = [2]
    batch_size_confs = args.batch_sizes if args.batch_sizes else default_batch_size_confs
    
    models_dict = load_models(args.resnet_gpu, args.inception_gpu)
    benchmarker = DriverBenchmarker(models_dict, args.trial_length)

    num_cpus = len(args.cpus)

    for batch_size in batch_size_confs:
        configs = get_heavy_node_configs(batch_size=batch_size,
                                         allocated_cpus=args.cpus,
                                         resnet_gpus=[args.resnet_gpu],
                                         inception_gpus=[args.inception_gpu])
        benchmarker.set_configs(configs)
        benchmarker.run(args.num_trials, batch_size, num_cpus, args.replica_num, args.process_file)
