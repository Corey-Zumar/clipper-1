import sys
import os
import argparse
import numpy as np
import json
import logging
import Queue
import time

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from datetime import timedelta 
from threading import Thread, Lock

from single_proc_utils import HeavyNodeConfig, save_results
from models import tf_resnet_model, inception_feats_model, tf_kernel_svm_model, tf_log_reg_model

from e2e_utils import load_tagged_arrival_deltas, load_arrival_deltas 
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
                                    num_replicas=0,
                                    allocated_cpus=allocated_cpus,
                                    gpus=resnet_gpus,
                                    batch_size=batch_size)

    inception_config = HeavyNodeConfig(model_name=INCEPTION_FEATS_MODEL_NAME,
                                       input_type="floats",
                                       num_replicas=0,
                                       allocated_cpus=allocated_cpus,
                                       gpus=inception_gpus,
                                       batch_size=batch_size)

    kernel_svm_config = HeavyNodeConfig(model_name=TF_KERNEL_SVM_MODEL_NAME,
                                        input_type="floats",
                                        num_replicas=0,
                                        allocated_cpus=allocated_cpus,
                                        gpus=[],
                                        batch_size=batch_size)

    log_reg_config = HeavyNodeConfig(model_name=TF_LOG_REG_MODEL_NAME,
                                     input_type="floats",
                                     num_replicas=0,
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

########## Input Generation ##########

def generate_inputs():
    resnet_inputs = [get_resnet_feats_input() for _ in range(1000)]
    resnet_inputs = [i for _ in range(40) for i in resnet_inputs]

    inception_inputs = [get_inception_input() for _ in range(1000)]
    inception_inputs = [i for _ in range(40) for i in inception_inputs]

    assert len(inception_inputs) == len(resnet_inputs)
    return np.array(resnet_inputs), np.array(inception_inputs)

def get_resnet_feats_input():
    resnet_input = np.array(np.random.rand(224, 224, 3) * 255, dtype=np.float32)
    return resnet_input.flatten()

def get_inception_input():
    inception_input = np.array(np.random.rand(299, 299, 3) * 255, dtype=np.float32)
    return inception_input.flatten()

########## Arrival Processes ##########
def condense_delays(arrival_process, replica_num):
    condensed_process = []
    idx = 0
    curr_delay = 0
    while idx < len(arrival_process):
        request_delay_millis, request_replica_num = arrival_process[idx]
        if request_replica_num == replica_num:
            condensed_process.append((curr_delay + request_delay_millis, replica_num))
            curr_delay = 0
        else:
            curr_delay += request_delay_millis

        idx += 1

    return condensed_process

########## Benchmarking ##########

class Predictor(object):

    def __init__(self, models_dict, trial_length):
        self.task_execution_thread_pool = ThreadPoolExecutor(max_workers=2)
        self.stats_thread_pool = ThreadPoolExecutor(max_workers=2)

        # Stats
        self.init_stats()
        self.stats = {
            "thrus": [],
            "p99_lats": [],
            "mean_lats": [],
            "all_lats": [],
            "p99_batch_predict_lats": [],
            "p99_queue_lats": [],
            "mean_batch_sizes": [],
            "per_message_lats": {}
        }
        self.total_num_complete = 0
        self.trial_length = trial_length

        # Models
        self.resnet_model = models_dict[TF_RESNET_MODEL_NAME]
        self.kernel_svm_model = models_dict[TF_KERNEL_SVM_MODEL_NAME]
        self.inception_model = models_dict[INCEPTION_FEATS_MODEL_NAME]
        self.log_reg_model = models_dict[TF_LOG_REG_MODEL_NAME]

        # Input generation
        logger.info("Generating random inputs")
        self.resnet_inputs, self.inception_inputs = generate_inputs()
        new_resnet = {}
        new_inception = {}
        for bs in range(200):
            new_resnet[bs] = self.resnet_inputs[xrange(bs)]
            new_inception[bs] = self.inception_inputs[xrange(bs)]
        self.resnet_inputs = new_resnet
        self.inception_inputs = new_inception

        logger.info("Warming up")
        self.warm_up()

    def warm_up(self):
        for _ in range(100):
            time = datetime.now()
            self.predict([(msg_id, time) for msg_id in range(10)])

    def init_stats(self):
        self.latencies = []
        self.batch_predict_latencies = []
        self.batch_sizes = []
        self.trial_num_complete = 0
        self.cur_req_id = 0
        self.start_time = datetime.now()

    def print_stats(self):
        lats = np.array(self.latencies)
        batch_predict_lats = np.array(self.batch_predict_latencies)
        p99 = np.percentile(lats, 99)
        p99_batch_predict = np.percentile(batch_predict_lats, 99)
        mean_batch_size = np.mean(self.batch_sizes)
        mean = np.mean(lats)
        end_time = datetime.now()
        thru = float(self.trial_num_complete) / (end_time - self.start_time).total_seconds()
        self.stats["thrus"].append(thru)
        self.stats["all_lats"].append(self.latencies)
        self.stats["p99_lats"].append(p99)
        self.stats["mean_lats"].append(mean)
        self.stats["p99_batch_predict_lats"].append(self.batch_predict_latencies)
        self.stats["mean_batch_sizes"].append(mean_batch_size)
        logger.info("p99_lat: {p99}, mean_lat: {mean}, p99_batch_predict: {p99_batch_pred},"
                    " thruput: {thru}, mean_batch: {mb}".format(p99=p99,
                                                          mean=mean,
                                                          thru=thru, 
                                                          p99_batch_pred=p99_batch_predict,
                                                          mb=mean_batch_size))

    def predict(self, requests):
        """
        Parameters
        ------------
        # msg_ids : [int]
        #     A list of message ids

        # send_times : [datetime]
        #     A list of send times

        requests : [(int, datetime)]
            A list of (msg_id, send_time) tuples
        """
        pred_begin = datetime.now()

        batch_size = len(requests)
        resnet_inputs = self.resnet_inputs[batch_size]
        inception_inputs = self.inception_inputs[batch_size]

        resnet_svm_future = self.task_execution_thread_pool.submit(
            lambda inputs : self.kernel_svm_model.predict(self.resnet_model.predict(inputs)), resnet_inputs)
        
        inception_log_reg_future = self.task_execution_thread_pool.submit(
            lambda inputs : self.log_reg_model.predict(self.inception_model.predict(inputs)), inception_inputs)

        resnet_svm_classes = resnet_svm_future.result()
        inception_log_reg_classes = inception_log_reg_future.result()

        pred_end = datetime.now()

        self.stats_thread_pool.submit(self._update_stats, requests, pred_begin, pred_end, batch_size)

    def _update_stats(self, requests, pred_begin_time, pred_end_time, batch_size):
        try:
            self.batch_predict_latencies.append((pred_end_time - pred_begin_time).total_seconds())
            self.batch_sizes.append(batch_size)
            for msg_id, send_time in requests:
                e2e_latency = (pred_end_time - send_time).total_seconds()
                self.latencies.append(e2e_latency)
                self.stats["per_message_lats"][msg_id] = e2e_latency

            self.trial_num_complete += batch_size

            if self.trial_num_complete >= self.trial_length:
                self.print_stats()
                self.init_stats()
                self.trial_num_complete = 0
        except Exception as e:
            print(e)
    

class DriverBenchmarker(object):
    def __init__(self, models_dict, trial_length):
        self.models_dict = models_dict
        self.trial_length = trial_length
        self.request_queue = Queue.Queue()

    def set_configs(self, configs):
        self.configs = configs

    def run(self, num_trials, batch_size, num_cpus, replica_num, slo_millis, process_file=None, request_delay=None):
        if process_file:
            self._benchmark_arrival_process(replica_num, num_trials, batch_size, slo_millis, process_file)
        elif request_delay:
            self._benchmark_over_under(replica_num, num_trials, batch_size, slo_millis, request_delay)
        else:
            self._benchmark_batches(replica_num, num_trials, batch_size, slo_millis)

    def _benchmark_batches(self, replica_num, num_trials, batch_size, slo_millis):
        logger.info("Generating random inputs")
        resnet_inputs, inception_inputs = generate_inputs()

        logger.info("Starting predictions")

        predictor = Predictor(self.models_dict, trial_length=self.trial_length)
        while True:
            send_time = datetime.now()
            send_times = [send_time for _ in range(batch_size)]
            
            batch_idx = np.random.randint(len(resnet_inputs) - batch_size)
            resnet_batch = resnet_inputs[batch_idx : batch_idx + batch_size]
            inception_batch = inception_inputs[batch_idx : batch_idx + batch_size]

            predictor.predict(send_times, resnet_batch, inception_batch)

            if len(predictor.stats["thrus"]) > num_trials:
                break

        save_results(self.configs, [predictor.stats], "single_proc_bs_{}_bench".format(batch_size), slo_millis, process_num=replica_num)

    def _benchmark_request_delay(self, replica_num, num_trials, batch_size, slo_millis, request_delay_seconds):
        logger.info("Initializing processor thread")
        processor_thread = Thread(target=self._run_async_query_processor, args=(replica_num, num_trials, batch_size, slo_millis, None))
        processor_thread.start()
        time.sleep(60)
        
        logger.info("Starting predictions")

        for i in range(10000):
            send_time = datetime.now()
            self.request_queue.put((msg_id, send_time))
            time.sleep(request_delay_seconds)
            # self._spin_sleep(request_delay_seconds)

        processor_thread.join()

    def _benchmark_arrival_process(self, replica_num, num_trials, batch_size, slo_millis, process_file):
        logger.info("Parsing arrival process")
        arrival_process = load_tagged_arrival_deltas(process_file)
        arrival_process = condense_delays(arrival_process, replica_num)
        with open("CONDENSED.deltas", "w+") as f:
            for delay, tag in arrival_process:
                line = "{},{}\n".format(delay, tag)
                f.write(line)

        logger.info("Initializing processor thread")
        processor_thread = Thread(target=self._run_async_query_processor, args=(replica_num, num_trials, batch_size, slo_millis, process_file))
        processor_thread.start()
        time.sleep(60)
        
        logger.info("Starting predictions")

        for idx in range(len(arrival_process)):
            request_delay_millis, request_replica_num = arrival_process[idx]
            request_delay_seconds = request_delay_millis * .001

            if request_replica_num == replica_num:
                send_time = datetime.now()
                msg_id = idx
                self.request_queue.put((msg_id, send_time))

            time.sleep(request_delay_seconds)
            # self._spin_sleep(request_delay_seconds)

        processor_thread.join()

    def _spin_sleep(self, request_delay_seconds):
        sleep_end_time = datetime.now() + timedelta(seconds=request_delay_seconds)
        while datetime.now() < sleep_end_time:
            continue

    def _run_async_query_processor(self, replica_num, num_trials, batch_size, slo_millis, process_file):
        predictor = Predictor(self.models_dict, trial_length=self.trial_length)
        loop_lats = []
        prev_stats_len = 0
        while True:
            curr_batch = []
            batch_item = self.request_queue.get(block=True)
            curr_batch.append(batch_item)
            while len(curr_batch) < batch_size and (not self.request_queue.empty()):
                try:
                    batch_item = self.request_queue.get_nowait()
                    curr_batch.append(batch_item)
                except Queue.Empty:
                    break

            predictor.predict(curr_batch)

            if len(predictor.stats["thrus"]) > num_trials:
                break

        save_results(self.configs, [predictor.stats], "single_proc_arrival_procs", slo_millis, process_num=replica_num, arrival_process=process_file)
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Single Process Image Driver 1')
    parser.add_argument('-b',  '--batch_sizes', type=int, nargs='+', help="The batch size configurations to benchmark for the driver. Each configuration will be benchmarked separately.")
    parser.add_argument('-c',  '--cpus', type=int, nargs='+', help="The set of VIRTUAL cpu cores on which to run the single process driver")
    parser.add_argument('-r',  '--resnet_gpu', type=int, default=0, help="The GPU on which to run the ResNet 152 featurization model")
    parser.add_argument('-i',  '--inception_gpu', type=int, default=1, help="The GPU on which to run the inception featurization model")
    parser.add_argument('-t',  '--num_trials', type=int, default=15, help="The number of trials to run")
    parser.add_argument('-tl', '--trial_length', type=int, default=200, help="The length of each trial, in requests")
    parser.add_argument('-n',  '--replica_num', type=int, help="The replica number corresponding to the driver")
    parser.add_argument('-p',  '--process_file', type=str, help="Path to a TAGGED arrival process file")
    parser.add_argument('-rd', '--request_delay', type=float, help="The request delay")
    parser.add_argument('-s', '--slo_millis', type=int, help="The SLO, in milliseconds")
    
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
        benchmarker.run(args.num_trials, batch_size, num_cpus, args.replica_num, args.slo_millis, args.process_file, args.request_delay)
