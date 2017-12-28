import sys
import os
import argparse
import numpy as np
import json
import logging
import math

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from single_proc_utils import HeavyNodeConfig, save_results
from models import nmt_model, tf_lstm_model, tf_lang_detect_model

LANG_CLASSIFICATION_ENGLISH = "en"
LANG_CLASSIFICATION_GERMAN = "de"

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(CURR_DIR, "models")

NMT_MODEL_NAME = "tf-nmt"
LSTM_MODEL_NAME = "tf-lstm"
LANG_DETECT_MODEL_NAME = "tf-lang-detect"

NMT_MODEL_PATH = os.path.join(MODELS_DIR, "nmt_model_data")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "tf_lstm_model_data")
LANG_DETECT_MODEL_PATH = os.path.join(MODELS_DIR, "tf_lang_detect_model_data")

WORKLOAD_RELATIVE_PATH = "workload"

LANG_CLASSIFICATION_ENGLISH = "en"
LANG_CLASSIFICATION_GERMAN = "de"

########## Setup ##########

def get_heavy_node_configs(batch_size, allocated_cpus, nmt_gpus=[]):
    nmt_config = HeavyNodeConfig(model_name=NMT_MODEL_NAME,
                                 input_type="bytes",
                                 allocated_cpus=allocated_cpus,
                                 gpus=nmt_gpus,
                                 batch_size=batch_size)

    lstm_config = HeavyNodeConfig(model_name=LSTM_MODEL_NAME,
                                  input_type="bytes",
                                  allocated_cpus=allocated_cpus,
                                  gpus=[],
                                  batch_size=batch_size)

    lang_detect_config = HeavyNodeConfig(model_name=LANG_DETECT_MODEL_NAME,
                                         input_type="bytes",
                                         allocated_cpus=allocated_cpus,
                                         gpus=[],
                                         batch_size=batch_size)

    return [nmt_config]

def create_nmt_model(model_path, gpu_num):
    return nmt_model.NMTModel(model_path, gpu_num)

def create_lstm_model(model_path):
    return tf_lstm_model.TfLstm(model_path)

def create_lang_detect_model(model_path):
    return tf_lang_detect_model.LangDetectModel(model_path)

def load_models(nmt_gpu):
    models_dict = {
        NMT_MODEL_NAME : create_nmt_model(NMT_MODEL_PATH, gpu_num=nmt_gpu),
        LANG_DETECT_MODEL_NAME : create_lang_detect_model(LANG_DETECT_MODEL_PATH),
        LSTM_MODEL_NAME : create_lstm_model(LSTM_MODEL_PATH)
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
        self.nmt_model = models_dict[NMT_MODEL_NAME]
        self.lstm_model = models_dict[LSTM_MODEL_NAME]
        self.lang_detect_model = models_dict[LANG_DETECT_MODEL_NAME]

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

    def predict(self, inputs):
        """
        Parameters
        ------------
        lstm_inputs : [str]
            A list of text items on which to perform sentiment analysis
        """

        begin_time = datetime.now()

        def update_stats(lats, num_completed):
            self.latencies += lats
            self.total_num_complete += num_completed
            self.trial_num_complete += num_completed

            if self.trial_num_complete >= self.trial_length:
                self.print_stats()
                self.init_stats()

        def lang_detect_fn(inputs):
            langs = self.lang_detect_model.predict(inputs)
            unk_count = 0
            english_inps = []
            german_inps = []
            for i in range(len(langs)):
                if lang == LANG_CLASSIFICATION_ENGLISH:
                    english_inps.append(inputs[i])
                elif lang == LANG_CLASSIFICATION_GERMAN:
                    german_inps.append(inputs[i])
                else:
                    unk_count += 1

            unk_end_time = datetime.now()
            latency = (end_time - begin_time).total_seconds()

            update_stats([latency for _ in range(unk_count)], unk_count)

            return english_inps, german_inps

        def lstm_fn(inputs):
            sentiments = self.lstm_model.predict(inputs)
            num_completed = len(sentiments)

            lstm_end_time = datetime.now()
            latency = (lstm_end_time - begin_time).total_seconds()

            update_stats([latency for _ in range(num_completed)], num_completed)

            return sentiments

        lang_detect_future = self.thread_pool.submit(lang_detect_fn, inputs)
        english_inps, german_inps = lang_detect_future.result()

        lstm_future = self.thread_pool.submit(lstm_fn, english_inps)

        nmt_future = self.thread_pool.submit(
            lambda inputs : lstm_fn(self.nmt_model.predict(inputs)), german_inps)

        lstm_future.result()
        nmt_future.result()

class DriverBenchmarker(object):
    def __init__(self, models_dict, trial_length, process_num):
        self.models_dict = models_dict
        self.trial_length = trial_length
        self.process_num = process_num
        self.loaded_text = False

    def set_configs(self, configs):
        self.configs = configs

    def run(self, num_trials, batch_size, input_length):
        predictor = Predictor(self.models_dict, trial_length=self.trial_length)

        logger.info("Generating random inputs")

        inputs = self._gen_inputs(num_inputs=1000, input_length=input_length)
        inputs = [i for _ in range(40) for i in inputs]
        
        logger.info("Starting predictions")
        while True:
            batch_idx = np.random.randint(len(inputs) - batch_size)
            inputs_batch = inputs[batch_idx : batch_idx + batch_size]

            predictor.predict(inputs_batch)

            if len(predictor.stats["thrus"]) > num_trials:
                break

        save_results(self.configs, [predictor.stats], "nmt_single_proc_exps", self.process_num)

    def _gen_inputs(self, num_inputs=1000, input_length=20):
        if not self.loaded_text:
            self.text = self._load_text()
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

        bytes_inputs = [np.frombuffer(bytearray(input_item), dtype=np.uint8) for input_item in inputs]

        return bytes_inputs

    def _load_text(self):
        workload_data_path = os.path.join(CURR_DIR, WORKLOAD_RELATIVE_PATH, "workload.txt")
        workload_data_file = open(workload_data_path, "rb")
        workload_text = workload_data_file.readlines()
        np.random.shuffle(workload_text)
        return workload_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Single Process Image Driver 1')
    parser.add_argument('-d',  '--duration', type=int, default=120, help='The maximum duration of the benchmarking process in seconds, per iteration')
    parser.add_argument('-b',  '--batch_sizes', type=int, nargs='+', help="The batch size configurations to benchmark for the driver. Each configuration will be benchmarked separately.")
    parser.add_argument('-c',  '--cpus', type=int, nargs='+', help="The set of cpu cores on which to run the single process driver")
    parser.add_argument('-ng',  '--nmt_gpu', type=int, default=1, help="The GPU on which to run the NMT model")
    parser.add_argument('-t',  '--num_trials', type=int, default=15, help="The number of trials to run")
    parser.add_argument('-tl', '--trial_length', type=int, default=200, help="The length of each trial, in requests")
    parser.add_argument('-p',  '--process_number', type=int, default=0)
    parser.add_argument('-l', '--input_lengths', type=int, nargs='+')
    
    args = parser.parse_args()

    if not args.cpus:
        raise Exception("The set of allocated cpus must be specified via the '--cpus' flag!")

    default_batch_size_confs = [2]
    batch_size_confs = args.batch_sizes if args.batch_sizes else default_batch_size_confs

    default_input_length_confs = [20]
    input_length_confs = args.input_lengths if args.input_lengths else default_input_length_confs
    
    models_dict = load_models(args.nmt_gpu)
    benchmarker = DriverBenchmarker(models_dict, args.trial_length, args.process_number)

    for input_length in input_length_confs:
        for batch_size in batch_size_confs:
            configs = get_heavy_node_configs(batch_size=batch_size,
                                             allocated_cpus=args.cpus,
                                             nmt_gpus=[args.nmt_gpu])
            benchmarker.set_configs(configs)
            benchmarker.run(args.num_trials, batch_size, input_length)