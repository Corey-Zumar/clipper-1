import sys
import os
import argparse
import numpy as np
import json
import logging

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from single_proc_utils import HeavyNodeConfig, save_results
from models import tf_lstm_model, nmt_model

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(CURR_DIR, "models")

LANG_DETECT_MODEL_NAME = "tf-lang-detect"
NMT_MODEL_NAME = "tf-nmt"
TF_LSTM_MODEL_NAME = "tf-lstm"

NMT_MODEL_PATH = os.path.join(MODELS_DIR, "nmt_model_data")
TF_LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "tf_lstm_model_data")

########## Setup ##########

def get_heavy_node_configs(batch_size, allocated_cpus, lstm_gpus=[], nmt_gpus=[]):
    # lstm_config = HeavyNodeConfig(model_name=TF_LSTM_MODEL_NAME,
    #                               input_type="strings",
    #                               allocated_cpus=allocated_cpus,
    #                               gpus=[0],
    #                               batch_size=batch_size)

    nmt_config = HeavyNodeConfig(model_name=NMT_MODEL_NAME,
                                 input_type="bytes",
                                 allocated_cpus=allocated_cpus,
                                 gpus=[0],
                                 batch_size=batch_size)

    return [nmt_config]

def create_lstm_model(model_path, gpu_num):
    return tf_lstm_model.TfLstm(model_path, gpu_num)

def create_nmt_model(model_path, gpu_num):
    return nmt_model.NMTModel(model_path, gpu_num)

def load_models(lstm_gpu, nmt_gpu):
    models_dict = {
        # TF_LSTM_MODEL_NAME : create_lstm_model(TF_LSTM_MODEL_PATH, gpu_num=lstm_gpu),
        NMT_MODEL_NAME : create_nmt_model(NMT_MODEL_PATH, gpu_num=nmt_gpu)
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
        #self.tf_lstm_model = models_dict[TF_LSTM_MODEL_NAME]
        self.nmt_model = models_dict[NMT_MODEL_NAME]

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

    def predict(self, lstm_inputs, nmt_inputs):
        """
        Parameters
        ------------
        lstm_inputs : [str]
            A list of text items on which to perform sentiment analysis
        """

        assert len(lstm_inputs) == len(nmt_inputs)

        batch_size = len(lstm_inputs)

        begin_time = datetime.now()

        # lstm_future = self.thread_pool.submit(self.tf_lstm_model.predict, lstm_inputs)

        # lstm_classes = lstm_future.result()

        nmt_future = self.thread_pool.submit(self.nmt_model.predict, nmt_inputs)

        nmt_preds = nmt_future.result()

        end_time = datetime.now()

        latency = (end_time - begin_time).total_seconds()
        self.latencies.append(latency)
        self.total_num_complete += batch_size
        self.trial_num_complete += batch_size
        if self.trial_num_complete % self.trial_length == 0:
            self.print_stats()
            self.init_stats()

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
        lstm_inputs = self._gen_inputs(TF_LSTM_MODEL_NAME, num_inputs=1000, input_length=input_length)
        lstm_inputs = [i for _ in range(40) for i in lstm_inputs]

        nmt_inputs = self._gen_inputs(NMT_MODEL_NAME, num_inputs=1000, input_length=input_length)
        nmt_inputs = [i for _ in range(40) for i in nmt_inputs]

        # TODO(czumar): Change this when there are more models / inputs
        assert len(lstm_inputs) == len(lstm_inputs)
        
        logger.info("Starting predictions")
        while True:
            batch_idx = np.random.randint(len(lstm_inputs) - batch_size)
            lstm_batch = lstm_inputs[batch_idx : batch_idx + batch_size]
            nmt_batch = nmt_inputs[bathc_idx : batch_idx + batch_size]

            predictor.predict(lstm_batch, nmt_batch)

            if len(predictor.stats["thrus"]) > num_trials:
                break

        save_results(self.configs, [predictor.stats], "nmt_single_proc_exps", self.process_num)

    def _gen_inputs(self, model_name, num_inputs=1000, input_length=20):
        if not self.loaded_text:
            self.text = self._get_load_text_fn(model_name)()
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

    def _get_load_text_fn(self, model_name):
        if model_name == NMT_MODEL_NAME:
            return self._load_nmt_text

        elif model_name == LANG_DETECT_MODEL_NAME:
            return self._load_detect_text

        elif model_name == TF_LSTM_MODEL_NAME:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Single Process Image Driver 1')
    parser.add_argument('-d',  '--duration', type=int, default=120, help='The maximum duration of the benchmarking process in seconds, per iteration')
    parser.add_argument('-b',  '--batch_sizes', type=int, nargs='+', help="The batch size configurations to benchmark for the driver. Each configuration will be benchmarked separately.")
    parser.add_argument('-c',  '--cpus', type=int, nargs='+', help="The set of cpu cores on which to run the single process driver")
    parser.add_argument('-lg',  '--lstm_gpu', type=int, default=0, help="The GPU on which to run the Tensorflow LSTM")
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
    
    models_dict = load_models(args.lstm_gpu, args.nmt_gpu)
    benchmarker = DriverBenchmarker(models_dict, args.trial_length, args.process_number)

    for input_length in input_length_confs:
        for batch_size in batch_size_confs:
            configs = get_heavy_node_configs(batch_size=batch_size,
                                             allocated_cpus=args.cpus,
                                             lstm_gpus=[args.lstm_gpu],
                                             nmt_gpus=[args.nmt_gpu])
            benchmarker.set_configs(configs)
            benchmarker.run(args.num_trials, batch_size, input_length)