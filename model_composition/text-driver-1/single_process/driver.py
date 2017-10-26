import sys
import os
import argparse
import numpy as np
import json
import logging

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from single_proc_utils import HeavyNodeConfig
from models import theano_sentiment_model as lstm_model
from models import nmt_model

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(CURR_DIR, "models")

LSTM_1_MODEL_NAME = "lstm_1"
LSTM_2_MODEL_NAME = "lstm_2"
NMT_MODEL_NAME = "nmt"

LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model_data")
NMT_MODEL_PATH = os.path.join(MODELS_DIR, "nmt_model_data")

########## Setup ##########

def get_heavy_node_configs(batch_size, allocated_cpus, nmt_gpus=[]):
    lstm_1_config = HeavyNodeConfig(model_name=LSTM_1_MODEL_NAME,
                                    input_type="strings",
                                    allocated_cpus=allocated_cpus,
                                    gpus=[],
                                    batch_size=batch_size)

    lstm_2_config = HeavyNodeConfig(model_name=LSTM_2_MODEL_NAME,
                                    input_type="strings",
                                    allocated_cpus=allocated_cpus,
                                    gpus=[],
                                    batch_size=batch_size)

    nmt_config = HeavyNodeConfig(model_name=NMT_MODEL_NAME,
                                 input_type="strings",
                                 allocated_cpus=allocated_cpus,
                                 gpus=nmt_gpus,
                                 batch_size=batch_size)

    return [lstm_1_config, lstm_2_config, nmt_config]

def load_models(nmt_gpu):
    models_dict = {
        LSTM_1_MODEL_NAME : create_lstm_model(LSTM_MODEL_PATH),
        LSTM_2_MODEL_NAME : create_lstm_model(LSTM_MODEL_PATH),
        NMT_MODEL_NAME : create_nmt_model(NMT_MODEL_PATH, gpu_num=nmt_gpu)
    }

    return models_dict

def create_nmt_model(model_data_path, gpu_num):
    return nmt_model.NMTModel(model_data_path, gpu_num=gpu_num)

def create_lstm_model(model_data_path):
    return lstm_model.MovieSentimentLstm(model_data_path)

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
        self.lstm_model_1 = models_dict[LSTM_1_MODEL_NAME]
        self.lstm_model_2 = models_dict[LSTM_2_MODEL_NAME]
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
            A list of English movie reviews, each represented as a string
        nmt_inputs : [str]
            A list of German text items, each represented as a string
        """
        assert len(lstm_inputs) == len(nmt_inputs)

        batch_size = len(lstm_inputs)

        begin_time = datetime.now()

        lstm_future = self.thread_pool.submit(self.lstm_model_1.predict, lstm_inputs)

        nmt_lstm_future = self.thread_pool.submit(
            lambda inputs : self.lstm_model_2.predict(self.nmt_model.predict(inputs)), nmt_inputs)

        lstm_classes = lstm_future.result()
        nmt_lstm_classes = nmt_lstm_future.result()

        end_time = datetime.now()

        latency = (end_time - begin_time).total_seconds()
        self.latencies.append(latency)
        self.total_num_complete += batch_size
        self.trial_num_complete += batch_size
        if self.trial_num_complete % self.trial_length == 0:
            self.print_stats()
            self.init_stats()

class DriverBenchmarker(object):
    def __init__(self, models_dict, trial_length):
        self.predictor = Predictor(models_dict, trial_length)
        self.loaded_reviews = False
        self.loaded_german = False

    def set_configs(self, configs):
        self.configs = configs

    def run(self, num_trials, batch_size, input_length):
        logger.info("Generating random inputs")
        lstm_inputs = self._gen_reviews_inputs(num_inputs=1000, input_length=input_length)
        lstm_inputs = [i for _ in range(40) for i in lstm_inputs]

        nmt_inputs = self._gen_german_inputs(num_inputs=1000, input_length=input_length)
        nmt_inputs = [i for _ in range(40) for i in nmt_inputs]

        assert len(nmt_inputs) == len(lstm_inputs)
        
        logger.info("Starting predictions")
        while True:
            batch_idx = np.random.randint(len(vgg_inputs) - batch_size)
            lstm_batch = lstm_inputs[batch_idx : batch_idx + batch_size]
            nmt_batch = nmt_inputs[batch_idx : batch_idx + batch_size]

            self.predictor.predict(lstm_batch, nmt_batch)

            if len(self.predictor.stats["thrus"]) > num_trials:
                break

        driver_utils.save_results(self.configs, [self.predictor.stats], "single_proc_gpu_and_batch_size_experiments")

    def _gen_reviews_inputs(self, num_inputs, input_length):
        if not self.loaded_reviews:
            self.reviews = self._load_reviews()
            self.loaded_reviews = True

        reviews_len = len(self.reviews)
        inputs = []
        for _ in range(num_inputs):
            review_idx = np.random.randint(reviews_len)
            review = self.reviews[review_idx]
            # Keep the first 200 words of the review,
            # or extend the review to exactly 200 words
            if len(review) < input_length:
                expansion_factor = int(math.ceil(float(input_length)/len(review)))
                for i in range(expansion_factor):
                    review = review + " " + review
            review = review[:input_length]
            inputs.append(review)
        return inputs

    def _gen_german_inputs(self, num_inputs, input_length):
        if not self.loaded_german:
            self.german_text = self._load_german()
            self.loaded_german = True

        inputs = []
        num_gen_inputs = 0
        while num_gen_inputs < num_inputs:
            idx = np.random.randint(len(self.german_text))
            text = self.german_text[idx]
            words = text.split()
            if len(words) > input_length:
                words = words[:input_length]
                inputs.append(" ".join(words))
                num_gen_inputs += 1

        return inputs

    def _load_german(self):
        german_data_path = os.path.join(CURR_DIR, "nmt_workload", "german_text.de")
        german_data_file = open(german_data_path, "rb")
        german_text = german_data_file.readlines()
        np.random.shuffle(german_text)
        return german_text

    def _load_reviews(self):
        base_path = os.path.join(CURR_DIR, "workload_data/aclImdb/test/")
        reviews = []
        pos_path = os.path.join(base_path, "pos")
        for rev_file in os.listdir(pos_path):
            with open(os.path.join(pos_path, rev_file), "r") as f:
                reviews.append(f.read().strip())

        neg_path = os.path.join(base_path, "neg")
        for rev_file in os.listdir(neg_path):
            with open(os.path.join(neg_path, rev_file), "r") as f:
                reviews.append(f.read().strip())
        # Shuffle in place
        np.random.shuffle(reviews)
        return reviews

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Single Process Text Driver 1')
    parser.add_argument('-d', '--duration', type=int, default=120, help='The maximum duration of the benchmarking process in seconds, per iteration')
    parser.add_argument('-b', '--batch_sizes', type=int, nargs='+', help="The batch size configurations to benchmark for the driver. Each configuration will be benchmarked separately.")
    parser.add_argument('-c', '--cpus', type=int, nargs='+', help="The set of cpu cores on which to run the single process driver")
    parser.add_argument('-g', '--nmt_gpu', type=int, default=0, help="The GPU on which to run the NMT model")
    parser.add_argument('-t', '--num_trials', type=int, default=15, help="The number of trials to run")
    parser.add_argument('-tl', '--trial_length', type=int, default=200, help="The length of each trial, in requests")
    parser.add_argument('-l', '--input_lengths', type=int, nargs='+', help="Input length configurations to benchmark")
    
    args = parser.parse_args()

    if not args.cpus:
        raise Exception("The set of allocated cpus must be specified via the '--cpus' flag!")

    default_batch_size_confs = [2]
    default_input_length_confs = [100]

    batch_size_confs = args.batch_sizes if args.batch_sizes else default_batch_size_confs
    input_length_confs = args.input_lengths if args.input_lengths else default_input_length_confs
    
    models_dict = load_models(args.nmt_gpu)
    benchmarker = DriverBenchmarker(models_dict, args.trial_length)

    for input_length in input_length_confs:
        for batch_size in batch_size_confs:
            configs = get_heavy_node_configs(batch_size=batch_size,
                                             allocated_cpus=args.cpus,
                                             nmt_gpus=[args.nmt_gpu])
            benchmarker.set_configs(configs)
            benchmarker.run(args.num_trials, batch_size, input_length)
