import sys
import os
import argparse
import numpy as np
import json
import logging

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from single_proc_utils import HeavyNodeConfig
from models import gensim_lda_model as lda_model
from models import gensim_similarity_model as docsim_model

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(CURR_DIR, "models")

LDA_MODEL_NAME = "lda"
DOCSIM_MODEL_NAME = "docsim"

LDA_MODEL_PATH = os.path.join(MODELS_DIR, "lda_model_data")
DOCSIM_MODEL_PATH = os.path.join(MODELS_DIR, "docsim_model_data")

########## Setup ##########

def get_heavy_node_configs(batch_size, allocated_cpus):
    lda_config = HeavyNodeConfig(model_name=LDA_MODEL_NAME,
                                 input_type="strings",
                                 allocated_cpus=allocated_cpus,
                                 gpus=[],
                                 batch_size=batch_size)

    docsim_config = HeavyNodeConfig(model_name=DOCSIM_MODEL_NAME,
                                    input_type="strings",
                                    allocated_cpus=allocated_cpus,
                                    gpus=[],
                                    batch_size=batch_size)

    return [lda_config, docsim_config]

def load_models():
    models_dict = {
        LDA_MODEL_NAME : create_lda_model(LDA_MODEL_PATH),
        DOCSIM_MODEL_NAME : create_docsim_model(DOCSIM_MODEL_PATH)
    }

    return models_dict

def create_lda_model(model_data_path):
    return lda_model.LDAModel(model_data_path)

def create_docsim_model(model_data_path):
    return docsim_model.SimilarityModel(model_data_path)

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
        self.lda_model = models_dict[LDA_MODEL_NAME]
        self.docsim_model = models_dict[DOCSIM_MODEL_NAME]

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

    def predict(self, lda_docsim_inputs):
        """
        Parameters
        ------------
        lda_docsim_inputs : [str]
            A list of English documents, each represented as a string
        """

        batch_size = len(lda_docsim_inputs)

        begin_time = datetime.now()

        lda_future = self.thread_pool.submit(self.lda_model.predict, lda_docsim_inputs)
        docsim_future = self.thread_pool.submit(self.docsim_model.predict, lda_docsim_inputs)

        lda_future.result()
        docsim_future.result()

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
        self.loaded_docs = False

    def set_configs(self, configs):
        self.configs = configs

    def run(self, num_trials, batch_size, input_length):
        logger.info("Generating random inputs")

        lda_docsim_inputs = self._gen_docs_inputs(num_inputs=1000, input_length=input_length)
        lda_docsim_inputs = [i for _ in range(40) for i in lda_docsim_inputs]
        
        logger.info("Starting predictions")
        while True:
            batch_idx = np.random.randint(len(lda_docsim_inputs) - batch_size)
            lda_docsim_batch = lda_docsim_inputs[batch_idx : batch_idx : batch_size]

            self.predictor.predict(lda_docsim_batch)

            if len(self.predictor.stats["thrus"]) > num_trials:
                break

        driver_utils.save_results(self.configs, [self.predictor.stats], "single_proc_gpu_and_batch_size_experiments")

    def _gen_docs_inputs(self, num_inputs, input_length):
        if not self.loaded_docs:
            self.doc_text = self._load_doc_text()
            self.loaded_docs = True

        inputs = []
        num_gen_inputs = 0
        while num_gen_inputs < num_inputs:
            idx = np.random.randint(len(self.doc_text))
            text = self.doc_text[idx]
            words = text.split()
            if len(words) < input_length:
                expansion_factor = int(math.ceil(float(input_length)/len(text)))
                for i in range(expansion_factor):
                    words = words + words
            words = words[:input_length]
            inputs.append(" ".join(words))
            num_gen_inputs += 1

        return inputs

    def _load_doc_text(self):
        doc_data_path = os.path.join(CURR_DIR, "gensim_workload", "docs.txt")
        doc_data_file = open(doc_data_path, "rb")
        doc_text = doc_data_file.readlines()
        np.random.shuffle(doc_text)
        return doc_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up and benchmark models for Single Process Text Driver 1')
    parser.add_argument('-d',  '--duration', type=int, default=120, help='The maximum duration of the benchmarking process in seconds, per iteration')
    parser.add_argument('-b',  '--batch_sizes', type=int, nargs='+', help="The batch size configurations to benchmark for the driver. Each configuration will be benchmarked separately.")
    parser.add_argument('-c',  '--cpus', type=int, nargs='+', help="The set of cpu cores on which to run the single process driver")
    parser.add_argument('-t',  '--num_trials', type=int, default=15, help="The number of trials to run")
    parser.add_argument('-tl', '--trial_length', type=int, default=200, help="The length of each trial, in requests")
    parser.add_argument('-l',  '--input_lengths', type=int, nargs='+', help="Input length configurations to benchmark")
    
    args = parser.parse_args()

    if not args.cpus:
        raise Exception("The set of allocated cpus must be specified via the '--cpus' flag!")

    default_batch_size_confs = [2]
    default_input_length_confs = [20]

    batch_size_confs = args.batch_sizes if args.batch_sizes else default_batch_size_confs
    input_length_confs = args.input_lengths if args.input_lengths else default_input_length_confs
    
    models_dict = load_models()
    benchmarker = DriverBenchmarker(models_dict, args.trial_length)

    for input_length in input_length_confs:
        for batch_size in batch_size_confs:
            configs = get_heavy_node_configs(batch_size=batch_size,
                                             allocated_cpus=args.cpus)
            benchmarker.set_configs(configs)
            benchmarker.run(args.num_trials, batch_size, input_length)
