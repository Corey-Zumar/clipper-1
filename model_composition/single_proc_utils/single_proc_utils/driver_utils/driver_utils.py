import os
import json
import logging
import datetime

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

class HeavyNodeConfig(object):
    def __init__(self,
                 model_name,
                 input_type,
                 allocated_cpus,
                 gpus=[],
                 batch_size=1):
        self.model_name = model_name
        self.input_type = input_type
        self.allocated_cpus = allocated_cpus
        self.gpus = gpus
        self.batch_size = batch_size

    def to_json(self):
        return json.dumps(self.__dict__)

def save_results(configs, client_metrics, results_dir, process_num):
    """
    Parameters
    ----------
    configs : list(HeavyNodeConfig)
        The configs for any models deployed


    """

    results_dir = os.path.abspath(os.path.expanduser(results_dir))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logger.info("Created experiments directory: %s" % results_dir)

    results_obj = {
        "node_configs": [c.__dict__ for c in configs],
        "client_metrics": client_metrics,
    }
    results_file = os.path.join(results_dir,
                                "results-{:%y%m%d_%H%M%S}-{}.json".format(datetime.datetime.now(), process_num))
    with open(results_file, "w") as f:
        json.dump(results_obj, f, indent=4)
        logger.info("Saved results to {}".format(results_file))