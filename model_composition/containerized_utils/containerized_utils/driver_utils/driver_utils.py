from __future__ import print_function
import json
import logging
import os
import datetime
import requests

logger = logging.getLogger(__name__)


class HeavyNodeConfig(object):
    def __init__(self,
                 name,
                 input_type,
                 model_image,
                 allocated_cpus,
                 cpus_per_replica,
                 num_replicas,
                 gpus,
                 batch_size,
                 use_nvidia_docker,
                 slo=5000000,
                 input_size=-1):
        self.name = name
        self.input_type = input_type
        self.model_image = model_image
        self.allocated_cpus = allocated_cpus
        self.cpus_per_replica = cpus_per_replica
        self.slo = slo
        self.num_replicas = num_replicas
        self.gpus = gpus
        self.batch_size = batch_size
        self.use_nvidia_docker = use_nvidia_docker
        self.input_size = input_size
        self.instance_type = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-type").text
        if len(gpus) == 0:
            self.gpus_per_replica = 0
        else:
            self.gpus_per_replica = 1

    def to_json(self):
        return json.dumps(self.__dict__)


def setup_heavy_node(clipper_conn, config, default_output="TIMEOUT"):
    clipper_conn.register_application(name=config.name,
                                      default_output=default_output,
                                      slo_micros=config.slo,
                                      input_type=config.input_type)

    clipper_conn.deploy_model(name=config.name,
                              version=1,
                              image=config.model_image,
                              input_type=config.input_type,
                              num_replicas=config.num_replicas,
                              batch_size=config.batch_size,
                              gpus=config.gpus,
                              allocated_cpus=config.allocated_cpus,
                              cpus_per_replica=config.cpus_per_replica,
                              use_nvidia_docker=config.use_nvidia_docker)

    clipper_conn.link_model_to_app(app_name=config.name, model_name=config.name)


def save_results(configs, clipper_conn, client_metrics, results_dir, prefix="results"):
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

    if "all_lats" not in client_metrics[0]:
        raise Exception("No latencies list found under key \"all_lats\"."
                        " Please update your driver to include all latencies so we can"
                        " plot the latency CDF")
    else:
        for c in client_metrics:
            all_lats_strs = [json.dumps(l) for l in c["all_lats"]]
            c["all_lats"] = all_lats_strs

    results_obj = {
        "node_configs": [c.__dict__ for c in configs],
        # "clipper_metrics": clipper_conn.inspect_instance(),
        "client_metrics": client_metrics,
    }
    results_file = os.path.join(results_dir, "{prefix}-{ts:%y%m%d_%H%M%S}.json".format(
        prefix=prefix, ts=datetime.datetime.now()))
    with open(results_file, "w") as f:
        json.dump(results_obj, f, indent=4)
        logger.info("Saved results to {}".format(results_file))
