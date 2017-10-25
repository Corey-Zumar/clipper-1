from __future__ import print_function
import json
import logging
import os
import datetime
import requests
from scipy.stats import linregress
import numpy as np

logger = logging.getLogger(__name__)

INCREASING = "increasing"
CONVERGED_HIGH = "converged_high"
DECREASING = "decreasing"
CONVERGED = "converged"
UNKNOWN = "unknown"


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
                 input_size=-1,
                 no_diverge=False):
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
        self.no_diverge = no_diverge

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


def check_convergence(stats, configs, latency_upper_bound=None):
    """
    Returns
    -------
    boolean : whether we're in a steady state yet
    slope_sign : if we're not in a steady state, the sign of
        the slope of the function computing latency
        as a function of time. If the slope is negative, we're
        draining the queues and we should keep waiting. If the slope is
        positive, we're filling the queues faster than they can be drained
        and we should quit and increase the delay.

    """
    window_size = min(15, len(stats["p99_lats"]) - 1)
    p99_lats = stats["p99_lats"][-1*window_size:]

    mean_batch_sizes = {}
    for c in configs:
        mean_batch_sizes[c.name] = np.mean([b[c.name] for b in stats["mean_batch_sizes"][-1*window_size:]])
    lr = linregress(x=range(len(p99_lats)), y=p99_lats)
    logger.info(lr)
    # pvalue checks against the null hypothesis that the
    # slope is 0. We are checking for the slope to be 0,
    # so we want to the null hypothesis to be true.

    # If pvalue less than 0.001, the line definitely has a slope
    if lr.pvalue < 0.001:
        if lr.slope > 0:
            return INCREASING
        else:
            return DECREASING
    elif lr.pvalue > 0.2:
        # Slope is 0, now check to see
        # if mean batch_sizes are less
        # than
        # configured batch size.

        if latency_upper_bound is not None:
            mean_p99_lats = np.mean(p99_lats)
            if mean_p99_lats > latency_upper_bound:
                logger.info("Slope is 0 but p99 latency ({lat} s) is too high for node {name}.".format(
                    lat=mean_p99_lats, name=c.name))
                return CONVERGED_HIGH

        # If any of the nodes batch sizes are set to 1, skip the batch size check
        for c in configs:
            if c.batch_size == 1.0:
                return CONVERGED

        # We don't know which node is the bottleneck, so we check that all the nodes have batch
        # sizes slightly less than the configured batch size.
        for c in configs:
            if mean_batch_sizes[c.name] == c.batch_size:
                logger.info("Slope is 0 but batch_sizes too big for node {name}.".format(name=c.name))
                return CONVERGED_HIGH
        return CONVERGED
    else:
        return UNKNOWN

    # if lr.pvalue < 0.1:
    #     return (False, np.sign(lr.slope))
    # else:
    #     # Slope is 0, now check to see if mean batch_sizes are less than
    #     # configured batch size.
    #
    #     # If any of the nodes batch sizes are set to 1 and we have a slope of 0, we've converged.
    #     for c in configs:
    #         if c.batch_size == 1.0:
    #             return (True, 0.0)
    #
    #     # We don't know which node is the bottleneck, so we check that all the nodes have batch
    #     # sizes slightly less than the configured batch size.
    #     for c in configs:
    #         if mean_batch_sizes[c.name] == c.batch_size:
    #             logger.info("Slope is 0 but batch_sizes too big for node {name}.\n{config}".format(
    #                 name=c.name,
    #                 config=json.dumps(c.__dict__, sort_keys=True, indent=2)
    #             ))
    #             return (False, np.sign(1.))
    #     return (True, 0.0)
