import subprocess32 as subprocess
import os
import sys
# import numpy as np
import time
import logging
import json
from clipper_admin import ClipperConnection, DockerContainerManager
from datetime import datetime
from containerized_utils import driver_utils

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))

DEFAULT_OUTPUT = "TIMEOUT"
CLIPPER_ADDRESS = "localhost"

RES50 = "res50"
RES152 = "res152"
ALEXNET = "alexnet"
INCEPTION_FEATS = "inception"
TF_KERNEL_SVM = "tf-kernel-svm"
TF_LOG_REG = "tf-log-reg"
TF_RESNET = "tf-resnet-feats"
TF_RESNET_VAR = "tf-resnet-feats-var"
TF_RESNET_SLEEP = "tf-resnet-feats-sleep"

TF_LANG_DETECT = "tf-lang-detect"
TF_NMT = "tf-nmt"
TF_LSTM = "tf-lstm"

REMOTE_ADDR = "localhost"


def get_heavy_node_config(model_name,
                          batch_size,
                          num_replicas,
                          cpus_per_replica,
                          allocated_cpus,
                          allocated_gpus,
                          remote_addr=None,
                          input_size=None):

    if model_name == ALEXNET:
        image = "gcr.io/clipper-model-comp/pytorch-alexnet:bench"
        return driver_utils.HeavyNodeConfig(name="alexnet",
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            remote_addr=remote_addr)

    elif model_name == RES50:
        image = "gcr.io/clipper-model-comp/pytorch-res50:bench"
        return driver_utils.HeavyNodeConfig(name="res50",
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            remote_addr=remote_addr
                                            )

    elif model_name == RES152:
        image = "gcr.io/clipper-model-comp/pytorch-res152:bench"
        return driver_utils.HeavyNodeConfig(name="res152",
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            remote_addr=remote_addr
                                            )

    elif model_name == INCEPTION_FEATS:
        image = "gcr.io/clipper-model-comp/inception-feats:bench"
        return driver_utils.HeavyNodeConfig(name=INCEPTION_FEATS,
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            remote_addr=remote_addr
                                            )

    elif model_name == TF_RESNET:
        image = "gcr.io/clipper-model-comp/tf-resnet-feats:bench"
        return driver_utils.HeavyNodeConfig(name=TF_RESNET,
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            remote_addr=remote_addr)

    elif model_name == TF_RESNET_VAR:
        image = "gcr.io/clipper-model-comp/tf-resnet-feats-variable-input:bench"
        assert input_size is not None
        return driver_utils.HeavyNodeConfig(name=TF_RESNET_VAR,
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            input_size=input_size,
                                            remote_addr=remote_addr)

    elif model_name == TF_RESNET_SLEEP:
        image = "gcr.io/clipper-model-comp/tf-resnet-feats-sleep:bench"
        assert input_size is not None
        return driver_utils.HeavyNodeConfig(name=TF_RESNET_SLEEP,
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            input_size=input_size,
                                            remote_addr=remote_addr)

    elif model_name == TF_LOG_REG:
        image = "gcr.io/clipper-model-comp/tf-log-reg:bench"
        return driver_utils.HeavyNodeConfig(name=TF_LOG_REG,
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=[],
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            remote_addr=remote_addr)

    elif model_name == TF_KERNEL_SVM:
        image = "gcr.io/clipper-model-comp/tf-kernel-svm:bench"
        return driver_utils.HeavyNodeConfig(name=TF_KERNEL_SVM,
                                            input_type="floats",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=[],
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            remote_addr=remote_addr)

    elif model_name == TF_LANG_DETECT:
        image = "gcr.io/clipper-model-comp/tf-lang-detect:bench"
        return driver_utils.HeavyNodeConfig(name=TF_LANG_DETECT,
                                            input_type="bytes",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=[],
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            remote_addr=remote_addr)

    elif model_name == TF_LSTM:
        image = "gcr.io/clipper-model-comp/tf-lstm:bench"
        return driver_utils.HeavyNodeConfig(name=TF_LSTM,
                                            input_type="bytes",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            remote_addr=remote_addr)

    elif model_name == TF_NMT:
        image = "gcr.io/clipper-model-comp/tf-nmt:bench"
        return driver_utils.HeavyNodeConfig(name=TF_NMT,
                                            input_type="bytes",
                                            model_image=image,
                                            allocated_cpus=allocated_cpus,
                                            cpus_per_replica=cpus_per_replica,
                                            gpus=allocated_gpus,
                                            batch_size=batch_size,
                                            num_replicas=num_replicas,
                                            use_nvidia_docker=True,
                                            no_diverge=True,
                                            remote_addr=remote_addr)


def get_input_size(config):
    if config.name in [TF_LOG_REG, TF_KERNEL_SVM]:
        return 2048
    elif config.name in [ALEXNET, RES50, RES152, INCEPTION_FEATS]:
        return 299*299*3
    elif config.name in [TF_RESNET, ]:
        return 224*224*3
    elif config.name in [TF_RESNET_VAR, TF_RESNET_SLEEP]:
        return config.input_size
    elif config.name in [TF_NMT, TF_LSTM, TF_LANG_DETECT]:
        return 20


def setup_clipper(configs):
    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.connect()
    # cl.stop_all(remote_addrs=[REMOTE_ADDR])
    cl.stop_all()
    cl.start_clipper(
        query_frontend_image="clipper/zmq_frontend:develop",
        redis_cpu_str="0",
        mgmt_cpu_str="0",
        query_cpu_str="0,16,1,17,2,18,3,19")
    time.sleep(10)
    for c in configs:
        driver_utils.setup_heavy_node(cl, c, DEFAULT_OUTPUT)
    time.sleep(10)
    logger.info("Clipper is set up!")
    return CLIPPER_ADDRESS


def get_clipper_batch_sizes(metrics_json):
    hists = metrics_json["histograms"]
    mean_batch_sizes = {}
    for h in hists:
        if "batch_size" in h.keys()[0]:
            name = h.keys()[0]
            model = name.split(":")[1]
            mean = float(h[name]["mean"])
            mean_batch_sizes[model] = round(float(mean), 2)
    return mean_batch_sizes


def get_clipper_queue_sizes(metrics_json):
    hists = metrics_json["histograms"]
    mean_queue_sizes = {}
    for h in hists:
        if "queue_size" in h.keys()[0]:
            name = h.keys()[0]
            model = name.split(":")[0]
            mean = float(h[name]["mean"])
            mean_queue_sizes[model] = round(float(mean), 2)
    return mean_queue_sizes


def get_clipper_thruputs(metrics_json):
    meters = metrics_json["meters"]
    thrus = {}
    for m in meters:
        if "prediction_throughput" in m.keys()[0]:
            name = m.keys()[0]
            rate = float(m[name]["rate"])
            thrus[name] = round(float(rate), 5)
    return thrus


def get_clipper_counts(metrics_json):
    counters = metrics_json["counters"]
    counts = {}
    for c in counters:
        if "internal" not in c.keys()[0]:
            name = c.keys()[0]
            count = c[name]["count"]
            counts[name] = int(count)
    return counts


def get_profiler_stats(metrics_json):
    mean_latencies = {}
    p99_latencies = {}
    thrus = {}
    counts = {}

    hists = metrics_json["histograms"]
    for h in hists:
        if "prediction_latency" in h.keys()[0]:
            name = h.keys()[0]
            model = name.split(":")[0]
            mean = float(h[name]["mean"]) / 1000.0
            p99 = float(h[name]["p99"]) / 1000.0
            mean_latencies[model] = round(float(mean), 3)
            p99_latencies[model] = round(float(p99), 3)
    meters = metrics_json["meters"]
    for m in meters:
        if "prediction_throughput" in m.keys()[0]:
            name = m.keys()[0]
            model = name.split(":")[0]
            rate = float(m[name]["rate"])
            thrus[model] = round(float(rate), 5)
    counters = metrics_json["counters"]
    counts = {}
    for c in counters:
        if "num_predictions" in c.keys()[0]:
            name = c.keys()[0]
            model = name.split(":")[0]
            count = c[name]["count"]
            counts[model] = int(count)
    return (mean_latencies, p99_latencies, thrus, counts)


def print_stats(client_metrics, clipper_metrics):
    results_dict = {}
    results_dict["batch_sizes"] = get_clipper_batch_sizes(clipper_metrics)
    results_dict["queue_sizes"] = get_clipper_queue_sizes(clipper_metrics)
    results_dict["clipper_thrus"] = get_clipper_thruputs(clipper_metrics)
    results_dict["clipper_counts"] = get_clipper_counts(clipper_metrics)
    results_dict["client_mean_lats"], results_dict["client_p99_lats"], \
        results_dict["client_thrus"], results_dict["client_counts"] = \
        get_profiler_stats(client_metrics)
    # logger.info(("\nClient thrus: {client_thrus}, clipper thrus: {clipper_thrus} "
    #              "\nclient counts: {client_counts}, clipper counts: {clipper_counts}, "
    #              "\nclient p99 lats: {client_p99_lats}, client mean lats: {client_mean_lats} "
    #              "\nqueue sizes: {queue_sizes}, "
    #              "batch sizes: {batch_sizes}\n").format(**results_dict))
    logger.info(("\nThroughput: {client_thrus}, p99 lat: {client_p99_lats}, "
                 "mean lat: {client_mean_lats} "
                 "\nbatches: {batch_sizes}, queues: {queue_sizes}\n").format(**results_dict))
    return results_dict


def load_metrics(client_path, clipper_path):
    with open(client_path, "r") as client_file, \
            open(clipper_path, "r") as clipper_file:
        client_metrics_str = client_file.read().strip()
        clipper_metrics_str = clipper_file.read().strip()
        if client_metrics_str[-1] == ",":
            client_metrics_str = client_metrics_str.rstrip(",")
            client_metrics_str += "]"
        if clipper_metrics_str[-1] == ",":
            clipper_metrics_str = clipper_metrics_str.rstrip(",")
            clipper_metrics_str += "]"
        try:
            client_metrics = json.loads(client_metrics_str)
        except ValueError:
            # logger.warn("Unable to parse client metrics: {}. Skipping...".format(e))
            return None
        try:
            clipper_metrics = json.loads(clipper_metrics_str)
        except ValueError:
            # logger.warn("Unable to parse clipper metrics: {}. Skipping...".format(e))
            return None
    return client_metrics, clipper_metrics


def load_lineage(lineage_path):
    with open(lineage_path, "r") as f:
        parsed = [json.loads(l) for l in f]
    return parsed


def run_profiler(config, trial_length, driver_path, input_size, profiler_cores_str,
                 workload_path=None):
    clipper_address = setup_clipper([config, ])
    clipper_address = CLIPPER_ADDRESS
    cl = ClipperConnection(DockerContainerManager(redis_port=6380))
    cl.connect()
    time.sleep(30)
    log_dir = "/tmp/{name}_profiler_logs_{ts:%y%m%d_%H%M%S}".format(name=config.name,
                                                                    ts=datetime.now())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    def run(target_throughput, num_trials, name, arrival_process, batch_size=None):
        cl.drain_queues()
        time.sleep(10)
        cl.drain_queues()
        time.sleep(10)
        log_path = os.path.join(log_dir, "{n}-{t}-{p}".format(n=name,
                                                              t=target_throughput,
                                                              p=arrival_process))
        cmd = ["numactl", "-C", profiler_cores_str,
               os.path.abspath(driver_path),
               "--name={}".format(config.name),
               "--input_type={}".format(config.input_type),
               "--input_size={}".format(input_size),
               "--target_throughput={}".format(target_throughput),
               "--request_distribution={}".format(arrival_process),
               "--trial_length={}".format(trial_length),
               "--num_trials={}".format(num_trials),
               "--log_file={}".format(log_path),
               "--clipper_address={}".format(clipper_address)]
        if batch_size is not None:
            cmd.append("--batch_size={}".format(batch_size))
        if workload_path is not None:
            cmd.append("--workload_path={}".format(workload_path))

        logger.info("Driver command: {}".format(" ".join(cmd)))
        client_path = "{p}-client_metrics.json".format(p=log_path)
        clipper_path = "{p}-clipper_metrics.json".format(p=log_path)
        lineage_path = "{p}-query_lineage.txt".format(p=log_path)
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
            recorded_trials = 0
            summary_results = []
            while recorded_trials < num_trials:
                time.sleep(5)
                if proc.poll() is not None:
                    if proc.returncode != 0:
                        logger.warn("Driver process finished with return code {}".format(
                            proc.returncode))
                        break
                    # else:
                    #     logger.warn("Driver process finished without enough trials. "
                    #                 "Expected {num}, recorded {rec}".format(num=num_trials,
                    #                                                         rec=recorded_trials))
                if not (os.path.exists(client_path) and os.path.exists(clipper_path)):
                    logger.info("metrics don't exist yet")
                    continue
                loaded_metrics = load_metrics(client_path, clipper_path)
                if loaded_metrics is not None:
                    client_metrics, clipper_metrics = loaded_metrics
                else:
                    continue
                client_trials = len(client_metrics)
                clipper_trials = len(clipper_metrics)
                if clipper_trials != client_trials:
                    logger.warn(("Clipper trials ({}) and client trials ({}) not the "
                                 "same length").format(clipper_trials, client_trials))
                else:
                    new_recorded_trials = clipper_trials
                    if new_recorded_trials > recorded_trials:
                        recorded_trials = new_recorded_trials
                        summary_results.append(print_stats(client_metrics[-1],
                                                           clipper_metrics[-1]))

            # prof_stdout = proc.stdout.read().strip()
            # if len(prof_stdout) > 0:
            #     logger.info("Profiler stdout: {}".format(prof_stdout))
            prof_stderr = proc.stderr.read().strip()
            if len(prof_stderr) > 0:
                logger.info("stderr: {}".format(prof_stderr))
            try:
                loaded_metrics = load_metrics(client_path, clipper_path)
                lineage = load_lineage(lineage_path)
                if loaded_metrics is not None:
                    client_metrics, clipper_metrics = loaded_metrics
                    return driver_utils.Results(client_metrics,
                                                clipper_metrics,
                                                summary_results,
                                                lineage)
                else:
                    logger.error("Error loading final metrics")
            except ValueError as e:
                logger.error("Unable to parse final metrics")
                raise e

    init_throughput = 2000
    run(init_throughput, 3, "warmup", "constant")
    throughput_results = run(init_throughput, 8, "throughput", "constant")
    cl.drain_queues()
    cl.set_full_batches()
    time.sleep(1)
    latency_results = run(0, 8, "latency", "batch", batch_size=config.batch_size)
    # cl.stop_all(remote_addrs=[REMOTE_ADDR])
    cl.stop_all()
    return throughput_results, latency_results


if __name__ == "__main__":
    gpu = 0
    model = TF_LANG_DETECT
    batch_sizes = [1, 2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64]
    for batch_size in batch_sizes:
        config = get_heavy_node_config(
            model_name=model,
            batch_size=batch_size,
            num_replicas=1,
            cpus_per_replica=1,
            allocated_cpus=[8],
            allocated_gpus=[7]
            # remote_addr=REMOTE_ADDR,
        )

        input_size = get_input_size(config)
        throughput_results, latency_results = run_profiler(
            config, 2000, "../../release/src/inferline_client/profiler",
            input_size, "9,25,10,26,11,27,12,28")
        fname = "v100-remote-{model}-batch-{batch}".format(
            model=model, batch=batch_size, gpu=gpu)
        results_dir = "{model}-SMP-gpu-{gpu}-remote".format(model=model, gpu=gpu)
        driver_utils.save_results_cpp_client(
            [config, ],
            throughput_results,
            latency_results,
            results_dir,
            prefix=fname)
    sys.exit(0)
