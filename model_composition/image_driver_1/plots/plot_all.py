import sys
import os
import json
import numpy as np
import seaborn as sns
import pandas as pd

from matplotlib import pyplot as plt

GPU_COST = 14
CPU_COST = 1

VGG_FEATS_IMAGE_NAME = "model-comp/vgg-feats"
INCEPTION_FEATS_IMAGE_NAME = "model-comp/inception-feats"
KERNEL_SVM_IMAGE_NAME = "model-comp/kernel-svm"
LGBM_IMAGE_NAME = "model-comp/lgbm"

CPU_IMAGES = [
    KERNEL_SVM_IMAGE_NAME,
    LGBM_IMAGE_NAME
]

GPU_IMAGES = [
    VGG_FEATS_IMAGE_NAME,
    INCEPTION_FEATS_IMAGE_NAME
]

IMG_ABBREV_MAP = {
    VGG_FEATS_IMAGE_NAME : "vgg",
    INCEPTION_FEATS_IMAGE_NAME : "incep",
    KERNEL_SVM_IMAGE_NAME : "svm",
    LGBM_IMAGE_NAME : "lgbm"
}

LATENCY_THRESHOLD = .005

def load_results_data(dir_path):
    file_paths = [os.path.join(dir_path, file_path) for file_path in os.listdir(dir_path) if "results" in file_path]
    json_data = []
    for file_path in file_paths:
        results_file = open(file_path, "rb")
        results_json = json.load(results_file)
        results_file.close()
        json_data.append(results_json)
    return json_data

def process_data(json_data):
    costs = []
    avg_thrus = []
    max_p99_lats = []
    # list of dicts mapping model image names to tuples of (num_cpus, num_gpus)
    resource_configs = []
    for results_json in json_data:
        relevant_thrus = np.array(results_json["client_metrics"]["thrus"][3:-2], dtype=np.float32)
        avg_thru = np.mean(relevant_thrus)
        p99_lats = np.array(results_json["client_metrics"]["p99_lats"], dtype=np.float32)
        p99_lats = [lat for lat in p99_lats if lat < min(p99_lats) + LATENCY_THRESHOLD]
        max_p99_lat = np.max(p99_lats)
        node_configs = results_json["node_configs"][1:]
        cost = 0
        resource_config = {}
        for node_config in node_configs:
            num_replicas = int(node_config["num_replicas"])
            image_name = node_config["model_image"]
            if image_name in CPU_IMAGES:
                cost += (num_replicas * CPU_COST)
                resource_config[image_name] = (num_replicas, 0)
            elif image_name in GPU_IMAGES:
                cost += (num_replicas * CPU_COST) + (num_replicas * GPU_COST)
                resource_config[image_name] = (num_replicas, num_replicas)
            else:
                raise Exception("Model image was not recognized!")
        costs.append(cost)
        avg_thrus.append(avg_thru)
        max_p99_lats.append(max_p99_lat)
        resource_configs.append(resource_config)


    bundle = sorted(zip(costs, avg_thrus, max_p99_lats, resource_configs))
    sorted_costs = [item[0] for item in bundle]
    sorted_thrus = [item[1] for item in bundle]
    sorted_lats = [item[2] for item in bundle]
    sorted_configs = [item[3] for item in bundle]

    return sorted_costs, sorted_thrus, sorted_lats, sorted_configs

def plot_max_thrus(fig, ax, costs, thrus):
    return ax.scatter(thrus, costs, label="Maximize Throughput")

def plot_min_lats(fig, ax, costs, p99_lats, thrus):
    max_p99_lat = np.max(p99_lats)
    return ax.scatter(thrus, costs, label="Minimize Latency - SLO: {0:.2f} ms".format(max_p99_lat * 1000))

def plot_expert(fig, ax, costs, p99_lats, thrus):
    max_p99_lat = np.max(p99_lats)
    return ax.scatter(thrus, costs, label="Expert - SLO: {0:.2f} ms".format(max_p99_lat * 1000))

def annotate(fig, ax):
    ax.set_xlabel("Throughput (qps)")
    ax.set_ylabel("Cost (Dollars)")
    ax.set_title("Image Driver 1:\nCost as a function of throughput for varying greedy approaches")
    ax.set_xlim(left=0, right=250)
    ax.set_ylim(bottom=0)

    cpu_label, = ax.plot([], [], label="CPU COST: $1")
    gpu_label, = ax.plot([], [], label="GPU COST: $14")
    label_legend = plt.legend(handles=[cpu_label, gpu_label], loc="upper left", 
        handlelength=0, handletextpad=0, fancybox=True)
    plt.gca().add_artist(label_legend)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise

    mt_path = sys.argv[1]
    ml_path = sys.argv[2]
    expert_path = sys.argv[3]

    mt_data = load_results_data(mt_path)
    ml_data = load_results_data(ml_path)
    expert_data = load_results_data(expert_path)

    mt_costs, mt_thrus, mt_p99_lats, _ = process_data(mt_data)
    ml_costs, ml_thrus, ml_p99_lats, _ = process_data(ml_data)
    exp_costs, exp_thrus, exp_p99_lats, _ = process_data(expert_data)

    fig, ax = plt.subplots()

    mt_plot = plot_max_thrus(fig, ax, mt_costs, mt_thrus)
    ml_plot = plot_min_lats(fig, ax, ml_costs, ml_p99_lats, ml_thrus)
    expert_plot = plot_expert(fig, ax, exp_costs, exp_p99_lats, exp_thrus)

    annotate(fig, ax)

    legend = plt.legend(handles=[mt_plot, ml_plot, expert_plot], loc='lower right')

    plt.savefig("test.png", bbox_extra_artists=[legend], bbox_inches='tight')
