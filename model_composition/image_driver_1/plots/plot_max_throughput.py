import sys
import os
import json
import numpy as np
import seaborn as sns
import pandas as pd

from matplotlib import pyplot as plt

GPU_COST = 10
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

def load_results_data(dir_path):
    file_paths = [os.path.join(dir_path, file_path) for file_path in os.listdir(dir_path) if "results" in file_path]
    json_data = []
    for file_path in file_paths:
        results_file = open(file_path, "rb")
        results_json = json.load(results_file)
        results_file.close()
        json_data.append(results_json)
    return json_data

def process_results_data(json_data):
    costs = []
    avg_thrus = []
    # list of dicts mapping model image names to tuples of (num_cpus, num_gpus)
    resource_configs = []
    for results_json in json_data:
        relevant_thrus = np.array(results_json["client_metrics"]["thrus"][5:-1], dtype=np.float32)
        avg_thru = np.mean(relevant_thrus)
        node_configs = results_json["node_configs"][1:]
        cost = 0
        resource_config = {}
        for node_config in node_configs:
            num_replicas = int(node_config["num_replicas"])
            image_name = node_config["model_image"]
            if image_name in CPU_IMAGES:
                resource_config[image_name] = (num_replicas, 0)
                cost += (num_replicas * CPU_COST)
            elif image_name in GPU_IMAGES:
                resource_config[image_name] = (num_replicas, num_replicas)
                cost += (num_replicas * CPU_COST) + (num_replicas * GPU_COST)
            else:
                raise Exception("Model image was not recognized!")
        costs.append(cost)
        avg_thrus.append(avg_thru)
        resource_configs.append(resource_config)

    return costs, avg_thrus, resource_configs

def get_resource_config_str(resource_config):
    config_str = ""
    for image_name in resource_config:
        num_cpus, num_gpus = resource_config[image_name]
        config_str = config_str + " {} - C:{} G:{}\n".format(IMG_ABBREV_MAP[image_name], num_cpus, num_gpus)
    return config_str


def create_plot(costs, thrus, resource_configs):
    thrus = [0] + thrus
    costs = [costs[0]] + costs

    fig, ax = plt.subplots()
    step_plot = ax.step(thrus, costs, where="pre")
    ax.set_xlabel("Throughput (qps)")
    ax.set_ylabel("Cost (Dollars)")
    ax.set_title("Throughput maximization:\nCost as a function of throughput for Image Driver 1")
    ax.set_xlim(left=0)

    cpu_label, = ax.plot([], [], label="CPU COST: $1")
    gpu_label, = ax.plot([], [], label="GPU COST: $10")
    label_legend = plt.legend(handles=[cpu_label, gpu_label], loc="upper left", 
        handlelength=0, handletextpad=0, fancybox=True)
    plt.gca().add_artist(label_legend)

    point_plots = []
    for i in range(0, len(resource_configs)):
        point_plot, = ax.plot(
            [thrus[i + 1]], [costs[i + 1]], 
            marker="o", markeredgewidth=1.5, markeredgecolor="black", 
            label=get_resource_config_str(resource_configs[i]))
        point_plots.append(point_plot)

    legend = plt.legend(handles=point_plots, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)

    plt.savefig("max_throughput.png", bbox_extra_artists=[legend], bbox_inches='tight')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise

    data_path = sys.argv[1]

    json_data = load_results_data(data_path)
    costs, thrus, resource_configs = process_results_data(json_data)
    create_plot(costs, thrus, resource_configs)
