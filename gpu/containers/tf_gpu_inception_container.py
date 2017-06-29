from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import sys
import base64
import time

cur_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.abspath('../../containers/python'))

import rpc

# Change this to the relative path of tf slim within your docker container
sys.path.insert(0, os.path.abspath('/tfslim'))

from nets import inception_v3
from preprocessing import inception_preprocessing
from datasets import imagenet

image_size = inception_v3.inception_v3.default_image_size
slim = tf.contrib.slim

class TfInceptionContainer(rpc.ModelContainerBase):

    def __init__(self, checkpoint_path, gpu_mem_frac):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac)
        self.checkpoint_path = checkpoint_path
        self.sess = tf.Session("", tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options))
        with self.sess.graph.as_default():
            self.inputs = tf.placeholder(tf.float32, (None, image_size, image_size, 3))
            with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                logits, _ = inception_v3.inception_v3(
                    self.inputs, num_classes=1001, is_training=False)
            self.all_probabilities = tf.nn.softmax(logits)
            init_fn = slim.assign_from_checkpoint_fn(
                self.checkpoint_path, slim.get_model_variables("InceptionV3"))
            init_fn(self.sess)


    def predict_bytes(self, inputs):
        outputs = []
        with self.sess.graph.as_default():
            decoded_images = [tf.image.decode_jpeg(input_image.tobytes(), channels=3) for input_image in inputs]
            preprocessed_images = [
                inception_preprocessing.preprocess_image(
                    decoded_image,
                    image_size,
                    image_size,
                    is_training=False) for decoded_image in decoded_images]

        processed_inputs = self.sess.run(preprocessed_images)
        all_probabilities = self.sess.run([self.all_probabilities], feed_dict={self.inputs: processed_inputs})
        for input_probabilities in all_probabilities[0]:
            sorted_inds = [i[0] for i in sorted(
                enumerate(-input_probabilities), key=lambda x:x[1])]
            outputs.append(str(sorted_inds[0]))

        print(outputs)
        return outputs


if __name__ == "__main__":
    print("Starting InceptionV3 Cifar container")
    try:
        model_name = os.environ["CLIPPER_MODEL_NAME"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_NAME environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        model_version = os.environ["CLIPPER_MODEL_VERSION"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_VERSION environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        model_checkpoint_path = os.environ["CLIPPER_MODEL_CHECKPOINT_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_CHECKPOINT_PATH environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
    try:
        gpu_mem_frac = os.environ["CLIPPER_GPU_MEM_FRAC"]
    except KeyError:
        print("ERROR: CLIPPER_GPU_MEM_FRAC environment variable must be set", file=sys.stdout)
        sys.exit(1)

    ip = "127.0.0.1"
    if "CLIPPER_IP" in os.environ:
        ip = os.environ["CLIPPER_IP"]
    else:
        print("Connecting to Clipper on localhost")

    print("CLIPPER IP: {}".format(ip))

    port = 7000
    if "CLIPPER_PORT" in os.environ:
        port = int(os.environ["CLIPPER_PORT"])
    else:
        print("Connecting to Clipper with default port: 7000")

    input_type = "bytes"
    container = TfInceptionContainer(model_checkpoint_path, gpu_mem_frac)
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, port, model_name, model_version, input_type)
                                                                                                                                                 1,1           Top

