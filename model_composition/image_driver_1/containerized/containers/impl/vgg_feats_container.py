from __future__ import print_function
import sys
import os
import rpc

import numpy as np
import tensorflow as tf

class VggFeaturizationContainer:

    def __init__(self, vgg_model_path, gpu_mem_frac):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        graph = tf.get_default_graph()
        self.imgs_tensor = self._load_vgg_model(vgg_model_path)
        self.fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")

    def predict_floats(self, inputs):
        """
        Given a list of image inputs encoded as numpy arrays of data type
        np.float32, outputs a corresponding list of numpy arrays, each of 
        which is a featurized image
        """
        time.sleep(.444)
        # reshaped_inputs = [input_item.reshape(224,224,3) for input_item in inputs]
        # all_img_features = self._get_image_features(reshaped_inputs)
        # return [np.array(item, dtype=np.float32) for item in all_img_features]
        return [np.array(np.random.rand(4096), dtype=np.float32) for item in inputs]

    def _load_vgg_model(self, vgg_model_path):
        vgg_file = open(vgg_model_path, mode='rb')
        vgg_text = vgg_file.read()
        vgg_file.close()

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(vgg_text)

        # Clear pre-existing device specifications
        # within the tensorflow graph
        for node in graph_def.node:
            node.device = ""

        with tf.device("/gpu:0"):
            # Create placeholder for an arbitrary number
            # of RGB-encoded 224 x 224 images
            images_tensor = tf.placeholder("float", [None, 224, 224, 3])
            tf.import_graph_def(graph_def, input_map={ "images" : images_tensor})

        return images_tensor

    def _get_image_features(self, images):
        feed_dict = { self.imgs_tensor : images }
        fc7_features = self.sess.run(self.fc7_tensor, feed_dict=feed_dict)
        return fc7_features

if __name__ == "__main__":
    print("Starting VGG Featurization Container")
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
        model_graph_path = os.environ["CLIPPER_MODEL_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_PATH environment variable must be set",
            file=sys.stdout)
        sys.exit(1)
        
    gpu_mem_frac = .9
    if "CLIPPER_GPU_MEM_FRAC" in os.environ:
        gpu_mem_frac = float(os.environ["CLIPPER_GPU_MEM_FRAC"])
    else:
        print("Using all available GPU memory")

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

    input_type = "floats"
    container = VggFeaturizationContainer(model_graph_path, gpu_mem_frac)
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, port, model_name, model_version,
                      input_type)