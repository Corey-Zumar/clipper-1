from __future__ import print_function
import sys
import os
import rpc
import logging
import numpy as np
import tensorflow as tf

from alexnet import AlexNet

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

DROPOUT_KEEP_PROBABILITY = .5

class TfAlexNetContainer(rpc.ModelContainerBase):
    def __init__(self, model_weights_path, gpu_mem_frac=.95): 
        assert os.path.exists(model_weights_path)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        self._load_model(model_weights_path)

        self.sess.run(tf.global_variables_initializer())

        logger.info("AlexNet is ready!")

    def predict_floats(self, inputs):
        """
        Parameters
        ----------
        inputs : list
            A list of 3-channel, 224 x 224 images, each represented
            as a numpy array
        """
        try:
            reshaped_inputs = [input_item.reshape(224,224,3) for input_item in inputs]
            all_img_features = self._get_image_features(reshaped_inputs)
            return all_img_features
        except Exception as e:
            print(e)

    def _get_image_features(self, images):
        feed_dict = { self.model.input_tensor : images }
        features = self.sess.run(self.model.output_tensor, feed_dict=feed_dict)
        return features

    def _load_model(self, model_weights_path):
        with tf.device("/gpu:0"):
            self.t_inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
            self.t_keep_probability = tf.constant(DROPOUT_KEEP_PROBABILITY, dtype=tf.float32)

            logger.info("Creating AlexNet architecture...")

            self.model = AlexNet(input_tensor=self.t_inputs, 
                                 keep_prob=self.t_keep_probability, 
                                 skip_layer=[],
                                 weights_path=model_weights_path)

            logger.info("Loading AlexNet weights...")

            self.model.load_initial_weights(self.sess)

if __name__ == "__main__":
    print("Starting Tensorflow AlexNet Container")
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
        model_weights_path = os.environ["CLIPPER_MODEL_WEIGHTS_PATH"]
    except KeyError:
        print(
            "ERROR: CLIPPER_MODEL_WEIGHTS_PATH environment variable must be set",
            file=sys.stdout)
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

    input_type = "floats"
    container = TfAlexNetContainer(model_weights_path)
    rpc_service = rpc.RPCService()
    rpc_service.start(container, ip, model_name, model_version, input_type)
