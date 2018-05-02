import sys
import os
import numpy as np
import tensorflow as tf
import logging

import AlexNet

from single_proc_utils import ModelBase

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

GRAPH_RELATIVE_PATH = "alexnet_frozen.pb"

DROPOUT_KEEP_PROBABILITY = .5

class AlexNetModel(ModelBase):

    def __init__(self, model_data_path, gpu_num, gpu_mem_frac=.95):
        ModelBase.__init__(self)

        graph_path = os.path.join(model_data_path, GRAPH_RELATIVE_PATH)

        assert os.path.exists(graph_path)

        self.t_inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.t_keep_probability = tf.constant(DROPOUT_KEEP_PROBABILITY, dtype=tf.float32)


        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        logger.info("Creating AlexNet architecture...")

        self.model = AlexNet(input_tensor=self.t_inputs, 
                             keep_prob=self.t_keep_probability, 
                             skip_layer=[])

        logger.info("Loading AlexNet weights...")

        self.model.load_initial_weights(self.sess)

        logger.info("AlexNet is ready!")

    def predict(self, inputs):
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
