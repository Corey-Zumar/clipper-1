import sys
import os

import numpy as np
import tensorflow as tf
import single_proc_utils

from single_proc_utils import ModelBase

GPU_MEM_FRAC = .95

class InceptionFeaturizationModel(ModelBase):

    def __init__(self, inception_model_path, gpu_num):
        ModelBase.__init__(self)

        self.gpu_num = gpu_num

        self.images_tensor = self._load_inception_model(inception_model_path)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEM_FRAC)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        graph = tf.get_default_graph()
        self.features_tensor = graph.get_tensor_by_name("pool_3:0")

    def predict(self, inputs):
        """
        Parameters
        ----------
        inputs : list
            A list of 299 x 299 x 3 images, represented as
            an np.ndarray of floats

        Returns
        ----------
        list
            A list of featurized images, each represented as a numpy array
        """
        
        reshaped_inputs = [input_item.reshape(299,299,3) for input_item in inputs]
        all_img_features = self._get_image_features(reshaped_inputs)
        return [np.array(item.flatten(), dtype=np.float32) for item in all_img_features]

    def _load_inception_model(self, inception_model_path):
        inception_file = open(inception_model_path, mode='rb')
        inception_text = inception_file.read()
        inception_file.close()

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(inception_text)

        # Clear pre-existing device specifications
        # within the tensorflow graph
        for node in graph_def.node:
            node.device = ""

        with tf.device("/gpu:{}".format(self.gpu_num)):
            images_tensor = tf.placeholder(tf.float32, shape=[None,299,299,3])
            tf.import_graph_def(graph_def, name='', input_map={ "ResizeBilinear:0" : images_tensor})

        return images_tensor

    def _get_image_features(self, image):
        feed_dict = { self.images_tensor : image }
        features = self.sess.run(self.features_tensor, feed_dict=feed_dict)
        return features