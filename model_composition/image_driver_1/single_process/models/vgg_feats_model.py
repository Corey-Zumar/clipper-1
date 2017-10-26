import sys
import os

import numpy as np
import tensorflow as tf

from datetime import datetime
from single_proc_utils import ModelBase

GPU_MEM_FRAC = .95

class VggFeaturizationModel(ModelBase):

	def __init__(self, vgg_model_path, gpu_num):
		ModelBase.__init__(self)

		self.gpu_num = gpu_num
		self.imgs_tensor = self._load_vgg_model(vgg_model_path)

		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEM_FRAC)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

		graph = tf.get_default_graph()
		self.fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")

	def predict(self, inputs):
		"""
		Given a list of image inputs encoded as numpy arrays of data type
		np.float32, outputs a corresponding list of numpy arrays, each of 
		which is a featurized image
		"""
		reshaped_inputs = [input_item.reshape(224,224,3) for input_item in inputs]
		all_img_features = self._get_image_features(reshaped_inputs)
		return all_img_features

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

		with tf.device("/gpu:{}".format(self.gpu_num)):
			# Create placeholder for an arbitrary number
			# of RGB-encoded 224 x 224 images
			images_tensor = tf.placeholder("float", [None, 224, 224, 3])
			tf.import_graph_def(graph_def, input_map={ "images" : images_tensor})

		return images_tensor

	def _get_image_features(self, images):
		feed_dict = { self.imgs_tensor : images }
		fc7_features = self.sess.run(self.fc7_tensor, feed_dict=feed_dict)
		return fc7_features




