import sys
import os
import numpy as np
import tensorflow as tf

from single_proc_utils import ModelBase

# Inception feature vectors are of size 2048
INPUT_VECTOR_SIZE = 2048

class TfLogRegModel(ModelBase):

    def __init__(self, gpu_mem_frac=.95):
        ModelBase.__init__(self)

        self.weights = self._generate_weights()
        self.bias = self._generate_bias()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        self._create_prediction_graph()


    def predict(self, inputs):
        """
        Parameters
        --------------
        inputs : [np.ndarray]
            A list of float vectors of length 2048,
            represented as numpy arrays
        """
        
        feed_dict = {
            self.t_weights : self.weights,
            self.t_bias : self.bias,
            self.t_inputs : inputs
        }

        outputs = self.sess.run(self.t_outputs, feed_dict=feed_dict)
        outputs = outputs.flatten()

        return [np.array(item, dtype=np.float32) for item in outputs]

    def _create_prediction_graph(self):
        with tf.device("/gpu:0"):
            self.t_inputs = tf.placeholder(tf.float32, [None, INPUT_VECTOR_SIZE])
            self.t_weights = tf.placeholder(tf.float32, [INPUT_VECTOR_SIZE, 1])
            self.t_bias = tf.placeholder(tf.float32)

            t_apply_weights = tf.reduce_sum(tf.multiply(self.t_weights, tf.transpose(self.t_inputs)), axis=0)
            t_sig_input = t_apply_weights + self.t_bias

            self.t_outputs = tf.sigmoid(t_sig_input)

    def _generate_bias(self):
        return np.random.uniform(-1,1) * 100

    def _generate_weights(self):
        return np.random.uniform(-1,1, size=(INPUT_VECTOR_SIZE,1))