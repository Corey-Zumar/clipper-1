import sys
import os
import numpy as np
import tensorflow as tf

from single_proc_utils import ModelBase

# ResNet feature vectors are of size 2048
INPUT_VECTOR_SIZE = 2048

class TFKernelSVM(ModelBase):

    def __init__(self, kernel_size=2000, gpu_mem_frac=.95):
        ModelBase.__init__(self)

        self.kernel_data = self._generate_kernel_data(kernel_size)
        self.weights = self._generate_weights(kernel_size)
        self.labels = self._generate_labels(kernel_size)
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
            self.t_kernel : self.kernel_data,
            self.t_weights : self.weights,
            self.t_labels : self.labels,
            self.t_bias : self.bias,
            self.t_inputs : inputs
        }

        outputs = self.sess.run(self.t_outputs, feed_dict=feed_dict)
        outputs = outputs.flatten()

        return [np.array(item, dtype=np.float32) for item in outputs]

    def _create_prediction_graph(self):
        with tf.device("/gpu:0"):
            self.t_kernel = tf.placeholder(tf.float32, [None, INPUT_VECTOR_SIZE])
            self.t_inputs = tf.placeholder(tf.float32, [None, INPUT_VECTOR_SIZE])
            self.t_weights = tf.placeholder(tf.float32, [None, 1])
            self.t_labels = tf.placeholder(tf.float32, [None, 1])
            self.t_bias = tf.placeholder(tf.float32)
            gamma = tf.constant(-50.0)

            # Taken from https://github.com/nfmcclure/tensorflow_cookbook
            rA = tf.reshape(tf.reduce_sum(tf.square(self.t_kernel), 1),[-1,1])
            rB = tf.reshape(tf.reduce_sum(tf.square(self.t_inputs), 1),[-1,1])
            pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2.0, tf.matmul(self.t_kernel, tf.transpose(self.t_inputs)))), tf.transpose(rB))
            pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

            t_preds = tf.matmul(tf.multiply(tf.transpose(tf.multiply(self.t_labels, self.t_weights)), self.t_bias), pred_kernel)
            self.t_outputs = tf.sign(t_preds - tf.reduce_mean(t_preds))

    def _generate_bias(self):
        return np.random.uniform(-1,1) * 100

    def _generate_weights(self, training_data_size):
        return np.random.uniform(-1,1, size=(training_data_size, 1))

    def _generate_labels(self, training_data_size):
        return np.array(np.random.choice([-1,1], size=(training_data_size, 1)), dtype=np.float32)

    def _generate_kernel_data(self, kernel_size):
        return np.random.rand(kernel_size, INPUT_VECTOR_SIZE) * 10