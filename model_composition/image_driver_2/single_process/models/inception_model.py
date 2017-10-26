import sys
import os
import numpy as np
import tensorflow as tf

from single_proc_utils import ModelBase

# Change this to the relative path of your local copy of tf slim
sys.path.insert(0, os.path.abspath('/home/ubuntu/tf_models/research/slim'))

from nets import inception_v3
from preprocessing import inception_preprocessing
from datasets import imagenet

image_size = inception_v3.inception_v3.default_image_size
slim = tf.contrib.slim

GPU_MEM_FRAC = .95

class InceptionModel(ModelBase):

    def __init__(self, inception_checkpoint_path, gpu_num):
        ModelBase.__init__(self)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEM_FRAC)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        with tf.device("/gpu:{}".format(gpu_num)):
            self.inputs = tf.placeholder(tf.float32, (None, image_size, image_size, 3))
            preprocessed_images = tf.map_fn(lambda input_img : inception_preprocessing.preprocess_image(input_img, image_size, image_size, is_training=False), self.inputs)

            with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                logits, _ = inception_v3.inception_v3(
                        preprocessed_images, num_classes=1001, is_training=False)
            self.all_probabilities = tf.nn.softmax(logits)
            init_fn = slim.assign_from_checkpoint_fn(
                    inception_checkpoint_path, slim.get_model_variables("InceptionV3"))
            init_fn(self.sess)

    def predict(self, inputs):
        """
        Parameters
        ----------
        inputs : list
                A list of image inputs encoded as numpy arrays of dimension (299, 299, 3)

        Returns
        ----------
        list 
                A list of integer ImageNet category classifications.
                The element at index `i` of this list is the classification
                of the corresponding element at index `i` in the inputs list.
        """
        all_probabilities = self.sess.run([self.all_probabilities], feed_dict={self.inputs: inputs})

        outputs = []
        for input_probabilities in all_probabilities[0]:
            sorted_inds = [i[0] for i in sorted(
                    enumerate(-input_probabilities), key=lambda x:x[1])]
            outputs.append(str(sorted_inds[0]))

        return outputs