import sys
import os

import numpy as np
import tensorflow as tf

from single_proc_utils import ModelBase

GRAPH_RELATIVE_PATH = "tf_resnet_152_feats_graph.meta"
CHECKPOINT_RELATIVE_PATH = "tf_resnet_152_feats.ckpt"

class TfResNetModel(ModelBase):

    def __init__(self, model_data_path, gpu_mem_frac=.95):
        ModelBase.__init__(self)

        graph_path = os.path.join(model_data_path, GRAPH_RELATIVE_PATH)
        checkpoint_path = os.path.join(model_data_path, CHECKPOINT_RELATIVE_PATH)

        assert os.path.exists(graph_path)
        assert os.path.exists(checkpoint_path)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_frac)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        self._load_model(graph_path, checkpoint_path)

    def predict(self, inputs):
        """
        Parameters
        ----------
        inputs : list
            A list of 3-channel, 224 x 224 images, each represented
            as a numpy array
        """

        reshaped_inputs = [input_item.reshape(224,224,3) for input_item in inputs]
        all_img_features = self._get_image_features(reshaped_inputs)
        return [np.array(item, dtype=np.float32) for item in all_img_features]

    def _get_image_features(self, images):
        feed_dict = { self.t_images : images }
        features = self.sess.run(self.t_features, feed_dict=feed_dict)
        return features

    def _load_model(self, model_graph_path, model_ckpt_path):
        with tf.device("/gpu:0"):
            saver = tf.train.import_meta_graph(model_graph_path, clear_devices=True)
            saver.restore(self.sess, model_ckpt_path)
            self.t_images = tf.get_default_graph().get_tensor_by_name('images:0')
            self.t_features = tf.get_default_graph().get_tensor_by_name('avg_pool:0')