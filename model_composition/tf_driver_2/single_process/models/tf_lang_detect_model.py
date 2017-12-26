from __future__ import print_function
import sys
import os
import json

import tensorflow as tf
import numpy as np

from tf_lang_detect_deps import cnn, util

from single_proc_utils import ModelBase

GPU_MEM_FRAC = .95

CONFIG_RELATIVE_PATH = "tf-lang-config.cPickle"
CHECKPOINT_RELATIVE_PATH = "model_ckpt"
VOCAB_RELATIVE_PATH = "vocab"

"""
Adapted from https://github.com/may-/cnn-ld-tf
"""
class LangDetectModel(ModelBase):

    def __init__(self, model_data_path):
        self.sess, self.inputs_tensor, self.scores_tensor = self._load_model(model_data_path)
        self.vocab = self._load_vocab(model_data_path)

    def predict(self, inputs):
        """
        Parameters
        ------------
        inputs : [str]
            A list of string inputs in one of 64 languages
        """
        inputs = [str(input_item.tobytes()) for input_item in inputs]

        ids_inputs = np.array([self.vocab.text2id(input_text) for input_text in inputs])
        print(ids_inputs.shape)

        feed_dict = {
            self.inputs_tensor : ids_inputs
        }
        all_scores = self.sess.run(self.scores_tensor, feed_dict=feed_dict)

        outputs = []
        for score_dist in all_scores:
            parsed_dist = [float(str(i)) for i in score_dist]
            pred_class = self.vocab.class_names[int(np.argmax(parsed_dist))]
            print(pred_class)
            outputs.append(str(pred_class.replace("#", "")))

        return outputs
        
    def _load_model(self, model_data_path):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEM_FRAC)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, 
            allow_soft_placement=True, device_count = {'GPU': 0}))

        config = self._load_config(model_data_path)

        checkpoint_path = os.path.join(model_data_path, CHECKPOINT_RELATIVE_PATH)

        with tf.device("/gpu:0"):
            with tf.variable_scope('cnn'):
                model = cnn.Model(config, is_train=False)

            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Loading checkpoint file failed!")

            inputs_tensor = model.inputs
            scores_tensor = model.scores

        return sess, inputs_tensor, scores_tensor

    def _load_config(self, model_data_path):
        config_path = os.path.join(model_data_path, CONFIG_RELATIVE_PATH)
        config = util.load_from_dump(config_path)
        return config

    def _load_vocab(self, model_data_path):
        vocab_path = os.path.join(model_data_path, VOCAB_RELATIVE_PATH)
        return util.VocabLoader(vocab_path)