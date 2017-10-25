import sys
import os
import json

import tensorflow as tf
import numpy as np

from nmt_deps import gnmt_model, model_helper, hparam_utils
from nmt_deps import misc_utils as utils
from single_proc_utils import ModelBase

GPU_MEM_FRAC = .95

CHECKPOINT_RELATIVE_PATH = "translate.ckpt"
DEFAULT_HPARAMS_RELATIVE_PATH = "default_hparams.json"
MODEL_HPARAMS_RELATIVE_PATH = "model_hparams.json"
SOURCE_VOCAB_RELATIVE_PATH = "source_vocab.de"
TARGET_VOCAB_RELATIVE_PATH = "target_vocab.en"

# These hyperparameters are required for inference and are not specified
# by the provided set of JSON-formatted GNMT model hyperparameters

NMT_TEXT_END = "</s>"
NUM_TRANSLATIONS_PER_INPUT = 1

class NMTModel(ModelBase):

  def __init__(self, model_data_path, gpu_num):
    """
    Initializes the container

    Parameters
    ------------
    checkpoint_path : str
      The path to the GNMTModel checkpoint
    default_hparams_path : str
      The path to the set of default hyperparameters holding the same values as
      the flags specified in the `nmt.py` file
    model_hparams_path : str
      The path to the set of tuned GNMT hyperparameters
    source_vocab_path : str
      The path of the vocabulary associated with the source text (German)
    target_vocab_path : str
      The path of the vocabulary associated with the target text (English)
    """

    ModelBase.__init__(self)

    checkpoint_path = os.path.join(model_data_path, CHECKPOINT_RELATIVE_PATH)
    default_hparams_path = os.path.join(model_data_path, DEFAULT_HPARAMS_RELATIVE_PATH)
    model_hparams_path = os.path.join(model_data_path, MODEL_HPARAMS_RELATIVE_PATH)
    source_vocab_path = os.path.join(model_data_path, SOURCE_VOCAB_RELATIVE_PATH)
    target_vocab_path = os.path.join(model_data_path, TARGET_VOCAB_RELATIVE_PATH)

    assert os.path.exists(default_hparams_path)
    assert os.path.exists(model_hparams_path)
    assert os.path.exists(source_vocab_path)
    assert os.path.exists(target_vocab_path)

    self.sess, self.nmt_model, self.infer_model, self.hparams = \
    self._load_model(checkpoint_path,
                     default_hparams_path,
                     model_hparams_path,
                     source_vocab_path,
                     target_vocab_path,
                     gpu_num)


  def predict(self, inputs):
    """
    Parameters
    -------------
    inputs : [string]
      A list of strings of German text
    """
    infer_batch_size = len(inputs)
    self.sess.run(
        self.infer_model.iterator.initializer,
        feed_dict={
            self.infer_model.src_placeholder: inputs,
            self.infer_model.batch_size_placeholder: infer_batch_size
    })

    outputs = []

    nmt_outputs, _ = self.nmt_model.decode(self.sess)
    for output_id in range(infer_batch_size):
      for translation_index in range(NUM_TRANSLATIONS_PER_INPUT):
        output = self._get_translation(nmt_outputs[translation_index],
                                       output_id,
                                       tgt_eos=None,
                                       subword_option=self.hparams.subword_option)
        end_idx = output.find(NMT_TEXT_END)
        if end_idx >= 0:
            output = output[:end_idx]
        outputs.append(output)

    return outputs

  def _create_hparams(self, default_hparams_path, model_hparams_path, source_vocab_path, target_vocab_path, gpu_num):
    partial_hparams = tf.contrib.training.HParams()
    default_hparams_file = open(default_hparams_path, "rb")
    default_hparams = json.load(default_hparams_file)
    default_hparams_file.close()
    for param in default_hparams:
      partial_hparams.add_hparam(param, default_hparams[param])
    partial_hparams.set_hparam("num_gpus", gpu_num)

    hparams = hparam_utils.load_hparams(model_hparams_path, partial_hparams)
    hparams = hparam_utils.extend_hparams(hparams, source_vocab_path, target_vocab_path)
    return hparams

  def _load_model(self,
                  checkpoint_path,
                  default_hparams_path,
                  model_hparams_path,
                  source_vocab_path,
                  target_vocab_path,
                  gpu_num):
    hparams = self._create_hparams(default_hparams_path, model_hparams_path, source_vocab_path, target_vocab_path, gpu_num)

    model_creator = gnmt_model.GNMTModel
    infer_model = model_helper.create_infer_model(model_creator, hparams, scope=None)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEM_FRAC)
    sess = tf.Session(graph=infer_model.graph, config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    with infer_model.graph.as_default():
      nmt_model = model_helper.load_model(infer_model.model, checkpoint_path, sess, "infer")

    return sess, nmt_model, infer_model, hparams

  def _get_translation(self, nmt_outputs, sent_id, tgt_eos, subword_option):
    """Given batch decoding outputs, select a sentence and turn to text."""
    if tgt_eos: 
      tgt_eos = tgt_eos.encode("utf-8")
    # Select a sentence
    output = nmt_outputs[sent_id, :].tolist()

    # If there is an eos symbol in outputs, cut them at that point.
    if tgt_eos and tgt_eos in output:
      output = output[:output.index(tgt_eos)]

    if subword_option is None:
      translation = utils.format_text(output)
    elif subword_option == "bpe":  # BPE
      translation = utils.format_bpe_text(output)

    if subword_option == "spm":  # SPM
      translation = utils.format_spm_text(output)

    return translation
