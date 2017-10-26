import sys
import os
import gensim
import numpy as np

from single_proc_utils import ModelBase

MODEL_RELATIVE_PATH = "#TODO: FILL IN"
DICTIONARY_RELATIVE_PATH = "#TODO: FILL IN"

class SimilarityModel(ModelBase):

	def __init__(self, model_data_path):
		ModelBase.__init__(self)

		model_path = os.path.join(model_data_path, MODEL_RELATIVE_PATH)
		dictionary_path = os.path.join(model_data_path, DICTIONARY_RELATIVE_PATH)

		assert os.path.exists(model_path)
		assert os.path.exists(dictionary_path)

		self.word_ids_dict = gensim.corpora.Dictionary.load_from_text(dictionary_path)
		self.model = gensim.similarities.docsim.SparseMatrixSimilarity.load(model_path, mmap='r')

	def predict(self, inputs):
		"""
		Parameters
		----------
		inputs : list
		   	A list of documents, represented as strings

		Returns
		----------
		list 
			A list of document ids. The output at index `i`
			is the index of the document predicted to be most
			similar to the input document at index `i`
		"""

		outputs = []
		for input_doc in inputs:
			doc_bow = self.word_ids_dict.doc2bow(input_doc.split())
			docsim_dist = self.model[doc_bow]
			best_doc_index = np.argmax(docsim_dist)
			outputs.append(str(best_doc_index))

		return outputs