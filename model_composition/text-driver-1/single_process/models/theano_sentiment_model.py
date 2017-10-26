import sys
import os
import pickle
import re
import numpy as np

from deps import lstm_utils, imdb_utils
from single_proc_utils import ModelBase 

MODEL_RELATIVE_PATH = "theano_lstm_model.npz"
MODEL_OPTIONS_RELATIVE_PATH = "theano_model_options.txt"
MODEL_IMDB_DICT_RELATIVE_PATH = "theano_imdb.dict.pkl"

class MovieSentimentLstm(ModelBase):

	def __init__(self, model_data_path):
		ModelBase.__init__(self)

		model_path = os.path.join(model_data_path, MODEL_RELATIVE_PATH)
		options_path = os.path.join(model_data_path, MODEL_OPTIONS_RELATIVE_PATH)
		dict_path = os.path.join(model_data_path, MODEL_IMDB_DICT_RELATIVE_PATH)

		assert os.path.exists(model_path)
		assert os.path.exists(options_path)
		assert os.path.exists(dict_path)

		model_options_file = open(options_path, "r")
		model_options = {}
		for line in model_options_file.readlines():
			key, value = line.split('@')
			value = value.strip()
			try:
			    value = int(value)
			except ValueError as e:
			    pass
			model_options[key] = value

		model_options_file.close()

		self.n_words = model_options['n_words']
		params = lstm_utils.init_params(model_options)
		params = lstm_utils.load_params(model_path, params)
		tparams = lstm_utils.init_tparams(params)

		(use_noise, x, mask, y, f_pred_prob, f_pred, cost) = lstm_utils.build_model(tparams, model_options)
		self.predict_function = f_pred

		dict_file = open(dict_path, 'rb')
		self.imdb_dict = pickle.load(dict_file)
		dict_file.close()
		print("Done!")

	def predict(self, inputs):
		reviews_features = self._get_reviews_features(inputs)
		x, mask, y = self._prepare_reviews_data(reviews_features)
		predictions = self.predict_function(x, mask)
		return [str(pred) for pred in predictions]

	def _get_reviews_features(self, reviews):
		all_review_indices = []
		for review in reviews:
			review_indices = self._get_imdb_indices(review)
			all_review_indices.append(review_indices)
		return all_review_indices

	def _get_imdb_indices(self, input_str):
		split_input = input_str.split(" ")
		indices = np.ones(len(split_input))
		for i in range(0, len(split_input)):
			term = split_input[i]
			term = re.sub('[^a-zA-Z\d\s:]', '', term)
			if term in self.imdb_dict:
				index = self.imdb_dict[term]
				if index < self.n_words:
					indices[i] = index
		return indices

	def _prepare_reviews_data(self, reviews_features):
		x, mask, y = imdb_utils.prepare_data(reviews_features, [], maxlen=None)
		return x, mask, y