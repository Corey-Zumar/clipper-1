from __future__ import division
from sklearn import svm, preprocessing
import numpy as np
import pdb
import sys

class KernelSvmModel:
	
	def __init__(self):
		"""
		Initializes the model
		"""

		self.trained = False

	def train(self, training_inputs, training_labels):
		"""
		Trains the model

		Parameters
		----------
		training_inputs : np.ndarray
			An m-by-n matrix of training inputs (m vectors of n features each)

		training_labels : np.ndarray
			An m-by-1 matrix of training labels
		"""
		self.scaler = preprocessing.StandardScaler().fit(training_inputs)
		training_inputs = self.scaler.transform(training_inputs)
		self.classifier = self.fit_svm(training_inputs, training_labels)
		self.trained = True


	def evaluate(self, inputs):
		"""
		Evaluates the model on a matrix of inputs

		Parameters
		----------
		inputs : np.ndarray
			An m-by-n matrix (m vectors of n features each)

		Returns
		----------
		np.ndarray
			An m-by-1 matrix of predicted labels
		"""

		if not self.trained:
			print("Cannot evaluate an untrained model!")
			raise

		inputs = self.scaler.transform(inputs)
		outputs = self.classifier.predict(inputs)

		return outputs	

	def fit_svm(self, inputs, labels):
		"""
		Fits an SVM to training data
		"""
		classifier = svm.SVC(kernel="rbf")
		classifier.fit(inputs, labels)
		return classifier