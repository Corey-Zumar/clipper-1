import sys
import os
import numpy as np
import pickle

from kernel_svm_model import KernelSvmModel

VGG_FEATURE_VEC_SIZE = 4096

def train_svm_model(training_data_size, output_path):
	kpca_num_components = 100
	kpca_gamma = 10e-8
	model = KernelSvmModel()

	training_data = 10 * np.random.rand(training_data_size, VGG_FEATURE_VEC_SIZE)
	training_labels = np.random.randint(2, size=training_data_size)

	model.train(training_data, training_labels)
	output_file = open(output_path, "w")
	pickle.dump(model, output_file)
	output_file.close()

	print("Trained and saved model!")

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("Usage is 'python kernel_svm_trainer.py <training_data_size> <model_output_path>")
		raise

	training_data_size = sys.argv[1]
	model_output_path = sys.argv[2]

	train_svm_model(training_data_size, model_output_path)
