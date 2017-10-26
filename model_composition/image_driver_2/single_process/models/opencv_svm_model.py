import pickle

from single_proc_utils import ModelBase

class OpenCVSVM(ModelBase):

	def __init__(self, model_path):
		ModelBase.__init__(self)

		model_file = open(model_path, "rb")
		self.model = pickle.load(model_file)
		model_file.close()

	def predict(self, inputs):
		"""
		Parameters
		----------
		inputs : list
		   	A list of SIFT feature vectors, each
		   	represented as numpy array of data type `np.int32`
		"""
		inputs = [input_item.flatten() for input_item in inputs]
		return self.model.predict(inputs)