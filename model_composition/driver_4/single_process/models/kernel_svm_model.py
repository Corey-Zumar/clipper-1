import pickle

from single_proc_utils import ModelBase

class KernelSVM(ModelBase):

	def __init__(self, model_path):
		ModelBase.__init__(self)

		model_file = open(model_path, "rb")
		self.model = pickle.load(model_file)
		model_file.close()

	def predict(self, inputs):
		"""
		Given a list of vgg feature vectors encoded as numpy arrays of data type
		np.float32, outputs a corresponding list of image category labels
		"""
		return self.model.evaluate(inputs)