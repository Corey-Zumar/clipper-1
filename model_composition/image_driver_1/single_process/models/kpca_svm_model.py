import pickle

from single_proc_utils import ModelBase
from deps.kpca_svm_utils import KpcaSvmModel

class KpcaSVM(ModelBase):

	def __init__(self, ks_model_path):
		ModelBase.__init__(self)

		ks_model_file = open(ks_model_path, "rb")
		self.ks_model = pickle.load(ks_model_file)
		ks_model_file.close()

	def predict(self, inputs):
		"""
		Given a list of vgg feature vectors encoded as numpy arrays of data type
		np.float32, outputs a corresponding list of image category labels
		"""
		return self.ks_model.evaluate(inputs)