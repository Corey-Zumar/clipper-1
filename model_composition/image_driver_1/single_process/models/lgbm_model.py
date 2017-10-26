import lightgbm as lgb
import numpy as np

from single_proc_utils import ModelBase

class ImagesGBM(ModelBase):

	def __init__(self, gbm_model_path):
		ModelBase.__init__(self)
		self.gbm_model = lgb.Booster(model_file=gbm_model_path)

	def predict(self, inputs):
		"""
		Parameters
		----------
		inputs : list
		   	A list of inception feature vectors encoded
		   	as numpy arrays of type float32
		"""
		stacked_inputs = np.stack([input_item[0][0] for input_item in inputs])
		return self.gbm_model.predict(stacked_inputs)