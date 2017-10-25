import cv2
import numpy as np

from single_proc_utils import ModelBase

NUM_SIFT_FEATURES = 20

class SIFTFeaturizationModel(ModelBase):

	def __init__(self):
		ModelBase.__init__(self)
		self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=NUM_SIFT_FEATURES)

	def predict(self, inputs):
		"""
		Parameters
		----------
		inputs : list
		   	A list of images, each of which is represented
		   	as a numpy array of floats with shape 299 x 299 x 3
		"""
		inputs = [input_item.astype(np.uint8) for input_item in inputs]
		return [self._get_keypoints(input_img) for input_img in inputs]

	def _get_keypoints(self, img):
		grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		keypoints, features = self.sift.detectAndCompute(grayscale_img, None)
		return features[:NUM_SIFT_FEATURES]
