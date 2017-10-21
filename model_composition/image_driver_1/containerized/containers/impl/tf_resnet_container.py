# from __future__ import print_function
import sys
# import os
# import rpc

import numpy as np
import tensorflow as tf

# class TfResNetContainer(rpc.ModelContainerBase):

# 	def __init__(self, model_path):
# 		pass

if __name__ == "__main__":
	if len(sys.argv) < 3:
		raise

	graph_def_path = sys.argv[1]
	ckpt_path = sys.argv[2]

	sess = tf.Session()

	saver = tf.train.import_meta_graph(graph_def_path)

	for n in tf.get_default_graph().as_graph_def().node:
		if not n.device or len(n.device) == 0:
			print("AH")
		else:
			print(n.device)

	# print([n.name for n in tf.get_default_graph().as_graph_def().node])
	# print("STATHUM")

	saver.restore(sess, ckpt_path)