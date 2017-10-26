import time
import numpy as np
import pickle

from io import BytesIO
from PIL import Image
from datetime import datetime

def benchmark_function(fn, input_gen_fn, batch_size, avg_after):
	iter_num = 0
	avg_latency = 0
	avg_throughput = 0
	while True:
		inputs = input_gen_fn(batch_size)
		begin = datetime.now()
		fn(inputs)
		end = datetime.now()

		latency_seconds = (end - begin).total_seconds()
		throughput = float(batch_size) / latency_seconds
		print("Latency: {} ms, Throughput: {} qps".format(latency_seconds * 1000, throughput))
		print("---------------------\n")

		avg_latency += latency_seconds
		avg_throughput += throughput

		if iter_num != 0 and (iter_num + 1) % avg_after == 0:
			avg_latency = float(avg_latency) / avg_after
			avg_throughput = avg_throughput / avg_after

			print("AVERAGE FOR {} RUNS - Latency: {} ms, Throughput: {} qps".format(avg_after, avg_latency * 1000, avg_throughput))
			print("---------------------\n")

			avg_latency = 0
			avg_throughput = 0

		iter_num += 1

		time.sleep(.1)

def gen_vgg_featurization_inputs(num_inputs):
	return [np.random.rand(224, 224, 3) * 255 for i in range(0, num_inputs)]

def gen_inception_featurization_inputs(num_inputs):
	input_imgs = [np.random.rand(299,299,3) * 255 for i in range(0, num_inputs)]
	input_imgs = [Image.fromarray(input_img.astype(np.uint8)) for input_img in input_imgs]
	inception_inputs = []
	for input_img in input_imgs:
		inmem_inception_jpeg = BytesIO()
		resized_inception = input_img.resize((299,299)).convert('RGB')
		resized_inception.save(inmem_inception_jpeg, format="JPEG")
		inmem_inception_jpeg.seek(0)
		inception_input = inmem_inception_jpeg.read()
		inception_inputs.append(inception_input)
	return inception_inputs

def gen_inception_classification_inputs(num_inputs):
	return [np.random.rand(299,299,3) * 255 for i in range(0, num_inputs)]

def gen_vgg_svm_classification_inputs(num_inputs):
	return np.array([np.random.rand(1,4096) * 10 for i in range(0, num_inputs)])

def gen_lgbm_classification_inputs(num_inputs):
	return [np.random.rand(1,2048) * 10 for i in range(0, num_inputs)]

def gen_opencv_svm_classification_inputs(num_inputs):
	return (np.random.rand(num_inputs, 128 * 20) * 150).astype(np.int32)

def gen_opencv_featurization_inputs(num_inputs):
	inputs = [np.random.rand(299, 299, 3) * 255 for i in range(0, num_inputs)]
	parsed_inputs = []
	for input_img in inputs:
		img = Image.fromarray(input_img.astype(np.uint8))
		img.resize((299, 299)).convert('RGB')
		rgb_input = np.array(img).astype(np.float32)
		parsed_inputs.append(rgb_input.astype(np.uint8))
	return parsed_inputs

def gen_text_inputs(num_inputs, text_segments_path, lengths_dist_path):
		text_segs_file = open(text_segments_path, "rb")
		text_segs_list = pickle.load(text_segs_file)
		num_segs = len(text_segs_list)
		text_segs_file.close()

		lengths_dist_file = open(lengths_dist_path, "rb")
		lengths_dist = pickle.load(lengths_dist_file)
		lengths_dist_size = len(lengths_dist)
		lengths_dist_file.close()

		def gen_text(length):
			index = np.random.randint(num_segs)
			text_length = 0
			text = u''
			while text_length < length:
				seg = text_segs_list[index]
				text_length += len(seg) + 1
				text = "{} {}".format(text, seg)
				index = (index + 1) % num_segs
			return text

		return [gen_text(lengths_dist[np.random.randint(lengths_dist_size)]) for i in range(0, num_inputs)]