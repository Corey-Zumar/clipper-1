from __future__ import print_function

import sys
import grpc
import logging
import numpy as np
import time

from datetime import datetime
from multiprocessing import Process

from clipper_zmq_client import Client

DATA_TYPE_BYTES = 0
DATA_TYPE_INTS = 1
DATA_TYPE_FLOATS = 2
DATA_TYPE_DOUBLES = 3
DATA_TYPE_STRINGS = 4

VGG_FEATURE_VEC_SIZE = 4096

input_type = "doubles"
app_name = "app1"
model_name = "m1"

def run(proc_num):
	"""
	Note: Throughput logging is performed by the ZMQ Frontend Client
	(clipper_zmq_client.py)
	"""
	client = Client("localhost", 4456, 4455)
	client.start()
	num_events = 0
	begin = datetime.now()
	while True:
		future = client.send_request(app_name, np.array(np.random.rand(VGG_FEATURE_VEC_SIZE), dtype=np.float32))
		future.get()
		num_events += 1

		if num_events % 20 == 0:
			end = datetime.now()
			print("THROUGHPUT: {} qps".format(num_events / (end - begin).total_seconds()))
			begin = end
			num_events = 0

if __name__ == "__main__":
	if len(sys.argv) < 2:
		raise

	num_procs = int(sys.argv[1])

	processes = []

	for i in range(num_procs):
		p = Process(target=run, args=('%d'.format(i),))
		p.start()
		processes.append(p)

	for p in processes:
		p.join()