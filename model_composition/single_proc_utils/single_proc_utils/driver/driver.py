import json
from datetime import datetime

class DriverBase:

	def __init__(self):
		pass

	def run(self, inputs):
		pass

	def benchmark(self, batch_size=1, avg_after=5, log_intermediate=False, **kwargs):
		pass

	def _log(self, msg, allow=False):
		if allow:
			print(msg)

	def _benchmark_model_step(self, fn, inputs):
		begin = datetime.now()
		inputs_length = len(inputs)
		outputs = fn(inputs)
		end = datetime.now()
		latency_seconds = (end - begin).total_seconds()
		throughput = float(inputs_length) / latency_seconds
		return (latency_seconds, throughput, outputs)

	def _load_gpu_config(self, config_path):
		config_file = open(config_path, "rb")
		config_json = json.load(config_file)
		return config_json