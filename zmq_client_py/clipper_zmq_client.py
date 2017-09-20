import zmq
import numpy as np
import struct
import time
from datetime import datetime
import socket
import sys

from threading import Lock, Thread
from Queue import Queue

DATA_TYPE_BYTES = 0
DATA_TYPE_INTS = 1
DATA_TYPE_FLOATS = 2
DATA_TYPE_DOUBLES = 3
DATA_TYPE_STRINGS = 4

NUM_RESPONSES_RECV = 10

class Client:

	active = False

	def __init__(self, clipper_host, clipper_send_port, clipper_recv_port):
		"""
		Parameters
		----------
		clipper_host : str
			The address of an active clipper host
		clipper_send_port : int
			The clipper port to which we will send requests
		clipper_recv_port : int
			The clipper port from which we will receive responses		
		"""
		self.clipper_host = clipper_host
		self.send_port = clipper_send_port
		self.recv_port = clipper_recv_port
		self.client_id = None
		self.request_id = 0
		self.outstanding_requests = {}
		self.request_lock = Lock()
		self.request_queue = Queue()

	def start(self):
		global active
		active = True
		self.recv_thread = Thread(target=self._run_recv, args=[])
		self.recv_thread.start()
		self.send_thread = Thread(target=self._run_send, args=[])
		self.send_thread.start()

	def stop(self):
		global active
		if active:
			active = False
			self.recv_thread.join()
			self.send_thread.join()

	def send_request(self, app_name, input_item, callback=None):
		self.request_lock.acquire()
		self.outstanding_requests[self.request_id] = callback
		self.request_queue.put((self.request_id, app_name, input_item))
		self.request_id += 1
		self.request_lock.release()

	def _run_recv(self):
		global active
		# The address of the socket from which we want to receive data
		clipper_recv_address = "tcp://{0}:{1}".format(self.clipper_host, self.recv_port)
		context = zmq.Context()
		socket = context.socket(zmq.DEALER)
		poller = zmq.Poller()
		poller.register(socket, zmq.POLLIN)

		socket.connect(clipper_recv_address)
		# Send a blank message to establish a connection
		socket.send("", zmq.SNDMORE)
		socket.send("")
		connected = False
		while active:
			timeout = 1000 if connected else 5000
			receivable_sockets = dict(poller.poll(timeout))
			if socket in receivable_sockets and receivable_sockets[socket] == zmq.POLLIN:
				if connected:
					self._receive_response(socket)
				else:
					self._handle_new_connection(socket)
					connected = True

	def _run_send(self):
		global active
		# The address of the socket to which we want to send data
		clipper_send_address = "tcp://{0}:{1}".format(self.clipper_host, self.send_port)
		context = zmq.Context()
		socket = context.socket(zmq.DEALER)
		poller = zmq.Poller()
		poller.register(socket, zmq.POLLIN)

		socket.connect(clipper_send_address)
		while active:
			if self.client_id is None:
				time.sleep(.001)
			else:
				self._send_requests(socket)

	def _handle_new_connection(self, socket):
		# Receive delimiter between routing identity and content
		socket.recv()
		client_id_bytes = socket.recv()
		self.client_id = struct.unpack("<I", client_id_bytes)[0]

	def _receive_response(self, socket):
		# Receive delimiter between routing identity and content
		socket.recv()
		request_id_bytes = socket.recv()
		data_type_bytes = socket.recv()
		output_data = socket.recv()

		request_id = struct.unpack("<I", request_id_bytes)
		data_type = struct.unpack("<I", data_type_bytes)
		output = np.frombuffer(output_data, dtype=self._clipper_type_to_dtype(data_type))

		self.request_lock.acquire()
		if request_id in self.outstanding_requests.keys() and self.outstanding_requests[request_id] is not None:
			self.outstanding_requests[request_id](output)
			del self.outstanding_requests[request_id]
		self.request_lock.release()

	def _send_requests(self, socket):
		if self.request_queue.empty():
			time.sleep(.001)

		while not self.request_queue.empty():
			app_name, input_item = self.request_queue.get()
			socket.send("", zmq.SNDMORE)
			socket.send(struct.pack("<I", self.client_id), zmq.SNDMORE)
			socket.send_string(app_name, zmq.SNDMORE)
			socket.send(struct.pack("<I", DATA_TYPE_DOUBLES), zmq.SNDMORE)
			socket.send(struct.pack("<I", len(input_item)), zmq.SNDMORE)
			socket.send(input_item)

	def _clipper_type_to_dtype(self, cl_type):
		if cl_type == DATA_TYPE_BYTES:
			return np.int8
		elif cl_type == DATA_TYPE_INTS:
			return np.int32
		elif cl_type == DATA_TYPE_FLOATS:
			return np.float32
		elif cl_type == DATA_TYPE_DOUBLES:
			return np.float64
		elif cl_type == DATA_TYPE_STRINGS:
			return np.str_
