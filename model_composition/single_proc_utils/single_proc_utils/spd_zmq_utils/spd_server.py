import sys
import os
import zmq
import logging
import numpy as np

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

SOCKET_POLLING_TIMEOUT_MILLIS = 5000

INPUT_DTYPE = np.dtype(np.float32)
MB = 1024 * 1024
INITIAL_INPUT_BUFFER_SIZE = 60 * MB 

class SpdFrontend:
    
    def predict(self, inputs, msg_ids):
        """
        Parameters
        ----------
        inputs : np.ndarray
            A numpy array of dimension (batch_size, FLATTENED_INPUT_SIZE),
            where each flattened input is a collection of scalars represented
            as numpy scalars of type INPUT_DTYPE (see above)

        msg_ids : np.ndarray 
            An array of message ids. Each id is represented as an unsigned, 32-bit integer

        Returns
        ----------
        np.ndarray
            An array of output ids. Each id is represented as an unsigned, 32-bit integer
        """
        pass

class SpdServer:

    def __init__(self, spd_frontend, host, port):
        self.spd_frontend = spd_frontend
        self.host = host
        self.port = port

        self.server = None

    def start(self):
        try:
            address = "tcp://127.0.0.1:{port}".format(port=self.port)
            context = zmq.Context()
            socket = context.socket(zmq.DEALER)
            socket.bind(address)
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            
            logger.info("Started model server at address: {}".format(address))

            input_content_buffer = bytearray(INITIAL_INPUT_BUFFER_SIZE)
            while True:
                receivable_sockets = dict(poller.poll(timeout=SOCKET_POLLING_TIMEOUT_MILLIS))
                if socket in receivable_sockets and receivable_sockets[socket] == zmq.POLLIN:
                    t0 = datetime.now()

                    # Receive delimiter between routing identity and content
                    socket.recv()
                    msg_ids = socket.recv()
                    parsed_msg_ids = np.frombuffer(msg_ids, dtype=np.uint32)

                    input_header = socket.recv()
                    parsed_input_header = np.frombuffer(input_header, dtype=np.uint32)
                    batch_size, input_sizes_bytes = parsed_input_header[0], parsed_input_header[1:]
                  
                    input_content_size = sum(input_sizes_bytes) 
                    if len(input_content_buffer) < input_content_size:
                        input_content_buffer = bytearray(len(input_content_buffer) * 2)
                    input_content_view = memoryview(input_content_buffer)[:input_content_size]

                    # We assume a fixed input datatype of np.float32
                    input_dtype = INPUT_DTYPE
                    typed_content_size = input_content_size / INPUT_DTYPE.itemsize

                    buffer_idx = 0
                    for _ in range(batch_size):
                        input_item = socket.recv(copy=False)
                        input_item_view = memoryview(input_item)
                        input_content_view[buffer_idx : buffer_idx + len(input_item_view)] = input_item_view
                        buffer_idx += len(input_item_view)

                    if len(set(input_sizes_bytes)) == 1:
                        # All inputs are of the same size, proceed to reshape
                        inputs = np.frombuffer(input_content_buffer, dtype=input_dtype)[:typed_content_size]
                        inputs = np.reshape(inputs, (batch_size, -1))
                    else:
                        logger.info("Received inputs of non-uniform length")
                        raise

                    
                    t1 = datetime.now()

                    # A list of output_ids, represented as a numpy array
                    # of unsigned, 32-bit integers
                    output_ids = self.spd_frontend.predict(inputs, parsed_msg_ids)

                    t2 = datetime.now()

                    # Send the empty delimeter required at the start of a new message
                    socket.send("", zmq.SNDMORE)

                    socket.send(memoryview(output_ids.view(np.uint8)), copy=False, flags=0)

                    t3 = datetime.now()

                    print((t3 - t2).total_seconds(), (t2 - t1).total_seconds(), (t1 - t0).total_seconds())

        except KeyboardInterrupt:
            os._exit()
