from __future__ import print_function
import zmq
import threading
import numpy as np
import struct
import time
from datetime import datetime
import socket
import sys
from collections import deque

from threading import Thread
from Queue import Queue

DATA_TYPE_BYTES = 0
DATA_TYPE_INTS = 1
DATA_TYPE_FLOATS = 2
DATA_TYPE_DOUBLES = 3
DATA_TYPE_STRINGS = 4

REQUEST_TYPE_PREDICT = 0
REQUEST_TYPE_FEEDBACK = 1

MESSAGE_TYPE_NEW_CONTAINER = 0
MESSAGE_TYPE_CONTAINER_CONTENT = 1
MESSAGE_TYPE_HEARTBEAT = 2

HEARTBEAT_TYPE_KEEPALIVE = 0
HEARTBEAT_TYPE_REQUEST_CONTAINER_METADATA = 1

SOCKET_POLLING_TIMEOUT_MILLIS = 5000
SOCKET_ACTIVITY_TIMEOUT_MILLIS = 30000

EVENT_HISTORY_BUFFER_SIZE = 30

EVENT_HISTORY_SENT_HEARTBEAT = 1
EVENT_HISTORY_RECEIVED_HEARTBEAT = 2
EVENT_HISTORY_SENT_CONTAINER_METADATA = 3
EVENT_HISTORY_RECEIVED_CONTAINER_METADATA = 4
EVENT_HISTORY_SENT_CONTAINER_CONTENT = 5
EVENT_HISTORY_RECEIVED_CONTAINER_CONTENT = 6

BYTES_PER_INT = 4
BYTES_PER_FLOAT = 4
BYTES_PER_BYTE = 1
BYTES_PER_CHAR = 1
BYTES_PER_LONG = 8

# Initial size of the buffers used for sending response
# header data and receiving request header data
INITIAL_HEADER_BUFFER_SIZE = 1024

# A mapping from python output data types
# to their corresponding clipper data types for serialization
SUPPORTED_OUTPUT_TYPES_MAPPING = {
    np.dtype(np.uint8): DATA_TYPE_BYTES,
    np.dtype(np.int32): DATA_TYPE_INTS,
    np.dtype(np.float32): DATA_TYPE_FLOATS,
    str: DATA_TYPE_STRINGS,
}

def input_type_to_dtype(input_type):
    if input_type == DATA_TYPE_BYTES:
        return np.int8
    elif input_type == DATA_TYPE_INTS:
        return np.int32
    elif input_type == DATA_TYPE_FLOATS:
        return np.float32
    elif input_type == DATA_TYPE_DOUBLES:
        return np.float64
    elif input_type == DATA_TYPE_STRINGS:
        return np.str_

def input_type_to_string(input_type):
    if input_type == DATA_TYPE_BYTES:
        return "bytes"
    elif input_type == DATA_TYPE_INTS:
        return "ints"
    elif input_type == DATA_TYPE_FLOATS:
        return "floats"
    elif input_type == DATA_TYPE_DOUBLES:
        return "doubles"
    elif input_type == DATA_TYPE_STRINGS:
        return "string"


class EventHistory:
    def __init__(self, size):
        self.history_buffer = deque(maxlen=size)

    def insert(self, msg_type):
        curr_time_millis = time.time() * 1000
        self.history_buffer.append((curr_time_millis, msg_type))

    def get_events(self):
        return self.history_buffer


class PredictionError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def handle_predictions(predict_fn, request_queue, response_queue):
    """
    Returns
    -------
    PredictionResponse
        A prediction response containing an output
        for each input included in the specified
        predict response
    """
    t2 = datetime.now()

    while True:
        prediction_request = request_queue.get(block=True)
        t1 = datetime.now()

        print("batch_delay: {} us".format((t1 - t2).total_seconds()))

        outputs = predict_fn(prediction_request.inputs)
        # Type check the outputs:
        if not type(outputs) == dict:
            raise PredictionError("Model did not return a dict. Instead, returned an object of type: {}".format(type(outputs)))

        response = PredictionResponse(prediction_request.msg_id)

        for model_name in outputs:
            if model_name not in outputs:
                raise PredictionError(
                        "Prediction function did not produce any outputs for model: {}".format(model_name))

            model_outputs = outputs[model_name]

            if len(model_outputs) != len(prediction_request.inputs[model_name]):
                raise PredictionError(
                    "Expected prediction function to return %d outputs for model %s, found %d outputs" %
                    (model_name, len(prediction_request.inputs), len(model_outputs)))

            outputs_type = type(model_outputs[0])
            if outputs_type == np.ndarray:
                outputs_type = outputs[0].dtype
            if outputs_type not in SUPPORTED_OUTPUT_TYPES_MAPPING.keys():
                raise PredictionError(
                    "Outputs list for model {} contains outputs of invalid type: {}!".
                    format(model_name, outputs_type))

            outputs_type = SUPPORTED_OUTPUT_TYPES_MAPPING[outputs_type]

            if outputs_type == DATA_TYPE_STRINGS:
                for i in range(0, len(model_outputs)):
                    model_outputs[i] = unicode(model_outputs[i], "utf-8").encode("utf-8")
            else:
                for i in range(0, len(model_outputs)):
                    model_outputs[i] = model_outputs[i].tobytes()

            for i in range(len(model_outputs)):
                batch_id = prediction_request.model_batch_ids[model_name][i]
                response.add_output(model_outputs[i], outputs_type, batch_id) 

        response_queue.put(response)

        t2 = datetime.now()
        print("handle: {pred_time} us".format(pred_time=(t2 - t1).total_seconds()))
        sys.stdout.flush()
        sys.stderr.flush()




class Server(threading.Thread):
    def __init__(self, context, clipper_ip, send_port, recv_port):
        threading.Thread.__init__(self)
        self.context = context
        self.clipper_ip = clipper_ip
        self.send_port = send_port
        self.recv_port = recv_port
        self.event_history = EventHistory(EVENT_HISTORY_BUFFER_SIZE)
        self.full_buffers = 0
        self.request_queue = Queue()
        self.response_queue = Queue()

    def connect(self):
        # 7000
        recv_address = "tcp://{0}:{1}".format(self.clipper_ip,
                                                 self.recv_port)
        # 7001
        send_address = "tcp://{0}:{1}".format(self.clipper_ip,
                                                 self.send_port)

        self.context = zmq.Context()
        self.recv_socket = self.context.socket(zmq.DEALER)
        self.recv_poller = zmq.Poller()
        self.recv_poller.register(self.recv_socket, zmq.POLLIN)

        print("Sending first connection message")
        sys.stdout.flush()
        sys.stderr.flush()

        self.recv_socket.connect(recv_address)
        # Send a blank message to establish a connection
        # self.recv_socket.send("", zmq.SNDMORE)
        # self.recv_socket.send("")


        # Now send container metadata to establish a connection
        num_models = len(self.model_info)

        self.recv_socket.send("", zmq.SNDMORE)
        self.recv_socket.send(struct.pack("<I", MESSAGE_TYPE_NEW_CONTAINER), zmq.SNDMORE)
        self.recv_socket.send(struct.pack("<I", num_models), zmq.SNDMORE)
        idx = 0
        for idx in range(num_models):
            model_name, model_version = self.model_info[idx]
            self.recv_socket.send_string(model_name, zmq.SNDMORE)
            if idx < num_models - 1:
                self.recv_socket.send_string(str(model_version), zmq.SNDMORE)
            else:
                self.recv_socket.send_string(str(model_version), 0)

        print("Sent container metadata!")
        sys.stdout.flush()
        sys.stderr.flush()


        receivable_sockets = dict(self.recv_poller.poll(None))
        if not(self.recv_socket in receivable_sockets and receivable_sockets[self.recv_socket] == zmq.POLLIN):
            raise RuntimeError

        self.recv_socket.recv()
        connection_id_bytes = self.recv_socket.recv()
        self.connection_id = struct.unpack("<I", connection_id_bytes)[0]

        print("Assigned connection ID: {}".format(self.connection_id))
        sys.stdout.flush()
        sys.stderr.flush()
        self.send_socket = self.context.socket(zmq.DEALER)
        self.send_socket.connect(send_address)

    def get_prediction_function(self):
        return self.predictor.predict

    def get_event_history(self):
        return self.event_history.get_events()

    def send_response(self):
        if not self.response_queue.empty() or self.full_buffers == 1:
            response = self.response_queue.get()
            self.full_buffers -= 1
            # t3 = datetime.now()
            response.send(self.send_socket, self.connection_id)
            sys.stdout.flush()
            sys.stderr.flush()

    def recv_request(self):
        self.recv_socket.recv()
        msg_type_bytes = self.recv_socket.recv()
        msg_type = struct.unpack("<I", msg_type_bytes)[0]
        if msg_type is not MESSAGE_TYPE_CONTAINER_CONTENT:
            raise RuntimeError("Wrong message type: {}".format(msg_type))
        self.event_history.insert(
            EVENT_HISTORY_RECEIVED_CONTAINER_CONTENT)
        msg_id_bytes = self.recv_socket.recv()
        msg_id = int(struct.unpack("<I", msg_id_bytes)[0])

        print("Got start of message %d " % msg_id)
        # list of byte arrays
        request_header = self.recv_socket.recv()
        request_type = struct.unpack("<I", request_header)[0]

        if request_type == REQUEST_TYPE_PREDICT:
            input_header_size = self.recv_socket.recv()
            input_header = self.recv_socket.recv()
            parsed_input_header = np.frombuffer(input_header, dtype=np.uint32)
            [
                num_inputs,
                input_metadata_items
            ] = [
                parsed_input_header[0],
                parsed_input_header[1:]
            ]

            parsed_input_types = [input_type_to_dtype([(3 * i) + 1]) for i in range(num_inputs)]

            inputs = {}
            model_batch_ids = {}
            for i in range(num_inputs):
                model_name = self.recv_socket.recv_string()
                input_item = self.recv_socket.recv()
                input_item = np.frombuffer(input_item, dtype=parsed_input_types[i])
                
                if model_name not in inputs:
                    inputs[model_name] = [input_item]
                else:
                    inputs[model_name].append(input_item)

                model_batch_id = input_metadata_items[(3 * i) + 0]
                if model_name not in model_batch_ids:
                    model_batch_ids[model_name] = [model_batch_id]
                else:
                    model_batch_ids[model_name].append(model_batch_id)

            t2 = datetime.now()

            prediction_request = PredictionRequest(
                msg_id_bytes, inputs, model_batch_ids)

            self.request_queue.put(prediction_request)
            self.full_buffers += 1

    def run(self):
        def target_fn():
            try:
                handle_predictions(self.get_prediction_function(),
                                   self.request_queue,
                                   self.response_queue)
            except Exception as e:
                print(e)

        self.handler_thread = Thread(target=target_fn)
        self.handler_thread.start()
        print("Serving predictions...")
        self.connect()
        print("Connected")
        sys.stdout.flush()
        sys.stderr.flush()
        while True:
            receivable_sockets = dict(self.recv_poller.poll(SOCKET_POLLING_TIMEOUT_MILLIS))
            if self.recv_socket in receivable_sockets and receivable_sockets[self.recv_socket] == zmq.POLLIN:
                self.recv_request()

            self.send_response()
            sys.stdout.flush()
            sys.stderr.flush()



class PredictionRequest:
    """
    Parameters
    ----------
    msg_id : bytes
        The raw message id associated with the RPC
        prediction request message
    inputs :
        One of [[byte]], [[int]], [[float]], [[double]], [string]
    """

    def __init__(self, msg_id, inputs, model_batch_ids):
        self.msg_id = msg_id
        self.inputs = inputs
        self.model_batch_ids = model_batch_ids

    def __str__(self):
        return self.inputs

class PredictionResponse:
    header_buffer = bytearray(INITIAL_HEADER_BUFFER_SIZE)

    def __init__(self, msg_id):
        """
        Parameters
        ----------
        msg_id : bytes
            The message id associated with the PredictRequest
            for which this is a response
        """
        self.msg_id = msg_id
        self.outputs = []
        self.num_outputs = 0
 
    def add_output(self, output, output_type, batch_id):
        """
        Parameters
        ----------
        output : string
        batch_id : int
        """
        output = unicode(output, "utf-8").encode("utf-8")
        self.outputs.append((output, output_type, batch_id))
        self.num_outputs += 1

    def send(self, socket, connection_id):
        """
        Sends the encapsulated response data via
        the specified socket

        Parameters
        ----------
        socket : zmq.Socket
        """
        assert self.num_outputs > 0
        output_header, header_length_bytes = self._create_output_header()
        socket.send("", flags=zmq.SNDMORE)
        socket.send(struct.pack("<I", connection_id), flags=zmq.SNDMORE)
        socket.send(
            struct.pack("<I", MESSAGE_TYPE_CONTAINER_CONTENT),
            flags=zmq.SNDMORE)
        socket.send(self.msg_id, flags=zmq.SNDMORE)
        socket.send(struct.pack("<Q", header_length_bytes), flags=zmq.SNDMORE)
        socket.send(output_header, flags=zmq.SNDMORE)
        for idx in range(self.num_outputs):
            output_item = self.outputs[idx][0]
            if idx == self.num_outputs - 1:
                # Don't use the `SNDMORE` flag if
                # this is the last output being sent
                socket.send_string(output_item)
            else:
                socket.send_string(output_item, flags=zmq.SNDMORE)

    def _expand_buffer_if_necessary(self, size):
        """
        If necessary, expands the reusable output
        header buffer to accomodate content of the
        specified size

        size : int
            The size, in bytes, that the buffer must be
            able to store
        """
        if len(PredictionResponse.header_buffer) < size:
            PredictionResponse.header_buffer = bytearray(size * 2)

    def _create_output_header(self):
        """
        Returns
        ----------
        (bytearray, int)
            A tuple with the output header as the first
            element and the header length as the second
            element
        """
        header_length = BYTES_PER_INT * ((3 * len(self.outputs)) + 1)
        self._expand_buffer_if_necessary(header_length)
        header_idx = 0
        struct.pack_into("<I", PredictionResponse.header_buffer, header_idx,
                         self.num_outputs)
        header_idx += BYTES_PER_INT
        for output, output_type, batch_id in self.outputs:
            struct.pack_into("<I", PredictionResponse.header_buffer,
                             header_idx, len(output))
            header_idx += BYTES_PER_INT 
            struct.pack_into("<I", PredictionResponse.header_buffer,
                             header_idx, output_type)
            header_idx += BYTES_PER_INT
            struct.pack_into("<I", PredictionResponse.header_buffer,
                             header_idx, batch_id)
            header_idx += BYTES_PER_INT

        return PredictionResponse.header_buffer[:header_length], header_length

class FeedbackRequest():
    def __init__(self, msg_id, content):
        self.msg_id = msg_id
        self.content = content

    def __str__(self):
        return self.content


class FeedbackResponse():
    def __init__(self, msg_id, content):
        self.msg_id = msg_id
        self.content = content

    def send(self, socket):
        socket.send("", flags=zmq.SNDMORE)
        socket.send(
            struct.pack("<I", MESSAGE_TYPE_CONTAINER_CONTENT),
            flags=zmq.SNDMORE)
        socket.send(self.msg_id, flags=zmq.SNDMORE)
        socket.send(self.content)


class ModelContainerBase(object):
    def predict(self, inputs):
        """
        Parameters
        ------------
        inputs : dict
            A mapping between model names and batched prediction data

        Returns
        ------------
        dict 
            A mapping between model names and model prediction outputs
        """
        pass

class RPCService:
    def __init__(self):
        pass

    def get_event_history(self):
        if self.server:
            return self.server.get_event_history()
        else:
            print("Cannot retrieve message history for inactive RPC service!")
            raise

    def start(self, predictor, host, model_info):
        """
        Args:
            predictor (object): A predictor conforming to the Clipper prediction interface.
            host (str): The Clipper RPC hostname or IP address.
            port (int): The Clipper RPC port.
            model_info ([str, int]): A list of models served by this container
                of the form (<model_name>, <model_version>)
        """

        recv_port = 7010
        send_port = 7011

        try:
            ip = socket.gethostbyname(host)
        except socket.error as e:
            print("Error resolving %s: %s" % (host, e))
            sys.exit(1)
        context = zmq.Context()
        self.server = Server(context, ip, send_port, recv_port)
        self.server.predictor = predictor
        self.server.model_info = model_info
        self.server.run()







