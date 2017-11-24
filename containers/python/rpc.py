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

# A mapping from python output data types
# to their corresponding clipper data types for serialization
SUPPORTED_OUTPUT_TYPES_MAPPING = {
    np.dtype(np.uint8): DATA_TYPE_BYTES,
    np.dtype(np.int32): DATA_TYPE_INTS,
    np.dtype(np.float32): DATA_TYPE_FLOATS,
    str: DATA_TYPE_STRINGS,
}


def string_to_input_type(input_str):
    input_str = input_str.strip().lower()
    byte_strs = ["b", "bytes", "byte"]
    int_strs = ["i", "ints", "int", "integer", "integers"]
    float_strs = ["f", "floats", "float"]
    double_strs = ["d", "doubles", "double"]
    string_strs = ["s", "strings", "string", "strs", "str"]

    if any(input_str == s for s in byte_strs):
        return DATA_TYPE_BYTES
    elif any(input_str == s for s in int_strs):
        return DATA_TYPE_INTS
    elif any(input_str == s for s in float_strs):
        return DATA_TYPE_FLOATS
    elif any(input_str == s for s in double_strs):
        return DATA_TYPE_DOUBLES
    elif any(input_str == s for s in string_strs):
        return DATA_TYPE_STRINGS
    else:
        return -1


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

    while True:
        prediction_request = request_queue.get(block=True)
        t1 = datetime.now()

        outputs = predict_fn(prediction_request.inputs)
        # Type check the outputs:
        if not type(outputs) == list:
            raise PredictionError("Model did not return a list")
        if len(outputs) != len(prediction_request.inputs):
            raise PredictionError(
                "Expected model to return %d outputs, found %d outputs" %
                (len(prediction_request.inputs), len(outputs)))

        outputs_type = type(outputs[0])
        if outputs_type == np.ndarray:
            outputs_type = outputs[0].dtype
        if not outputs_type in SUPPORTED_OUTPUT_TYPES_MAPPING.keys():
            raise PredictionError(
                "Model outputs list contains outputs of invalid type: {}!".
                format(outputs_type))

        if outputs_type == str:
            for i in range(0, len(outputs)):
                outputs[i] = unicode(outputs[i], "utf-8").encode("utf-8")
        else:
            for i in range(0, len(outputs)):
                outputs[i] = outputs[i].tobytes()

        total_length_elements = sum(len(o) for o in outputs)

        response = PredictionResponse(prediction_request.msg_id,
                                        len(outputs), total_length_elements,
                                        outputs_type)
        for output in outputs:
            response.add_output(output)

        response_queue.put(response)

        t2 = datetime.now()
        print("handle: {pred_time} us".format((t2 - t1).microseconds))




class Server(threading.Thread):
    def __init__(self, context, clipper_ip, clipper_port):
        threading.Thread.__init__(self)
        self.context = context
        self.clipper_ip = clipper_ip
        self.clipper_port = clipper_port
        self.event_history = EventHistory(EVENT_HISTORY_BUFFER_SIZE)
        self.full_buffers = 0
        self.request_queue = Queue()
        self.response_queue = Queue()


    def handle_feedback_request(self, feedback_request):
        """
        Returns
        -------
        FeedbackResponse
            A feedback response corresponding
            to the specified feedback request
        """
        response = FeedbackResponse(feedback_request.msg_id, "ACK")
        return response

    def get_prediction_function(self):
        if self.model_input_type == DATA_TYPE_INTS:
            return self.model.predict_ints
        elif self.model_input_type == DATA_TYPE_FLOATS:
            return self.model.predict_floats
        elif self.model_input_type == DATA_TYPE_DOUBLES:
            return self.model.predict_doubles
        elif self.model_input_type == DATA_TYPE_BYTES:
            return self.model.predict_bytes
        elif self.model_input_type == DATA_TYPE_STRINGS:
            return self.model.predict_strings
        else:
            print(
                "Attempted to get predict function for invalid model input type!"
            )
            raise

    def get_event_history(self):
        return self.event_history.get_events()

    def run(self):
        self.handler_thread = Thread(target=handle_predictions,
                                     args=(self.get_prediction_function(),
                                           self.request_queue,
                                           self.response_queue))
        self.handler_thread.start()

        print("Serving predictions for {0} input type.".format(
            input_type_to_string(self.model_input_type)))
        connected = False
        clipper_address = "tcp://{0}:{1}".format(self.clipper_ip,
                                                 self.clipper_port)
        poller = zmq.Poller()
        sys.stdout.flush()
        sys.stderr.flush()
        while True:
            socket = self.context.socket(zmq.DEALER)
            poller.register(socket, zmq.POLLIN)
            socket.connect(clipper_address)
            self.send_heartbeat(socket)
            while True:
                receivable_sockets = dict(
                    poller.poll(SOCKET_POLLING_TIMEOUT_MILLIS))
                if socket not in receivable_sockets or receivable_sockets[socket] != zmq.POLLIN:
                    # Failed to receive a message before the specified polling timeout
                    if connected:
                        curr_time = datetime.now()
                        time_delta = curr_time - last_activity_time_millis
                        time_delta_millis = (time_delta.seconds * 1000) + (
                            time_delta.microseconds / 1000)
                        if time_delta_millis >= SOCKET_ACTIVITY_TIMEOUT_MILLIS:
                            # Terminate the session
                            print("Connection timed out, reconnecting...")
                            sys.stdout.flush()
                            sys.stderr.flush()
                            connected = False
                            poller.unregister(socket)
                            socket.close()
                            break
                        else:
                            self.send_heartbeat(socket)
                    continue

                # Received a message before the polling timeout
                if not connected:
                    connected = True
                last_activity_time_millis = datetime.now()

                t1 = datetime.now()
                # Receive delimiter between routing identity and content
                socket.recv()
                msg_type_bytes = socket.recv()
                msg_type = struct.unpack("<I", msg_type_bytes)[0]
                if msg_type == MESSAGE_TYPE_HEARTBEAT:
                    self.event_history.insert(EVENT_HISTORY_RECEIVED_HEARTBEAT)
                    print("Received heartbeat!")
                    heartbeat_type_bytes = socket.recv()
                    heartbeat_type = struct.unpack("<I",
                                                   heartbeat_type_bytes)[0]
                    if heartbeat_type == HEARTBEAT_TYPE_REQUEST_CONTAINER_METADATA:
                        self.send_container_metadata(socket)
                    continue
                elif msg_type == MESSAGE_TYPE_NEW_CONTAINER:
                    self.event_history.insert(
                        EVENT_HISTORY_RECEIVED_CONTAINER_METADATA)
                    print(
                        "Received erroneous new container message from Clipper!"
                    )
                    continue
                elif msg_type == MESSAGE_TYPE_CONTAINER_CONTENT:
                    self.event_history.insert(
                        EVENT_HISTORY_RECEIVED_CONTAINER_CONTENT)
                    msg_id_bytes = socket.recv()
                    msg_id = int(struct.unpack("<I", msg_id_bytes)[0])

                    print("Got start of message %d " % msg_id)
                    # list of byte arrays
                    request_header = socket.recv()
                    request_type = struct.unpack("<I", request_header)[0]

                    if request_type == REQUEST_TYPE_PREDICT:
                        input_header_size = socket.recv()
                        input_header = socket.recv()
                        parsed_input_header = np.frombuffer(input_header, dtype=np.uint32)
                        [
                            input_type,
                            num_inputs,
                            input_sizes
                        ] = [
                            parsed_input_header[0],
                            parsed_input_header[1],
                            parsed_input_header[2:]
                        ]

                        if int(input_type) != int(self.model_input_type):
                            print((
                                "Received incorrect input. Expected {expected}, "
                                "received {received}").format(
                                    expected=input_type_to_string(
                                        int(self.model_input_type)),
                                    received=input_type_to_string(
                                        int(input_type))))
                            raise

                        inputs = []
                        for _ in range(num_inputs):
                            input_item = socket.recv()
                            input_item = np.frombuffer(input_item, dtype=input_type_to_dtype(input_type))
                            inputs.append(input_item)

                        t2 = datetime.now()

                        prediction_request = PredictionRequest(
                            msg_id_bytes, inputs)

                        self.request_queue.put(prediction_request)
                        self.full_buffers += 1

                        if not self.response_queue.empty() or self.full_buffers == 2:
                            response = self.response_queue.get()
                            self.full_buffers -= 1
                            # t3 = datetime.now()
                            response.send(socket, self.event_history)

                        sys.stdout.flush()
                        sys.stderr.flush()


                sys.stdout.flush()
                sys.stderr.flush()

    def send_container_metadata(self, socket):
        socket.send("", zmq.SNDMORE)
        socket.send(struct.pack("<I", MESSAGE_TYPE_NEW_CONTAINER), zmq.SNDMORE)
        socket.send_string(self.model_name, zmq.SNDMORE)
        socket.send_string(str(self.model_version), zmq.SNDMORE)
        socket.send_string(str(self.model_input_type))
        self.event_history.insert(EVENT_HISTORY_SENT_CONTAINER_METADATA)
        print("Sent container metadata!")

    def send_heartbeat(self, socket):
        socket.send("", zmq.SNDMORE)
        socket.send(struct.pack("<I", MESSAGE_TYPE_HEARTBEAT))
        self.event_history.insert(EVENT_HISTORY_SENT_HEARTBEAT)
        print("Sent heartbeat!")


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

    def __init__(self, msg_id, inputs):
        self.msg_id = msg_id
        self.inputs = inputs

    def __str__(self):
        return self.inputs


class PredictionResponse:
    output_buffer = bytearray(1024)

    def __init__(self, msg_id, num_outputs, content_length, py_output_type):
        """
        Parameters
        ----------
        msg_id : bytes
            The message id associated with the PredictRequest
            for which this is a response
        num_outputs : int
            The number of outputs to be included in the prediction response
        content_length: int
            The total length of all outputs, in bytes
        py_output_type : type
            The python data type of output element content
        """
        self.msg_id = msg_id
        self.num_outputs = num_outputs
        self.output_type = SUPPORTED_OUTPUT_TYPES_MAPPING[py_output_type]
        self.expand_buffer_if_necessary(num_outputs, content_length)

        self.memview = memoryview(self.output_buffer)
        struct.pack_into("<I", self.output_buffer, 0, num_outputs)
        self.content_end_position = BYTES_PER_INT + (
            BYTES_PER_INT * num_outputs)
        self.current_output_sizes_position = BYTES_PER_INT

    def add_output(self, output):
        """
        Parameters
        ----------
        output : str
            A byte-serialized output or utf-8 encoded string
        """
        output_len = len(output)
        struct.pack_into("<I", self.output_buffer,
                         self.current_output_sizes_position, output_len)
        self.current_output_sizes_position += BYTES_PER_INT
        self.memview[self.content_end_position:
                     self.content_end_position + output_len] = output
        self.content_end_position += output_len

    def send(self, socket, event_history):
        socket.send("", flags=zmq.SNDMORE)
        socket.send(
            struct.pack("<I", MESSAGE_TYPE_CONTAINER_CONTENT),
            flags=zmq.SNDMORE)
        socket.send(self.msg_id, flags=zmq.SNDMORE)
        socket.send(struct.pack("<I", self.output_type), flags=zmq.SNDMORE)
        socket.send(
            struct.pack("<I", self.content_end_position), flags=zmq.SNDMORE)
        socket.send(self.output_buffer[0:self.content_end_position])
        event_history.insert(EVENT_HISTORY_SENT_CONTAINER_CONTENT)

    def expand_buffer_if_necessary(self, num_outputs, content_length_bytes):
        new_size_bytes = BYTES_PER_INT + (
            BYTES_PER_INT * num_outputs) + content_length_bytes
        if len(self.output_buffer) < new_size_bytes:
            self.output_buffer = bytearray(new_size_bytes * 2)


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
    def predict_ints(self, inputs):
        pass

    def predict_floats(self, inputs):
        pass

    def predict_doubles(self, inputs):
        pass

    def predict_bytes(self, inputs):
        pass

    def predict_strings(self, inputs):
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

    def start(self, model, host, port, model_name, model_version, input_type):
        """
        Args:
            model (object): The loaded model object ready to make predictions.
            host (str): The Clipper RPC hostname or IP address.
            port (int): The Clipper RPC port.
            model_name (str): The name of the model.
            model_version (int): The version of the model
            input_type (str): One of ints, doubles, floats, bytes, strings.
        """

        try:
            ip = socket.gethostbyname(host)
        except socket.error as e:
            print("Error resolving %s: %s" % (host, e))
            sys.exit(1)
        context = zmq.Context()
        self.server = Server(context, ip, port)
        model_input_type = string_to_input_type(input_type)
        self.server.model_name = model_name
        self.server.model_version = model_version
        self.server.model_input_type = model_input_type
        self.server.model = model
        print("Starting RPC Server. Model: {}:{}, host: {}, port: {}".format(
            model_name, model_version, host, port))
        self.server.run()







