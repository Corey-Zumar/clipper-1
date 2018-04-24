# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: spd_frontend.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='spd_frontend.proto',
  package='clipper.grpc',
  syntax='proto3',
  serialized_pb=_b('\n\x12spd_frontend.proto\x12\x0c\x63lipper.grpc\"1\n\x0ePredictRequest\x12\x0e\n\x06inputs\x18\x01 \x03(\x0c\x12\x0f\n\x07msg_ids\x18\x02 \x03(\x05\"\"\n\x0fPredictResponse\x12\x0f\n\x07msg_ids\x18\x01 \x03(\x05\x32Y\n\x07Predict\x12N\n\rPredictFloats\x12\x1c.clipper.grpc.PredictRequest\x1a\x1d.clipper.grpc.PredictResponse\"\x00\x62\x06proto3')
)




_PREDICTREQUEST = _descriptor.Descriptor(
  name='PredictRequest',
  full_name='clipper.grpc.PredictRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='inputs', full_name='clipper.grpc.PredictRequest.inputs', index=0,
      number=1, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='msg_ids', full_name='clipper.grpc.PredictRequest.msg_ids', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=36,
  serialized_end=85,
)


_PREDICTRESPONSE = _descriptor.Descriptor(
  name='PredictResponse',
  full_name='clipper.grpc.PredictResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg_ids', full_name='clipper.grpc.PredictResponse.msg_ids', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=87,
  serialized_end=121,
)

DESCRIPTOR.message_types_by_name['PredictRequest'] = _PREDICTREQUEST
DESCRIPTOR.message_types_by_name['PredictResponse'] = _PREDICTRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PredictRequest = _reflection.GeneratedProtocolMessageType('PredictRequest', (_message.Message,), dict(
  DESCRIPTOR = _PREDICTREQUEST,
  __module__ = 'spd_frontend_pb2'
  # @@protoc_insertion_point(class_scope:clipper.grpc.PredictRequest)
  ))
_sym_db.RegisterMessage(PredictRequest)

PredictResponse = _reflection.GeneratedProtocolMessageType('PredictResponse', (_message.Message,), dict(
  DESCRIPTOR = _PREDICTRESPONSE,
  __module__ = 'spd_frontend_pb2'
  # @@protoc_insertion_point(class_scope:clipper.grpc.PredictResponse)
  ))
_sym_db.RegisterMessage(PredictResponse)



_PREDICT = _descriptor.ServiceDescriptor(
  name='Predict',
  full_name='clipper.grpc.Predict',
  file=DESCRIPTOR,
  index=0,
  options=None,
  serialized_start=123,
  serialized_end=212,
  methods=[
  _descriptor.MethodDescriptor(
    name='PredictFloats',
    full_name='clipper.grpc.Predict.PredictFloats',
    index=0,
    containing_service=None,
    input_type=_PREDICTREQUEST,
    output_type=_PREDICTRESPONSE,
    options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_PREDICT)

DESCRIPTOR.services_by_name['Predict'] = _PREDICT

# @@protoc_insertion_point(module_scope)
