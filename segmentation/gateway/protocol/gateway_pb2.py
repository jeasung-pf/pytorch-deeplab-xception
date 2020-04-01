# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: gateway/protocol/gateway.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.api import annotations_pb2 as google_dot_api_dot_annotations__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='gateway/protocol/gateway.proto',
  package='protocol',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x1egateway/protocol/gateway.proto\x12\x08protocol\x1a\x1cgoogle/api/annotations.proto\"\xcc\x01\n\x07\x46\x65\x61ture\x12\x15\n\rimage_encoded\x18\x01 \x01(\x0c\x12\x16\n\x0eimage_filename\x18\x02 \x01(\t\x12\x14\n\x0cimage_format\x18\x03 \x01(\t\x12\x14\n\x0cimage_height\x18\x04 \x01(\x05\x12\x13\n\x0bimage_width\x18\x05 \x01(\x05\x12(\n image_segmentation_class_encoded\x18\x06 \x01(\x0c\x12\'\n\x1fimage_segmentation_class_format\x18\x07 \x01(\t\"/\n\x08\x46\x65\x61tures\x12#\n\x08\x66\x65\x61tures\x18\x01 \x03(\x0b\x32\x11.protocol.Feature\"\x8e\x01\n\x08Response\x12\x16\n\x0eimage_filename\x18\x01 \x01(\t\x12!\n\x19image_recognition_numbers\x18\x02 \x01(\t\x12&\n\x1eimage_recognition_verification\x18\x03 \x01(\t\x12\x1f\n\x17image_recognition_dates\x18\x04 \x01(\t\"2\n\tResponses\x12%\n\tresponses\x18\x01 \x03(\x0b\x32\x12.protocol.Response2\xb9\x01\n\x07Gateway\x12P\n\x0brecvFeature\x12\x11.protocol.Feature\x1a\x12.protocol.Response\"\x1a\x82\xd3\xe4\x93\x02\x14\"\x0f/v1/recognition:\x01*\x12\\\n\x0crecvFeatures\x12\x12.protocol.Features\x1a\x13.protocol.Responses\"#\x82\xd3\xe4\x93\x02\x1d\"\x18/v1/multiple/recognition:\x01*b\x06proto3'
  ,
  dependencies=[google_dot_api_dot_annotations__pb2.DESCRIPTOR,])




_FEATURE = _descriptor.Descriptor(
  name='Feature',
  full_name='protocol.Feature',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='image_encoded', full_name='protocol.Feature.image_encoded', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_filename', full_name='protocol.Feature.image_filename', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_format', full_name='protocol.Feature.image_format', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_height', full_name='protocol.Feature.image_height', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_width', full_name='protocol.Feature.image_width', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_segmentation_class_encoded', full_name='protocol.Feature.image_segmentation_class_encoded', index=5,
      number=6, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_segmentation_class_format', full_name='protocol.Feature.image_segmentation_class_format', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=75,
  serialized_end=279,
)


_FEATURES = _descriptor.Descriptor(
  name='Features',
  full_name='protocol.Features',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='features', full_name='protocol.Features.features', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=281,
  serialized_end=328,
)


_RESPONSE = _descriptor.Descriptor(
  name='Response',
  full_name='protocol.Response',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='image_filename', full_name='protocol.Response.image_filename', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_recognition_numbers', full_name='protocol.Response.image_recognition_numbers', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_recognition_verification', full_name='protocol.Response.image_recognition_verification', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_recognition_dates', full_name='protocol.Response.image_recognition_dates', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=331,
  serialized_end=473,
)


_RESPONSES = _descriptor.Descriptor(
  name='Responses',
  full_name='protocol.Responses',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='responses', full_name='protocol.Responses.responses', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=475,
  serialized_end=525,
)

_FEATURES.fields_by_name['features'].message_type = _FEATURE
_RESPONSES.fields_by_name['responses'].message_type = _RESPONSE
DESCRIPTOR.message_types_by_name['Feature'] = _FEATURE
DESCRIPTOR.message_types_by_name['Features'] = _FEATURES
DESCRIPTOR.message_types_by_name['Response'] = _RESPONSE
DESCRIPTOR.message_types_by_name['Responses'] = _RESPONSES
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Feature = _reflection.GeneratedProtocolMessageType('Feature', (_message.Message,), {
  'DESCRIPTOR' : _FEATURE,
  '__module__' : 'gateway.protocol.gateway_pb2'
  # @@protoc_insertion_point(class_scope:protocol.Feature)
  })
_sym_db.RegisterMessage(Feature)

Features = _reflection.GeneratedProtocolMessageType('Features', (_message.Message,), {
  'DESCRIPTOR' : _FEATURES,
  '__module__' : 'gateway.protocol.gateway_pb2'
  # @@protoc_insertion_point(class_scope:protocol.Features)
  })
_sym_db.RegisterMessage(Features)

Response = _reflection.GeneratedProtocolMessageType('Response', (_message.Message,), {
  'DESCRIPTOR' : _RESPONSE,
  '__module__' : 'gateway.protocol.gateway_pb2'
  # @@protoc_insertion_point(class_scope:protocol.Response)
  })
_sym_db.RegisterMessage(Response)

Responses = _reflection.GeneratedProtocolMessageType('Responses', (_message.Message,), {
  'DESCRIPTOR' : _RESPONSES,
  '__module__' : 'gateway.protocol.gateway_pb2'
  # @@protoc_insertion_point(class_scope:protocol.Responses)
  })
_sym_db.RegisterMessage(Responses)



_GATEWAY = _descriptor.ServiceDescriptor(
  name='Gateway',
  full_name='protocol.Gateway',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=528,
  serialized_end=713,
  methods=[
  _descriptor.MethodDescriptor(
    name='recvFeature',
    full_name='protocol.Gateway.recvFeature',
    index=0,
    containing_service=None,
    input_type=_FEATURE,
    output_type=_RESPONSE,
    serialized_options=b'\202\323\344\223\002\024\"\017/v1/recognition:\001*',
  ),
  _descriptor.MethodDescriptor(
    name='recvFeatures',
    full_name='protocol.Gateway.recvFeatures',
    index=1,
    containing_service=None,
    input_type=_FEATURES,
    output_type=_RESPONSES,
    serialized_options=b'\202\323\344\223\002\035\"\030/v1/multiple/recognition:\001*',
  ),
])
_sym_db.RegisterServiceDescriptor(_GATEWAY)

DESCRIPTOR.services_by_name['Gateway'] = _GATEWAY

# @@protoc_insertion_point(module_scope)