# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: segmentation.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from gateway.protocol import gateway_pb2 as gateway_dot_protocol_dot_gateway__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='segmentation.proto',
  package='protocol',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x12segmentation.proto\x12\x08protocol\x1a\x1egateway/protocol/gateway.proto2}\n\x0cSegmentation\x12\x34\n\x0brecvFeature\x12\x11.protocol.Feature\x1a\x12.protocol.Response\x12\x37\n\x0crecvFeatures\x12\x12.protocol.Features\x1a\x13.protocol.Responsesb\x06proto3'
  ,
  dependencies=[gateway_dot_protocol_dot_gateway__pb2.DESCRIPTOR,])



_sym_db.RegisterFileDescriptor(DESCRIPTOR)



_SEGMENTATION = _descriptor.ServiceDescriptor(
  name='Segmentation',
  full_name='protocol.Segmentation',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=64,
  serialized_end=189,
  methods=[
  _descriptor.MethodDescriptor(
    name='recvFeature',
    full_name='protocol.Segmentation.recvFeature',
    index=0,
    containing_service=None,
    input_type=gateway_dot_protocol_dot_gateway__pb2._FEATURE,
    output_type=gateway_dot_protocol_dot_gateway__pb2._RESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='recvFeatures',
    full_name='protocol.Segmentation.recvFeatures',
    index=1,
    containing_service=None,
    input_type=gateway_dot_protocol_dot_gateway__pb2._FEATURES,
    output_type=gateway_dot_protocol_dot_gateway__pb2._RESPONSES,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_SEGMENTATION)

DESCRIPTOR.services_by_name['Segmentation'] = _SEGMENTATION

# @@protoc_insertion_point(module_scope)
