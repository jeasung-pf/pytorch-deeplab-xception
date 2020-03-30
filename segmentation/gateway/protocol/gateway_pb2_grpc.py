# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from gateway.protocol import gateway_pb2 as gateway_dot_protocol_dot_gateway__pb2


class GatewayStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.recvFeature = channel.unary_unary(
        '/protocol.Gateway/recvFeature',
        request_serializer=gateway_dot_protocol_dot_gateway__pb2.Feature.SerializeToString,
        response_deserializer=gateway_dot_protocol_dot_gateway__pb2.Response.FromString,
        )
    self.recvFeatures = channel.unary_unary(
        '/protocol.Gateway/recvFeatures',
        request_serializer=gateway_dot_protocol_dot_gateway__pb2.Features.SerializeToString,
        response_deserializer=gateway_dot_protocol_dot_gateway__pb2.Responses.FromString,
        )


class GatewayServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def recvFeature(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def recvFeatures(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_GatewayServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'recvFeature': grpc.unary_unary_rpc_method_handler(
          servicer.recvFeature,
          request_deserializer=gateway_dot_protocol_dot_gateway__pb2.Feature.FromString,
          response_serializer=gateway_dot_protocol_dot_gateway__pb2.Response.SerializeToString,
      ),
      'recvFeatures': grpc.unary_unary_rpc_method_handler(
          servicer.recvFeatures,
          request_deserializer=gateway_dot_protocol_dot_gateway__pb2.Features.FromString,
          response_serializer=gateway_dot_protocol_dot_gateway__pb2.Responses.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'protocol.Gateway', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
