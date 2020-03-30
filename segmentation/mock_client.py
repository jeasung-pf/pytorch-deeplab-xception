# Copyright © 2020 Jea Sung Park jeasung@peoplefund.co.kr
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import grpc

from protocol import segmentation_pb2
from protocol import segmentation_pb2_grpc


def guide_recv_feature(stub):



if __name__ == "__main__":
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = segmentation_pb2_grpc.SegmentationStub(channel)
        print("---------- RecvFeature ----------")
        guide_recv_feature(stub)
