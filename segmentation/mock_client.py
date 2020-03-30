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

import os
import grpc
import argparse
from PIL import Image, ImageFile
from tqdm import tqdm

from dataloaders import make_data_loader

from protocol import segmentation_pb2
from protocol import segmentation_pb2_grpc
from gateway.protocol import gateway_pb2


def guide_recv_feature(stub, args):
    prefix = "/dltraining/test/VOCdevkit"
    with open(os.path.join(prefix, "ImageSets/Segmentation/test.txt"), 'r') as f:
        for line in tqdm(f):
            line = line.strip()
            line = line + "jpg"
            with open(os.path.join(prefix, "JPEGImages", line)) as image:
                image_encoded = image.read()
                parser = ImageFile.Parser()
                parser.feed(image)
                image = parser.close()

                feature = gateway_pb2.Feature(image_encoded=image_encoded,
                                              image_filename=line,
                                              image_format="jpg",
                                              image_height=image.height,
                                              image_width=image.width,
                                              image_segmentation_class_encoded="",
                                              image_segmentation_class_format="jpg")
                try:
                    response = stub.recvFeature(feature)
                except Exception as e:
                    print(e)
                encoded = response.image_segmentation_class_encoded
                segmentation = Image.frombytes('L', (image.width, image.height), encoded)
                with open(os.path.join(prefix, "RESULT", line), "w+") as fp:
                    segmentation.save(fp)


if __name__ == "__main__":
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = segmentation_pb2_grpc.SegmentationStub(channel)
        print("---------- RecvFeature ----------")
        guide_recv_feature(stub)
        stub.recvFeature()
