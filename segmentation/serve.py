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

import argparse
import os
import numpy
from tqdm import tqdm
from PIL import Image, ImageFile
from torchvision import transforms

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

from protocol import segmentation_pb2_grpc

class Server(segmentation_pb2_grpc.SegmentationServicer):

    def __init__(self, params, *args, **kwargs):
        super(Server, self).__init__(*args, **kwargs)
        self.args = params

        # Define Saver
        self.saver = Saver(self.args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        # Define Dataloader
        kwargs = {
            'num_worker': self.args.worker,
            'pin_memory': True
        }
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(self.args, **kwargs)
        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=self.args.backbone,
                        output_stride=self.args.out_stride,
                        sync_bn=self.args.sync_bn,
                        freeze_bn=self.args.freeze_bn)
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)

        # Using cuda
        if self.args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        if not os.path.isfile(self.args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(self.args.resume))
        checkpoint = torch.load(self.args.resume)
        self.args.start_epoch = checkpoint['epoch']
        if self.args.cuda:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.evaluator.reset()



    def process(self, request):

        if not hasattr(request, "image_encoded"):
            raise ValueError("One of current request doesn't contain encoded image.")

        image_encoded = request.image_encoded

        def _decode_image(content):
            image_parser = ImageFile.Parser()
            image_parser.feed(content)
            return image_parser.close()

        image = transforms.ToTensor(_decode_image(image_encoded))
        if self.args.cuda:
            image = image.cuda()
        with torch.no_grad():
            output = self.model(image)

        return output

    def recvFeature(self, request, context):
        output = self.process(request)
        request.image_segmentation_class_encoded = output

    def recvFeatures(self, request, context):
        for r in request:
            output = self.process(r)
            r.image_segmentation_class_encoded = output
