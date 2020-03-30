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
import numpy as np
import logging
from tqdm import tqdm
import grpc
from concurrent import futures
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

parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'xception', 'drn', 'mobilenet'],
                    help='backbone name (default: resnet)')
parser.add_argument('--out-stride', type=int, default=16, help='network output stride (default: 8)')
parser.add_argument('--dataset', type=str, default='pascal', choices=['pascal', 'coco', 'cityscapes'],
                    help='dataset name (default: pascal)')
parser.add_argument('--use-sbd', action='store_true', default=True,
                    help='whether to use SBD dataset (default: True)')
parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
parser.add_argument('--base-size', type=int, default=513, help='base image size')
parser.add_argument('--crop-size', type=int, default=513, help='crop image size')
parser.add_argument('--sync-bn', type=bool, default=None, help='whether to use sync bn (default: auto)')
parser.add_argument('--freeze-bn', type=bool, default=False,
                    help='whether to freeze bn parameters (default: False)')
parser.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'focal'],
                    help='loss func type (default: ce)')
# training hyper params
parser.add_argument('--epochs', type=int, default=None, metavar='N',
                    help='number of epochs to train (default: auto)')
parser.add_argument('--start_epoch', type=int, default=0,
                    metavar='N', help='start epochs (default:0)')
parser.add_argument('--batch-size', type=int, default=None,
                    metavar='N', help='input batch size for \
                            training (default: auto)')
parser.add_argument('--test-batch-size', type=int, default=None,
                    metavar='N', help='input batch size for \
                            testing (default: auto)')
parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                    help='whether to use balanced weights (default: False)')
# optimizer params
parser.add_argument('--lr', type=float, default=None, metavar='LR',
                    help='learning rate (default: auto)')
parser.add_argument('--lr-scheduler', type=str, default='poly',
                    choices=['poly', 'step', 'cos'],
                    help='lr scheduler mode: (default: poly)')
parser.add_argument('--momentum', type=float, default=0.9,
                    metavar='M', help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                    metavar='M', help='w-decay (default: 5e-4)')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='whether use nesterov (default: False)')
# cuda, seed and logging
parser.add_argument('--no-cuda', action='store_true', default=
False, help='disables CUDA training')
parser.add_argument('--gpu-ids', type=str, default='0',
                    help='use which gpu to train, must be a \
                    comma-separated list of integers only (default=0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# checking point
parser.add_argument('--resume', type=str, default=None,
                    help='put the path to resuming file if needed')
parser.add_argument('--checkname', type=str, default=None,
                    help='set the checkpoint name')
# finetuning pre-trained models
parser.add_argument('--ft', action='store_true', default=False,
                    help='finetuning on a different dataset')
# evaluation option
parser.add_argument('--eval-interval', type=int, default=1,
                    help='evaluuation interval (default: 1)')
parser.add_argument('--no-val', action='store_true', default=False,
                    help='skip validation during training')


class Server(segmentation_pb2_grpc.SegmentationServicer):

    def __init__(self, *args, **kwargs):
        super(Server, self).__init__(*args, **kwargs)
        args = parser.parse_known_args()

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        if args.cuda:
            try:
                args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            except ValueError:
                raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

        if args.sync_bn is None:
            if args.cuda and len(args.gpu_ids) > 1:
                args.sync_bn = True
            else:
                args.sync_bn = False

        # default settings for epochs, batch_size and lr
        if args.epochs is None:
            epoches = {
                'coco': 30,
                'cityscapes': 200,
                'pascal': 50,
            }
            args.epochs = epoches[args.dataset.lower()]

        if args.batch_size is None:
            args.batch_size = 4 * len(args.gpu_ids)

        if args.test_batch_size is None:
            args.test_batch_size = args.batch_size

        if args.lr is None:
            lrs = {
                'coco': 0.1,
                'cityscapes': 0.01,
                'pascal': 0.007,
            }
            args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

        if args.checkname is None:
            args.checkname = 'deeplab-' + str(args.backbone)
        print(args)
        torch.manual_seed(args.seed)

        self.initialize_model(args)

    def initialize_model(self, args):
        self.args = args

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
        pred = output.data.cpu().numpy()
        pred = Image.fromarray(np.argmax(pred, axis=1), mode='L')
        request.image_segmentation_class_encoded = pred.tobytes()

        return request

    def recvFeatures(self, request, context):
        for r in request:
            output = self.process(r)
            pred = output.data.cpu().numpy()
            pred = Image.fromarray(np.argmax(pred, axis=1), mode='L')
            r.image_segmentation_class_encoded = pred.tobytes()

        return request


def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    segmentation_pb2_grpc.add_SegmentationServicer_to_server(Server(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig()
    main()
