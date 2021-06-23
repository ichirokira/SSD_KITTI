"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import shutil
from argparse import ArgumentParser

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_original import SSD,  ResNet
from utils import generate_dboxes, Encoder
from transform import SSDTransformer
from loss import Loss
from process import train, evaluate
from kitti_dataset import collate_fn, KittiDataset
from config import *

def get_args():
    parser = ArgumentParser(description="Implementation of SSD")
    parser.add_argument("--data-path", type=str, default="/coco",
                        help="the root folder of dataset")
    parser.add_argument("--save-folder", type=str, default="trained_models/original",
                        help="path to folder containing model checkpoint file")
    parser.add_argument("--log-path", type=str, default="tensorboard/SSD_original")

    # parser.add_argument("--model", type=str, default="ssd", choices=["ssd", "ssdlite"],
    #                     help="ssd-resnet50 or ssdlite-mobilenetv2")
    parser.add_argument("--epochs", type=int, default=65, help="number of total epochs to run")
    parser.add_argument("--batch-size", type=int, default=32, help="number of samples for each iteration")
    parser.add_argument("--multistep", nargs="*", type=int, default=[43, 54],
                        help="epochs at which to decay learning rate")
    parser.add_argument("--amp", action='store_true', help="Enable mixed precision training")

    parser.add_argument("--lr", type=float, default=2.6e-3, help="initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum argument for SGD optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="momentum argument for SGD optimizer")
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m multiproc\'.')
    args = parser.parse_args()
    return args


def main(opt):
    num_gpus = torch.cuda.device_count()

    train_params = {"batch_size": opt.batch_size * num_gpus,
                    "shuffle": True,
                    "drop_last": False,
                    "num_workers": opt.num_workers,
                    "collate_fn": collate_fn}

    test_params = {"batch_size": opt.batch_size * num_gpus,
                   "shuffle": False,
                   "drop_last": False,
                   "num_workers": opt.num_workers,
                   "collate_fn": collate_fn}


    dboxes = generate_dboxes(model="ssd")
    model = SSD(backbone=ResNet(), num_classes=len(KITTI_CLASSES))

    train_set = KittiDataset(opt.data_path, "train", SSDTransformer(dboxes, (300, 300), val=False))
    train_loader = DataLoader(train_set, **train_params)
    test_set = KittiDataset(opt.data_path, "val", SSDTransformer(dboxes, (300, 300), val=True))
    test_loader = DataLoader(test_set, **test_params)

    encoder = Encoder(dboxes)

    opt.lr = opt.lr * num_gpus * (opt.batch_size / 32)
    criterion = Loss(dboxes)

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=opt.multistep, gamma=0.1)

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

       


    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    checkpoint_path = os.path.join(opt.save_folder, "SSD.pth")

    writer = SummaryWriter(opt.log_path)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        first_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        first_epoch = 0

    for epoch in range(first_epoch, opt.epochs):
        train(model, train_loader, epoch, writer, criterion, optimizer, scheduler, opt.amp)
        #evaluate(model, test_loader, epoch, writer, encoder, opt.nms_threshold)

        checkpoint = {"epoch": epoch,
                      "model_state_dict": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    opt = get_args()
    main(opt)
