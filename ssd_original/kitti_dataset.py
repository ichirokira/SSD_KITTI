"""
The code from pytorch web
Modified by TuyenNQ - s1262008@u-aizu.ac.jp
"""

import csv
import os
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import VisionDataset
from torch.utils.data.dataloader import default_collate
import torch

# Load config
from config import *


def collate_fn(batch):
    items = list(zip(*batch))
    items[0] = default_collate([i for i in items[0] if torch.is_tensor(i)])
    #items[1] = default_collate([i for i in items[1] if torch.is_tensor(i)])
    items[1] = list([i for i in items[1] if i])
    items[2] = list([i for i in items[2] if i])
    items[3] = default_collate([i for i in items[3] if torch.is_tensor(i)])
    items[4] = default_collate([i for i in items[4] if torch.is_tensor(i)])
    return items


class Kitti(VisionDataset):
    """
        It is modified from pytorch website to add loading local pattern image
    """
    # data_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
    # resources = [
    #     "data_object_image_2.zip",
    #     "data_object_label_2.zip",
    # ]

    

    def __init__(
            self,
            root: str,
            mode: str,

            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            #download: bool = False,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.image_dir_name = IMAGE_DIR
        self.labels_dir_name = LABEL_DIR
        self.lp_dir_name = LPIMAGE_DIR
        self.images = []
        self.lp_images = []
        self.targets = []
        self.img_ids = []
        self.root = root
        self.mode = mode
        self._location = "training" if self.mode == "train" or self.mode =="val" else "testing"

        # if download:
        #     self.download()
        # if not self._check_exists():
        #     raise RuntimeError(
        #         "Dataset not found. You may use download=True to download it."
        #     )

        image_dir = os.path.join(self.root, self._location,self.image_dir_name)
        lp_dir = os.path.join(self.root, self._location,self.lp_dir_name)

        img_files = os.listdir(image_dir)
        length = int(0.7*len(img_files))

        if self.mode == "train":
            img_files = img_files[:length]

        elif self.mode == "val":
            img_files = img_files[length:]
        if self.mode == "train" or self.mode == "val":
            labels_dir = os.path.join(self.root, self._location,self.labels_dir_name)
        for img_file in img_files:
            if img_file == "train.cache" or img_file=="val.cache":
              continue
            im_file = os.path.join(image_dir, img_file)
            lp_file = os.path.join(lp_dir, img_file)
            label_file = os.path.join(labels_dir, f"{img_file.split('.')[0]}.txt")
            if os.path.exists(label_file) and os.path.exists(im_file) and os.path.exists(lp_file):
              self.lp_images.append(lp_file)
              self.images.append(im_file)
              self.img_ids.append(img_file)
            if self.mode == "train" or self.mode == "val":
                if os.path.exists(label_file):
                  self.targets.append(label_file)

    def __getitem__(self, index):
        """Get item at a given index.

                Args:
                    index (int): Index
                Returns:
                    tuple: (image, lp_img, img_id,target ), where
                    target is a list of dictionaries with the following keys:

                    - type: str
                    - truncated: float
                    - occluded: int
                    - alpha: float
                    - bbox: float[4]
                    - dimensions: float[3]
                    - locations: float[3]
                    - rotation_y: float

                """
        image = Image.open(self.images[index])
        lp_image = Image.open(self.lp_images[index])
        image_id = self.img_ids[index]
        target = self._parse_target(index) if self.mode == "train" or self.mode == "val" else None
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, image_id, target

    def _parse_target(self, index: int) -> List:
        target = []
        with open(self.targets[index]) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                target.append({
                    "type": line[0],
                    "truncated": float(line[1]),
                    "occluded": int(line[2]),
                    "alpha": float(line[3]),
                    "bbox": [float(x) for x in line[4:8]],
                    "dimensions": [float(x) for x in line[8:11]],
                    "location": [float(x) for x in line[11:14]],
                    "rotation_y": float(line[14]),
                })
        return target

    def __len__(self) -> int:
        return len(self.images)

    # @property
    # def _raw_folder(self) -> str:
    #     return os.path.join(self.root, self.__class__.__name__, "raw")
    #
    # def _check_exists(self) -> bool:
    #     """Check if the data directory exists."""
    #     folders = [self.image_dir_name]
    #     if self.train:
    #         folders.append(self.labels_dir_name)
    #     return all(
    #         os.path.isdir(os.path.join(self._raw_folder, self._location, fname))
    #         for fname in folders
    #     )
    #
    # def download(self) -> None:
    #     """Download the KITTI data if it doesn't exist already."""
    #
    #     if self._check_exists():
    #         return
    #
    #     os.makedirs(self._raw_folder, exist_ok=True)
    #
    #     # download files
    #     for fname in self.resources:
    #         download_and_extract_archive(
    #             url=f"{self.data_url}{fname}",
    #             download_root=self._raw_folder,
    #             filename=fname,
    #         )


class KittiDataset(Kitti):
    def __init__(self, root, mode, transform=None):
        super(KittiDataset, self).__init__(root, mode)
        self._load_categories()
        self.transform = transform

    def _load_categories(self):
        self.label_map = {}
        self.label_info = {}
        counter = 1
        self.label_info[0] = "background"
        for c in KITTI_CLASSES:
            self.label_map[c] = counter
            self.label_info[counter] = c
            counter += 1

    def __getitem__(self, item):
        image, image_id,target = super(KittiDataset, self).__getitem__(item)
        width, height = image.size
        boxes = []
        labels = []
        if len(target) == 0:
            return None, None, None, None
        for annotation in target:
            bbox = annotation.get("bbox")
            boxes.append([bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height])
            labels.append(self.label_map[annotation.get("type")])
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)
        if self.transform is not None:
            image, (height, width), boxes, labels = self.transform(image, (height, width), boxes, labels)

        return image, image_id, (height, width), boxes, labels

        # `label`: 8732
        # `boxes`: 8732 x 4