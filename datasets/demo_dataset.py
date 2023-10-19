# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import pickle

import numpy as np
import torch

from PIL import Image

from pytorch_lightning.core import LightningDataModule

from utils import collate
from datasets.transforms import ScanNetBaseTransform

    

class DemoModule(LightningDataModule):
    """
    data module for demo snapshots.
    """

    def __init__(self, cfg):
        super().__init__()

        self.train_anno_path = cfg.TRAIN_ANNOTATION_PATH
        self.val_anno_path = cfg.VAL_ANNOTATION_PATH
        self.data_path = cfg.DATA_PATH
        self.num_workers = cfg.NUM_WORKERS
        self.batch_size = cfg.BATCH_SIZE
        self.drop_last = True
        self.transform = ScanNetBaseTransform(nviews=cfg.NUM_FRAMES_PER_SNIPPET, gravity_aligned=cfg.GRAVITY_ALIGNED)
        self.shuffle = cfg.SHUFFLE
        self.num_framesets_per_snippet=cfg.NUM_FRAMES_PER_SNIPPET

    def train_dataloader(self):
        return self.getLoader(self.data_path, self.train_anno_path)

    def val_dataloader(self):
        return self.getLoader(self.data_path, self.val_anno_path)

    def test_dataloader(self):
        return self.getLoader(self.data_path, self.val_anno_path)

    def getLoader(self, data_path, gt_path):
        dataset = DemoDataset(
            data_path=data_path,
            gt_path=gt_path,
            transform=self.transform,
            num_framesets_per_snippet=self.num_framesets_per_snippet,
        )

        params = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "collate_fn": collate,
            "drop_last": self.drop_last,
            "shuffle": self.shuffle,
        }
        dataloader = torch.utils.data.DataLoader(dataset, **params)
        return dataloader



class DemoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        gt_path: str,
        transform=None,
        shuffle=False,
        num_framesets_per_snippet=3,
    ):
        self.data_path = data_path
        self.gt_path = gt_path
        if transform is None:
            transform = ScanNetBaseTransform()
        self.transform = transform
        self.shuffle = shuffle
        self.item_list = self.get_roidb()
        self.anno_path = os.path.join(*(self.gt_path.split("/")[:-1] + ["scene_anno"]))
        self.scene_cache = {}
        self.max_cache = 100
        self.num_framesets_per_snippet = num_framesets_per_snippet
        
    def get_roidb(self):
        with open(self.gt_path, "rb") as f:
            item_list = pickle.load(f)
        print(
            "roidb loaded from {}, totally {} samples".format(
                self.gt_path, len(item_list)
            )
        )
        # item_list = item_list[:100]
        return item_list

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            scene_name:         str, e.g. scene0000_00, the scene name of the snippet
            rgb_img:            tensor (N,3,H,W), the RGB image sequence, N is the number of frames
            sym:                tensor (B,), symmetry type of the object, B is number of objects
            camera:             camera intrinsics 
            obb_padded:         ground truth bounding boxes
            T_world_camera:     Pose (N, 12), pose matrixes from camera to world
            T_world_pseudoCam:  Pose (N, 12), pose matrixes from pseudo camera to world
            T_camera_pseudoCam: Pose (N, 12), pose matrixes from pseudo camera to camera
            T_world_local:      Pose (N, 12), pose matrixes from local to world
        Note: pseudoCam coordinate is the gravity-aligned camera coordinate. 
        We rotate the camera coordinate to make it gravity-aligned.
        """
        item = self.item_list[idx]
        scene_name = item["scene"]
        snippet_id = item["fragment_id"]

        image_ids = item["image_ids"]
        T_scan_camera = item["extrinsics"]
        intrinsic = item["intrinsics"]

        imgs = []
        for id_ in image_ids:
            imgs.append(
                self.read_img(
                    os.path.join(
                        self.data_path,
                        scene_name,
                        "images",
                        '{}.jpg'.format(id_),
                    )
                )
            )

        out_dict = {
            "scene_name": scene_name,
            "snippet_id": snippet_id,
            "image_ids": image_ids,
            "rgb_img": imgs,
            "bboxes": None,
            "intrinsics": np.copy(intrinsic),
            "T_world_camera": T_scan_camera,
            "T_world_object": None,
            "label": None,
            "sym": None,
        }

        out_dict = self.transform(out_dict)
        out_dict.pop("bboxes")
        out_dict.pop("T_world_object")
        out_dict.pop("label")
        out_dict.pop("intrinsics")
        out_dict.pop("sym")
        out_dict.pop("obbs_padded")
        return out_dict
