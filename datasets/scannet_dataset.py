# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import pickle

import numpy as np
import torch

from PIL import Image

from pytorch_lightning.core import LightningDataModule

from utils import collate
from datasets.transforms import ScanNetBaseTransform

    

class ScanNetDataModule(LightningDataModule):
    """
    data module for scannet snapshots.
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
        dataset = ScanNetDataset(
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



class ScanNetDataset(torch.utils.data.Dataset):
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

    def read_anno(self, scene_name, snippet_id):
        if scene_name not in self.scene_cache.keys():
            if len(self.scene_cache) > self.max_cache:
                self.scene_cache = {}
            anno_path = os.path.join(self.anno_path, scene_name + ".pkl")
            with open(anno_path, "rb") as f:
                anno = pickle.load(f)
            self.scene_cache[scene_name] = anno
        return self.scene_cache[scene_name][snippet_id]

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
        scene_name = item["scene_name"]
        snippet_id = item["snippet_id"]
        item = self.read_anno(item["scene_name"], item["snippet_id"])

        image_ids = item["image_ids"]
        T_scan_camera = item["T_scan_camera"]
        intrinsic = item["intrinsic"]
        
        num_frames = len(image_ids)
        if self.num_framesets_per_snippet != num_frames:
            replace = True if self.num_framesets_per_snippet > num_frames else False
            choose = np.random.choice(num_frames, self.num_framesets_per_snippet, replace=replace)
            choose = np.sort(choose)
            image_ids = [image_ids[choo] for choo in choose]
            T_scan_camera = [T_scan_camera[choo] for choo in choose]
            intrinsic = [intrinsic[choo] for choo in choose]
            
        annos = item["annotations"]
        bboxes = np.array(annos["bboxes"])
        T_scan_object = np.array(annos["T_scan_object"])
        label = annos["label"]
        sym = annos["sym"]
        sym_mapping = {
            "__SYM_NONE": 0,
            "__SYM_ROTATE_UP_2": 1,
            "__SYM_ROTATE_UP_4": 2,
            "__SYM_ROTATE_UP_INF": 3,
        }
        for i in range(len(sym)):
            if sym[i] in sym_mapping.keys():
                sym[i] = sym_mapping[sym[i]]
        # padding sym
        padding_size = 50
        for i in range(padding_size - len(sym)):
            sym.append(-1)

        sym = np.array(sym)

        imgs = []
        for id_ in image_ids:
            imgs.append(
                self.read_img(
                    os.path.join(
                        self.data_path,
                        scene_name,
                        "color",
                        'frame-{0:06d}.color.jpg'.format(id_),
                    )
                )
            )

        out_dict = {
            "scene_name": scene_name,
            "snippet_id": snippet_id,
            "image_ids": image_ids,
            "rgb_img": imgs,
            "bboxes": bboxes,
            "intrinsics": np.copy(intrinsic),
            "T_world_camera": T_scan_camera,
            "T_world_object": T_scan_object,
            "label": label,
            "sym": sym,
        }

        out_dict = self.transform(out_dict)
        out_dict.pop("bboxes")
        out_dict.pop("T_world_object")
        out_dict.pop("label")
        out_dict.pop("intrinsics")
        return out_dict
