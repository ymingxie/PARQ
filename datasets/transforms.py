# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import torch
from PIL import Image, ImageOps
from utils import Camera, Obb3D, Pose
from typing import Literal


UP_W = {
    "SCANNET": [0, 0, 1],
}

class GravityAligned(object):
    """
    Generate pseudoCam coordinates, which is the gravity-aligned camera coordinate
    """
    def __init__(self, dataset_type: Literal["SCANNET"]):
        self.dataset_type = dataset_type
        assert dataset_type in UP_W.keys()
        pass

    def __call__(self, batch):
        T_world_camera = batch["T_world_camera"]
        T_world_pseudoCam = self.camera_to_gravity_aligned(T_world_camera.matrix)
        T_world_pseudoCam = Pose.from_4x4mat(T_world_pseudoCam)
        batch["T_world_pseudoCam"] = T_world_pseudoCam
        batch["T_camera_pseudoCam"] = T_world_camera.inverse() @ T_world_pseudoCam

        return batch


    def camera_to_gravity_aligned(self, T_world_camera):
        up_w = UP_W[self.dataset_type]

        up_w = torch.tensor(up_w, device=T_world_camera.device).float()
        T_wv = torch.clone(T_world_camera)

        camForward = T_world_camera[..., :3, 2]
        R_wv = torch.zeros_like(T_wv[..., :3, :3], device=T_world_camera.device)
        R_wv[..., 1] = up_w
        R_wv[..., 2] = self.normalize(self.reject(camForward, up_w))
        R_wv[..., 0] = self.normalize(torch.cross(R_wv[..., 1], R_wv[..., 2], dim=-1))
        T_wv[..., :3, :3] = R_wv
        return T_wv


    def normalize(self, v):
        """normalize' a vector, in the traditional linear algebra sense."""
        norm = torch.norm(v, dim=-1, keepdim=True)
        if (norm == 0).any():
            return v
        return v / norm


    def reject(self, A, B):
        """Create a 'projection', and subract it from the original vector"""
        project = self.bdot(A, self.normalize(B)) * self.normalize(B)
        return A - project


    def bdot(self, a, b):
        return (a.unsqueeze(-2) @ b.unsqueeze(-1)).squeeze(-1)


def pad_scannet(img, intrinsics):
    """Scannet images are 1296x968 but 1296x972 is 4x3
    so we pad vertically 4 pixels to make it 4x3
    """

    w, h = img.size
    if w == 1296 and h == 968:
        img = ImageOps.expand(img, border=(0, 2))
        intrinsics[1, 2] += 2
    return img, intrinsics


class ResizeImage(object):
    """Resize everything to given size.
    Intrinsics are assumed to refer to image prior to resize.
    After resize everything (ex: depth) should have the same intrinsics
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        for i, im in enumerate(data["rgb_img"]):
            # intrinsic = np.copy(data["intrinsic"][i])
            im, intrinsics = pad_scannet(im, data["intrinsics"][i])
            w, h = im.size
            im = im.resize(self.size, Image.BILINEAR)
            intrinsics[0, :] /= w / self.size[0]
            intrinsics[1, :] /= h / self.size[1]

            im = np.array(im, dtype=np.float32)
            # if load with opencv
            # im = im[..., [2, 1, 0]]
            data["rgb_img"][i] = im
            data["intrinsics"][i] = intrinsics
        return data

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class Compose(object):
    """Apply a list of transforms sequentially"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class ToTensor(object):
    """Convert to torch tensors"""

    def __call__(self, data):
        data["rgb_img"] = torch.Tensor(
            np.stack(data["rgb_img"]).transpose([0, 3, 1, 2])
        )
        data["intrinsics"] = torch.Tensor(data["intrinsics"])
        data["T_world_camera"] = torch.Tensor(data["T_world_camera"])
        if data["bboxes"] is not None:
            data["bboxes"] = torch.Tensor(data["bboxes"])
            data["T_world_object"] = torch.Tensor(data["T_world_object"])
            data["label"] = torch.Tensor(data["label"])
            data["sym"] = torch.Tensor(data["sym"])
        return data


class Convert2Objects:
    """
    Wrap the 3d object, camera parameters, and poses. See utils/wrapper.py for details
    """
    def __init__(self, nviews=9):
        self.nviews = nviews

    def __call__(self, data):
        t, c, h, w = data["rgb_img"].shape
        intrinsics = data["intrinsics"][0]
        # fx, fy, cx, cy, w, h
        camera_param = torch.stack(
            [
                torch.Tensor([w]),
                torch.Tensor([h]),
                intrinsics[0, 0].unsqueeze(0),
                intrinsics[1, 1].unsqueeze(0),
                intrinsics[0, 2].unsqueeze(0),
                intrinsics[1, 2].unsqueeze(0),
            ],
            dim=-1
        )
        camera_param = camera_param.expand(self.nviews, -1)

        data["camera"] = Camera(camera_param)

        if data["bboxes"] is not None:
            T_world_object = Pose.from_4x4mat(data["T_world_object"])._data
            obbs = Obb3D.separate_init(
                bb3_object=data["bboxes"],
                T_world_object=T_world_object,
                sem_id=data["label"],
            )
            data["obbs_padded"] = obbs.add_padding()
        else:
            data["obbs_padded"] = None

        data["T_world_camera"] = Pose.from_4x4mat(data["T_world_camera"].float())

        return data


class Normalize:
    def __init__(self):
        self.pixel_mean = torch.Tensor([103.53, 116.28, 123.675]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([1.0, 1.0, 1.0]).view(-1, 1, 1)

    def __call__(self, data):
        """Normalizes the RGB images to the input range"""
        # data["img_rgb"] = (
        #     data["img_rgb"] - self.pixel_mean.type_as(data["img_rgb"])
        # ) / self.pixel_std.type_as(data["img_rgb"])
        data["rgb_img"] = data["rgb_img"] / 255
        return data


class SnippetLocal:
    """
    Select a coordinate as the local coordinate in the snippet. We will predict 3d objects in this local coordinate
    """
    def __init__(
        self,         
        frame_selection: float = 0.5,
    ):    
        self.frame_selection = frame_selection

    def __call__(self, batch):
        with torch.no_grad():
            if "T_world_pseudoCam" in batch.keys():
                Ts_world_psuedoCam = batch["T_world_pseudoCam"]
                T = Ts_world_psuedoCam.shape[0]
                t = int(T * self.frame_selection)
                batch["T_world_local"] = Ts_world_psuedoCam[t, :].clone().unsqueeze(0)
        return batch


class ScanNetBaseTransform(object):
    """
    Transforms for ScanNet dataset.
    """

    def __init__(self, nviews=9, gravity_aligned=True):
        transform = []
        transform += [
            ResizeImage((320, 240)), # resize image to 320x240
            ToTensor(),              # convert to torch tensor
            Normalize(),             # normalize the image
        ]
        transform += [
            Convert2Objects(nviews), # wrap the 3d object, camera parameters, and poses into a class. See utils/wrapper.py for details
        ]
        if gravity_aligned:
            transform += [
            GravityAligned(dataset_type='SCANNET') # generate pseudoCam coordinates, which is the gravity-aligned camera coordinate
            ]
        transform += [
            SnippetLocal(), # select a coordinate as the local coordinate in the snippet. We will predict 3d objects in this local coordinate
        ]
        self.transforms = Compose(transform)

    def __call__(self, batch):
        batch = self.transforms(batch)
        return batch
