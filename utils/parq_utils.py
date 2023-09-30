# Copyright (c) Meta Platforms, Inc. and its affiliates. All Rights Reserved

import colorsys
import itertools
from fractions import Fraction

import cv2
import numpy as np
import torch

from .matcher import HungarianMatcherModified
from .f1_eval import F1Calculator
from .ortho6d_transforms import (
    compute_rotation_matrix_from_ortho6d,
    rot_to_6d,
)
from .nms import nms

__all__ = [
    "BoxProcessor",
    "HungarianMatcherModified",
    "prepare_for_dn",
    "dn_post_process",
    "rot_to_6d",
    "compute_rotation_matrix_from_ortho6d",
    "F1Calculator",
    "nms",
]


class BoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    Convertion between different representation of boxes (corners, dof, etc.)
    """

    def __init__(self, num_semcls, mean_size_path):
        self.num_semcls = num_semcls

        self.num_class = 9
        self.mean_size_path = mean_size_path
        if self.mean_size_path is not None:
            self.init_mean_size()

    def init_mean_size(self):
        # TODO: only for ScanNet
        self.type2class = {
            "chair": 0,
            "table": 1,
            "cabinet": 2,
            "trash bin": 3,
            "bookshelf": 4,
            "display": 5,
            "sofa": 6,
            "bathtub": 7,
            "other": 8,
        }

        self.class2type = {self.type2class[t]: t for t in self.type2class}

        self.typelong_mean_size = {}
        with open(self.mean_size_path, "r") as f:
            for line in f.readlines():
                type_cat, size = line.split(": ")
                size = size[1:-3].split(" ")
                size_ = []
                for j, s in enumerate(size):
                    if len(s) != 0:
                        size_.append(s)
                size = [float(size_[i]) for i in [0, 1, 2]]
                self.typelong_mean_size[type_cat] = size

        self.mean_size_arr = []
        self.type_mean_size = {}
        for i in range(self.num_class):
            object_type = self.class2type[i]
            for key, value in self.typelong_mean_size.items():
                key = key.split(",")
                if object_type in key:
                    self.mean_size_arr.append(value)
                    self.type_mean_size[object_type] = value
                    break

        self.mean_size_arr.append([1, 1, 1])
        self.type_mean_size["other"] = [1, 1, 1]
        self.mean_size_arr.append([1, 1, 1])
        self.type_mean_size["non-object"] = [1, 1, 1]
        self.mean_size_arr = torch.from_numpy(np.array(self.mean_size_arr))

    def compute_predicted_center(self, center_normalized):
        center_unnormalized = center_normalized * 1
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_scale, cls_prob):
        if self.mean_size_path is not None:
            pred_cls = cls_prob.argmax(-1)
            mean_size = self.mean_size_arr[pred_cls.data.cpu()]
            size_pred = torch.exp(size_scale) * mean_size.to(size_scale.device).float()
        return size_pred

    def compute_objectness_and_cls_prob(self, cls_logits):
        assert cls_logits.shape[-1] == self.num_semcls + 1
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob, objectness_prob


def get_faces():
    return [
        [0, 1, 2, 3],
        [0, 3, 7, 4],
        [0, 4, 5, 1],
        [1, 2, 6, 5],
        [2, 6, 7, 3],
        [7, 4, 5, 6],
    ]


def infinite_hues():
    yield Fraction(0)
    for k in itertools.count():
        i = 2**k  # zenos_dichotomy
        for j in range(1, i, 2):
            yield Fraction(j, i)


def hue_to_hsvs(h: Fraction):
    # tweak values to adjust scheme
    for s in [Fraction(6, 10)]:
        for v in [Fraction(6, 10), Fraction(9, 10)]:
            yield (h, s, v)


def get_colors(n):
    hues = infinite_hues()
    hsvs = itertools.chain.from_iterable(hue_to_hsvs(hue) for hue in hues)
    rgbs = (colorsys.hsv_to_rgb(*hsv) for hsv in hsvs)
    return list((float(r), float(g), float(b)) for r, g, b in itertools.islice(rgbs, n))


def draw_detections(
    imgs, calibs, box_corners, T_world_object, T_pseudoCam_world, T_camera_pseudoCam, labels, tags, num_semcls, mask=None
):
    # make sure the number of cla
    id2color = get_colors(num_semcls)
    imgs_output = {}
    # default order in the img list: rgb, slaml, slamr
    for j in range(len(imgs)):
        cam = calibs[j]
        T = cam.shape[0]
        imgs_mviews = []
        imgs_withbox_mviews = []
        for t in range(T):
            cam_t = cam[t : t + 1]
            # log the first img in T
            if j == 0:
                img = imgs[j][t].permute(1, 2, 0).data.cpu().numpy()
            else:
                img = imgs[j][t, 0].data.cpu().numpy()
                img = np.stack([img, img, img], axis=-1)
            # normalization
            img = (img - img.min()) / (img.max() - img.min())
            img_withbox = img.copy()
            T_pseudoCam_object = T_pseudoCam_world[t : t + 1] @ T_world_object
            T_camera_pseudoCam_ = T_camera_pseudoCam[t : t + 1]
            for m, (box_corners_, T_pseudoCam_object_, sem_) in enumerate(zip(
                box_corners, T_pseudoCam_object, labels
            )):
                if sem_ != num_semcls:
                    if mask is not None and mask[m] == False:
                        continue
                    color = tuple((np.array(id2color[sem_])).tolist())
                    # compute projections of the corners into the image
                    box_corners_pseudo = T_pseudoCam_object_.transform(box_corners_)
                    box_corners_c = T_camera_pseudoCam_.transform(box_corners_pseudo)
                    corners_im, valid = cam_t.project(box_corners_c)

                    corners_im = corners_im[0].data.cpu().numpy()
                    valid = valid[0].data.cpu().numpy()
                    # img_withbox = img_withbox.copy()
                    faces = get_faces()
                    for face in faces:
                        for i in range(len(face) - 1):
                            if valid[face[i]] and valid[face[i + 1]]:
                                img_withbox = cv2.line(
                                    img_withbox,
                                    tuple(
                                        corners_im[face[i], :].astype(np.int).tolist()
                                    ),
                                    tuple(
                                        corners_im[face[i + 1], :]
                                        .astype(np.int)
                                        .tolist()
                                    ),
                                    color,
                                    thickness=2,
                                )
            img = torch.from_numpy(img)
            img_withbox = torch.from_numpy(img_withbox)
            if len(imgs) != 1:
                img = torch.rot90(img, -1, [0, 1])
            imgs_mviews.append(img)
            imgs_withbox_mviews.append(img_withbox)
        img = torch.cat(imgs_mviews, dim=0)
        img = img.permute((2, 0, 1)).contiguous()
        img_withbox = torch.cat(imgs_withbox_mviews, dim=0)
        img_withbox = img_withbox.permute((2, 0, 1)).contiguous()
        if "gt" not in tags[j]:
            imgs_output[tags[j]] = img
        imgs_output[tags[j] + 'withbox'] = img_withbox
    return imgs_output


def roty(t, device="cpu"):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return torch.Tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]]).to(device)


def rotz(t, device="cpu"):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return torch.Tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]]).to(device)
