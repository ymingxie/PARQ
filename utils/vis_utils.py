# Copyright (c) Meta Platforms, Inc. and its affiliates. All Rights Reserved

import torch


def pca_compress(img):
    B, C, H, W = img.shape
    assert B == 1
    x = img.view(C, -1).transpose(1, 0)
    x -= torch.mean(x, 0, True)
    U, S, V = torch.pca_lowrank(x, center=False)
    x = torch.matmul(x, V[:, :3])
    return x.transpose(1, 0).view(1, 3, H, W)


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())