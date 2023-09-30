# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging

import einops
import torch
from utils.encoding_utils import (
    grid_2d,
    ray_points_snippet,
)

from torch.nn import Module

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.WARNING)


EPS = 1e-5


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


class AddRayPE(Module):
    def __init__(
        self,
        dim_out: int,
        # [min_x, max_x, min_y, max_y, min_z, max_z]
        ray_points_scale: list = [-2, 2, -1.5, 0, 0.25, 4.25],
        num_samples: int = 64,  # number of ray points
        min_depth: float = 0.25,
        max_depth: float = 5.25,
    ):
        """
        Args:
            out_channels: Output channels required from model
            ray_points_scale: [min_x, max_x, min_y, max_y, min_z, max_z] used to normalize the points along each ray
            num_samples:  number of ray points
            min_depth:    minimum depth of the ray points
            max_depth:    maximum depth of the ray points
        """
        super().__init__()

        self.dim_out = dim_out

        self.ray_points_scale = ray_points_scale
        self.num_samples = num_samples
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.encoder = torch.nn.Sequential(
                torch.nn.Linear(3 * self.num_samples, dim_out),
                torch.nn.ReLU(),
                torch.nn.Linear(dim_out, dim_out),
        )

    def forward(
        self,
        images_feat: torch.Tensor,
        camera: torch.Tensor = None,
        T_camera_pseudoCam: torch.Tensor = None,
        T_world_pseudoCam: torch.Tensor = None,
        T_world_local: torch.Tensor = None,
    ):
        '''
        input:
            images_feat:        (B, T, C, H, W), image features
            camera:             (B, T, 6), camera intrinsics: with, height, fx, fy, cx, cy
            T_camera_pseudoCam: (B, T, 12), pseudo camera to camera transformation
            T_world_pseudoCam:  (B, T, 12), pseudo camera to world transformation
            T_world_local:      (B, 12), local to world transformation
        output:
            images_feat:        (B, T*H*W, C), tokenized image features with ray position encoding, patch size = 1
        '''
        B, T = images_feat.shape[0], images_feat.shape[1]

        width, height = camera.size[0, 0]
        width, height = width.round().int().item(), height.round().int().item()
        pos_2d = grid_2d(width, height, output_range=[0.0, width, 0.0, height])
        pos_2d = pos_2d.to(T_camera_pseudoCam.device)
        min_depth = torch.tensor([self.min_depth], device=T_camera_pseudoCam.device)[0]
        max_depth = torch.tensor([self.max_depth], device=T_camera_pseudoCam.device)[0]

        T_local_world = T_world_local.inverse()
        points3d = ray_points_snippet(
            pos_2d,
            camera,
            T_camera_pseudoCam,
            T_world_pseudoCam,
            T_local_world,
            self.num_samples,
            min_depth,
            max_depth,
        )
        points3d = einops.rearrange(
            points3d,
            "b t h w (n c) -> (b t) h w n c",
            b=B,
            t=T,
            h=height,
            w=width,
            n=self.num_samples,
        )
        points3d[..., 0] = (points3d[..., 0] - self.ray_points_scale[0]) / (
            self.ray_points_scale[1] - self.ray_points_scale[0]
        )
        points3d[..., 1] = (points3d[..., 1] - self.ray_points_scale[2]) / (
            self.ray_points_scale[3] - self.ray_points_scale[2]
        )
        points3d[..., 2] = (points3d[..., 2] - self.ray_points_scale[4]) / (
            self.ray_points_scale[5] - self.ray_points_scale[4]
        )
        points3d = inverse_sigmoid(points3d)
        points3d = einops.rearrange(
            points3d,
            "(b t) h w n c -> (b t) h w (n c)",
            b=B,
            t=T,
            h=height,
            w=width,
            n=self.num_samples,
        )

        encoding = self.encoder(points3d.contiguous())  # B x C x H x W
        encoding = einops.rearrange(
            encoding,
            "(b t) h w c -> b t c h w",
            b=B,
            t=T,
            h=height,
            w=width,
        )

        logger.debug(f"ray grid encoding {encoding.shape}")
        return encoding
