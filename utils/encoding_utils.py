# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging

import einops

import torch

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.setLevel(logging.WARNING)
# logger.setLevel(logging.DEBUG)


def grid_2d(width: int, height: int, output_range=(-1.0, 1.0, -1.0, 1.0)):
    x = torch.linspace(output_range[0], output_range[1], width + 1)[:-1]
    y = torch.linspace(output_range[2], output_range[3], height + 1)[:-1]
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    grid = torch.stack([xx, yy], dim=-1)
    return grid


def ray_points_snippet(
    pixels: torch.Tensor,
    camera: torch.Tensor,
    T_camera_pseudoCam: torch.Tensor,
    T_world_pseudoCam: torch.Tensor,
    T_local_world: torch.Tensor,
    num_samples: int = 32,
    min_depth: float = 0.25,
    max_depth: float = 3.25,
):
    """
    ray positional encoding: sample points along the ray as positional encoding

    input:
        pixels: grid of rays [H x W x 2]
        T_camera_pseudoCam:   -
        T_world_pseudoCam:    -
        T_local_world:        -
        num_samples: number of points to sample along the ray
        min_depth: minimum depth of the ray
        max_depth: maximum depth of the ray

    output:
        points: [B x T x H x W x (num_samples x 3)] rays points in a consistent local coordinate frame
    """
    batch_size, time_snippet = T_camera_pseudoCam.shape[0], T_camera_pseudoCam.shape[1]

    # T_camera_pseudoCam = T_camera_pseudoCam.view(batch_size * time_snippet, 4, 4)
    # intrinsics = intrinsics.view(batch_size * time_snippet, 4, 4)
    T_camera_pseudoCam = T_camera_pseudoCam.view(batch_size * time_snippet, -1)
    camera = camera.view(batch_size * time_snippet, -1)

    points = ray_points(pixels, camera, T_camera_pseudoCam, num_samples, min_depth, max_depth)

    T_local_pseudoCam = T_local_world @ T_world_pseudoCam
    T_local_pseudoCam = T_local_pseudoCam.view(batch_size * time_snippet, -1)
    points = einops.rearrange(points, "b h w n c -> b (h w n) c")
    points = T_local_pseudoCam.transform(points)

    points = einops.rearrange(
        points,
        "(b t) (h w n) c -> b t h w (n c)",
        b=batch_size,
        t=time_snippet,
        h=pixels.shape[0],
        w=pixels.shape[1],
        n=num_samples,
    )
    return points


def ray_points(pixel_grid, camera, T_camera_pseudoCam, num_samples=32, min_depth=0.25, max_depth=3.25):
    grid_height, grid_width = pixel_grid.shape[0], pixel_grid.shape[1]
    batch_size = T_camera_pseudoCam.shape[0]
    pixel_grid = pixel_grid.reshape(-1, 2)
    pixel_grid = einops.repeat(pixel_grid, "n c -> b n c", b=batch_size)
    # unproject
    rays = camera.unproject(pixel_grid)

    linear_ramp = (
        torch.linspace(0, 1, num_samples).view(1, 1, num_samples, 1).to(rays.device)
    )
    log_depth_planes_bd11 = (
        torch.log(min_depth) + torch.log(max_depth / min_depth) * linear_ramp
    )
    depth_planes_bd11 = torch.exp(log_depth_planes_bd11)
    points = rays.unsqueeze(-2) * depth_planes_bd11

    points = points.view(batch_size, -1, 3)

    assert not torch.isnan(
        points
    ).any(), f"have {torch.isnan(points).count_nonzero().item()} nans in rays"
    T_pseudoCam_camera = T_camera_pseudoCam.inverse()
    points = T_pseudoCam_camera.transform(points)

    assert not torch.isnan(points).any()
    return points.view([batch_size, grid_height, grid_width, num_samples, -1])

