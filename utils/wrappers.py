# Copyright (c) Meta Platforms, Inc. and its affiliates. All Rights Reserved
# This file is derived from [Pixloc](https://github.com/cvg/pixloc).
# Originating Author: Paul-Edouard Sarlin
# Modified for [PARQ] by Yiming Xie.

# Original header:
# Copyright 2021 Paul-Edouard Sarlin

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Union

import torch

import functools
import inspect
import numpy as np
import math
from torch.utils.data._utils.collate import (default_collate_err_msg_format,
                                             np_str_obj_array_pattern)
from torch._six import string_classes
import collections


# https://github.com/cvg/pixloc/blob/master/pixloc/pixlib/datasets/base_dataset.py#L44
def collate(batch):
    """Difference with PyTorch default_collate: it can stack of other objects.
    """
    if not isinstance(batch, list):  # no batching
        return batch
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    else:
        # try to stack anyway in case the object implements stacking.
        return torch.stack(batch, 0)
    

# https://github.com/cvg/pixloc/blob/master/pixloc/pixlib/geometry/wrappers.py#L17
def autocast(func):
    """Cast the inputs of a TensorWrapper method to PyTorch tensors
       if they are numpy arrays. Use the device and dtype of the wrapper.
    """
    @functools.wraps(func)
    def wrap(self, *args):
        device = torch.device('cpu')
        dtype = None
        if isinstance(self, TensorWrapper):
            if self._data is not None:
                device = self.device
                dtype = self.dtype
        elif not inspect.isclass(self) or not issubclass(self, TensorWrapper):
            raise ValueError(self)

        cast_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
                arg = arg.to(device=device, dtype=dtype)
            cast_args.append(arg)
        return func(self, *cast_args)

    return wrap


# https://github.com/cvg/pixloc/blob/master/pixloc/pixlib/geometry/wrappers.py#L43
# Modified for [PARQ] by Yiming Xie: add squeeze, unsqueeze, clone, view
class TensorWrapper:
    _data = None

    @autocast
    def __init__(self, data: torch.Tensor):
        self._data = data

    @property
    def shape(self):
        return self._data.shape[:-1]

    @property
    def device(self):
        return self._data.device
    
    @property
    def ndim(self):
        return self._data.ndim
    
    @property
    def dtype(self):
        return self._data.dtype

    def __getitem__(self, index):
        return self.__class__(self._data[index])

    def __setitem__(self, index, item):
        self._data[index] = item.data

    def to(self, *args, **kwargs):
        return self.__class__(self._data.to(*args, **kwargs))

    def cpu(self):
        return self.__class__(self._data.cpu())

    def cuda(self):
        return self.__class__(self._data.cuda())

    def pin_memory(self):
        return self.__class__(self._data.pin_memory())

    def float(self):
        return self.__class__(self._data.float())

    def double(self):
        return self.__class__(self._data.double())

    def detach(self):
        return self.__class__(self._data.detach())
    
    def squeeze(self, dim=None):
        assert dim != -1 and dim != self._data.dim() - 1
        return self.__class__(self._data.squeeze(dim=dim))

    def unsqueeze(self, dim=None):
        assert dim != -1 and dim != self._data.dim()
        return self.__class__(self._data.unsqueeze(dim=dim))

    def clone(self):
        return self.__class__(self._data.clone())
    
    def view(self, *shape):
        assert shape[-1] == -1 or shape[-1] == self._data.shape[-1]
        return self.__class__(self._data.view(*shape))
    
    @classmethod
    def stack(cls, objects: List, dim=0, *, out=None):
        data = torch.stack([obj._data for obj in objects], dim=dim, out=out)
        return cls(data)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.stack:
            return self.stack(*args, **kwargs)
        else:
            return NotImplemented


# https://github.com/cvg/pixloc/blob/master/pixloc/pixlib/geometry/wrappers.py#L103
class Pose(TensorWrapper):
    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] == 12
        super().__init__(data)

    @classmethod
    @autocast
    def from_Rt(cls, R: torch.Tensor, t: torch.Tensor):
        '''Pose from a rotation matrix and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            R: rotation matrix with shape (..., 3, 3).
            t: translation vector with shape (..., 3).
        '''
        assert R.shape[-2:] == (3, 3)
        assert t.shape[-1] == 3
        assert R.shape[:-2] == t.shape[:-1]
        data = torch.cat([R.flatten(start_dim=-2), t], -1)
        return cls(data)

    @classmethod
    def from_4x4mat(cls, T: torch.Tensor):
        '''Pose from an SE(3) transformation matrix.
        Args:
            T: transformation matrix with shape (..., 4, 4).
        '''
        assert T.shape[-2:] == (4, 4)
        R, t = T[..., :3, :3], T[..., :3, 3]
        return cls.from_Rt(R, t)

    @property
    def R(self) -> torch.Tensor:
        '''Underlying rotation matrix with shape (..., 3, 3).'''
        rvec = self._data[..., :9]
        return rvec.reshape(rvec.shape[:-1]+(3, 3))

    @property
    def t(self) -> torch.Tensor:
        '''Underlying translation vector with shape (..., 3).'''
        return self._data[..., -3:]
    
    @property
    def matrix(self) -> torch.Tensor:
        """Underlying transformation matrix with shape (..., 4, 4)."""
        rvec = self._data[..., :9]
        rmat = rvec.reshape(rvec.shape[:-1] + (3, 3))
        tvec = self._data[..., -3:].unsqueeze(-1)
        T_3x4 = torch.cat([rmat, tvec], dim=-1)
        bot_row = T_3x4.new_zeros(T_3x4.shape[:-2] + (1, 4))
        bot_row[..., 0, 3] = 1
        return torch.cat([T_3x4, bot_row], dim=-2)

    def inverse(self) -> 'Pose':
        '''Invert an SE(3) pose.'''
        R = self.R.transpose(-1, -2)
        t = -(R @ self.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    def compose(self, other: 'Pose') -> 'Pose':
        '''Chain two SE(3) poses: T_B2C.compose(T_A2B) -> T_A2C.'''
        R = self.R @ other.R
        t = self.t + (self.R @ other.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    @autocast
    def transform(self, p3d: torch.Tensor) -> torch.Tensor:
        '''Transform a set of 3D points.
        Args:
            p3d: 3D points, numpy array or PyTorch tensor with shape (..., 3).
        '''
        assert p3d.shape[-1] == 3
        # assert p3d.shape[:-2] == self.shape  # allow broadcasting
        return p3d @ self.R.transpose(-1, -2) + self.t.unsqueeze(-2)

    def __mul__(self, p3D: torch.Tensor) -> torch.Tensor:
        '''Transform a set of 3D points: T_A2B * p3D_A -> p3D_B.'''
        return self.transform(p3D)

    def __matmul__(self, other: 'Pose') -> 'Pose':
        '''Chain two SE(3) poses: T_B2C @ T_A2B -> T_A2C.'''
        return self.compose(other)

    def numpy(self) -> Tuple[np.ndarray]:
        return self.R.numpy(), self.t.numpy()

    def magnitude(self) -> Tuple[torch.Tensor]:
        '''Magnitude of the SE(3) transformation.
        Returns:
            dr: rotation anngle in degrees.
            dt: translation distance in meters.
        '''
        trace = torch.diagonal(self.R, dim1=-1, dim2=-2).sum(-1)
        cos = torch.clamp((trace - 1) / 2, -1, 1)
        dr = torch.acos(cos).abs() / math.pi * 180
        dt = torch.norm(self.t, dim=-1)
        return dr, dt

    def __repr__(self):
        return f'Pose: {self.shape} {self.dtype} {self.device}'



class Obb3D(TensorWrapper):
    def __init__(self, data: torch.Tensor):
        '''
            bb3_object (..., 6): 3D bounding box [xmin,xmax,ymin,ymax,zmin,zmax] in object coord frame
            T_world_object (..., 12): 3D SE3 transform from object to world coords
            sem_id (..., 1): semantic id
        '''
        assert data.shape[-1] == 19
        super().__init__(data)

    @classmethod
    def separate_init(cls, bb3_object: torch.Tensor, T_world_object: Pose, sem_id: torch.Tensor):
        if sem_id.dim() != bb3_object.dim():
            sem_id = sem_id.unsqueeze(-1)
        data = torch.cat(
            [
                bb3_object, # ..., 6
                T_world_object, # ..., 12
                sem_id, # ..., 1
            ],
            dim=-1,
        )
        return cls(data)
    
    @property
    def bb3_object(self) -> torch.Tensor:
        """3D bounding box [xmin,xmax,ymin,ymax,zmin,zmax] in object coord frame, with shape (..., 6)."""
        return self._data[..., :6]

    @property
    def bb3_min_object(self) -> torch.Tensor:
        """3D bounding box minimum corner [xmin,ymin,zmin] in object coord frame, with shape (..., 3)."""
        return self._data[..., 0:6:2]

    @property
    def bb3_max_object(self) -> torch.Tensor:
        """3D bounding box maximum corner [xmax,ymax,zmax] in object coord frame, with shape (..., 3)."""
        return self._data[..., 1:6:2]

    @property
    def bb3_center_object(self) -> torch.Tensor:
        """3D bounding box center in object coord frame, with shape (..., 3)."""
        return 0.5 * (self.bb3_min_object + self.bb3_max_object)

    @property
    def bb3_size(self) -> torch.Tensor:
        """3D bounding box diangonal, with shape (..., 3)."""
        return self.bb3_max_object - self.bb3_min_object

    @property
    def T_world_object(self) -> torch.Tensor:
        """3D SE3 transform from object to world coords, with shape (..., 12)."""
        return Pose(self._data[..., 6:18])

    @property
    def sem_id(self) -> torch.Tensor:
        """semantic id, with shape (..., 1)."""
        return self._data[..., 18].unsqueeze(-1)

    @property
    def bb3corners_object(self) -> torch.Tensor:
        """return the 8 corners of the 3D BB in object coord frame (..., 8, 3)."""
        b3o = self.bb3_object
        x_min, x_max = b3o[..., 0], b3o[..., 1]
        y_min, y_max = b3o[..., 2], b3o[..., 3]
        z_min, z_max = b3o[..., 4], b3o[..., 5]
        c3o = torch.stack(
            [
                x_min,
                y_min,
                z_min,
                x_max,
                y_min,
                z_min,
                x_max,
                y_max,
                z_min,
                x_min,
                y_max,
                z_min,
                x_min,
                y_min,
                z_max,
                x_max,
                y_min,
                z_max,
                x_max,
                y_max,
                z_max,
                x_min,
                y_max,
                z_max,
            ],
            dim=-1,
        )
        c3o = c3o.reshape(*c3o.shape[:-1], 8, 3)
        return c3o

    def add_padding(self, max_box: int = 100) -> "Obb3D":
        """
        Adds padding to Obbs to make them all the same size. Returns a new Obb3D.
        """
        assert self._data.ndim <= 2
        boxes = self._data
        num_pad = max_box - len(boxes)
        # All -1's denotes a pad box.
        pad_box = -1 * self._data.new_ones(self._data.shape[-1])
        if num_pad > 0:
            rep_box = torch.stack([pad_box for _ in range(num_pad)], dim=0)
            boxes = torch.cat([boxes, rep_box], dim=0)
        elif num_pad < 0:
            boxes = boxes[:max_box]
        return self.__class__(boxes)

    def remove_padding(self) -> List["Obb3D"]:
        """
        Removes any padding by finding Obbs with all -1s. Returns a list.
        """
        assert self._data.ndim <= 3

        if self._data.ndim == 1:
            return self

        # All -1's denotes a pad box.
        pad_box = (-1 * self._data.new_ones(self._data.shape[-1])).unsqueeze(-2)
        not_pad = ~torch.all(self._data == pad_box, dim=-1)

        if self.ndim == 2:
            num_valid = not_pad.sum()
            new_data = self.__class__(self._data[:num_valid])
        else:
            B = self._data.shape[0]
            new_data = []
            for b in range(B):
                num_valid = not_pad[b].sum()
                new_data.append(self.__class__(self._data[b][:num_valid]))
        return new_data

    def __repr__(self):
        return f"Obb3D {self.shape} {self.dtype} {self.device}"


# https://github.com/cvg/pixloc/blob/master/pixloc/pixlib/geometry/wrappers.py#L224
# Modified for [PARQ] by Yiming Xie.
class Camera(TensorWrapper):
    eps = 1e-3

    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] in {6}
        super().__init__(data)

    @classmethod
    def separate_init(cls, fx, fy, cx, cy, width, height, dist=None):
        '''
        '''
        if dist is None:
            data = torch.cat([width, height, fx, fy, cx, cy], dim=-1)
        else:
            data = torch.cat([width, height, fx, fy, cx, cy, dist], dim=-1)
        return cls(data)

    @property
    def size(self) -> torch.Tensor:
        '''Size (width height) of the images, with shape (..., 2).'''
        return self._data[..., :2]

    @property
    def f(self) -> torch.Tensor:
        '''Focal lengths (fx, fy) with shape (..., 2).'''
        return self._data[..., 2:4]

    @property
    def c(self) -> torch.Tensor:
        '''Principal points (cx, cy) with shape (..., 2).'''
        return self._data[..., 4:6]

    @property
    def dist(self) -> torch.Tensor:
        '''Distortion parameters, with shape (..., {0, 2, 4}).'''
        return self._data[..., 6:]

    def scale(self, scales: Union[float, int, Tuple[Union[float, int]]]):
        '''Update the camera parameters after resizing an image.'''
        if isinstance(scales, (int, float)):
            scales = (scales, scales)
        s = self._data.new_tensor(scales)
        data = torch.cat([
            self.size*s,
            self.f*s,
            (self.c+0.5)*s-0.5,
            self.dist], -1)
        return self.__class__(data)

    def crop(self, left_top: Tuple[float], size: Tuple[int]):
        '''Update the camera parameters after cropping an image.'''
        left_top = self._data.new_tensor(left_top)
        size = self._data.new_tensor(size)
        data = torch.cat([
            size,
            self.f,
            self.c - left_top,
            self.dist], -1)
        return self.__class__(data)

    @autocast
    def in_image(self, p2d: torch.Tensor):
        '''Check if 2D points are within the image boundaries.'''
        assert p2d.shape[-1] == 2
        # assert p2d.shape[:-2] == self.shape  # allow broadcasting
        size = self.size.unsqueeze(-2)
        valid = torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)
        return valid

    @autocast
    def project(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Project 3D points into the camera plane and check for visibility.'''
        z = p3d[..., -1]
        in_front = z > self.eps
        z = z.clamp(min=self.eps)
        p2d = p3d[..., :-1] / z.unsqueeze(-1)
        
        p2d = p2d * self.f.unsqueeze(-2) + self.c.unsqueeze(-2)

        valid = in_front & self.in_image(p2d)

        return p2d, valid

    @autocast
    def unproject(self, uv: torch.Tensor) -> torch.Tensor:
        """
        Batched implementation of the Pinhole (aka Linear) camera
        model. 

        Inputs:
            uv: BxNx3 tensor of 2D pixels to be projected
            params: Bx4 tensor of Pinhole parameters formatted like this:
                    [f_u f_v c_u c_v]
        Outputs:
            xyz: BxNx3 tensor of 3D rays of uv points with z = 1.

        """

        assert uv.ndim == 3, "Expected batched input shaped BxNx3"
        B, N = uv.shape[0], uv.shape[1]

        # Focal length and principal point
        fx_fy = self.f.reshape(B, 1, 2)
        cx_cy = self.c.reshape(B, 1, 2)

        uv_dist = (uv - cx_cy) / fx_fy

        p3d = torch.cat([uv_dist, uv.new_ones(B, N, 1)], dim=2)

        return p3d

    def __repr__(self):
        return f'Camera {self.shape} {self.dtype} {self.device}'