"""
Copyright (c) Meta Platforms, Inc. and its affiliates. All Rights Reserved
This file is derived from [VoteNet](https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py).
Modified for [PARQ] by Yiming Xie.

Original header:
Copyright (c) 2019 Yi_Zhou

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

# the representation of rotation --> ortho6d
# https://github.com/papagina/RotationContinuity/blob/758b0ce551c06372cab7022d4c0bdf331c89c696/sanity_test/code/tools.py#L47
def rot_to_6d(rot_matrix):
    # batch x 3 x 3
    return torch.concat((rot_matrix[..., 0], rot_matrix[..., 1]), dim=-1)


# batch*n
def normalize_vector(v, return_mag=False):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(
        v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])).to(v.device)
    )
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if return_mag == True:
        return v, v_mag[:, 0]
    else:
        return v


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat(
        (i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1
    )  # batch*3

    return out


def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix
