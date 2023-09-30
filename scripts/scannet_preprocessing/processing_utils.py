# Copyright (c) Meta Platforms, Inc. and affiliates.
import copy
import csv
import os
from io import BytesIO
from typing import Union

import cv2

import numpy as np
import torch

try:
    import quaternion  # @manual
except:
    import Quaternion as quaternion  # @manual


def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M


def get_homogeneous(
    pts: Union["np.ndarray", "torch.tensor"]
):
    """convert [(b), N, 3] pts to homogeneous coordinate
    Args:
        pts ([(b), N, 3] Union['np.ndarray', 'torch.tensor']): input point cloud
    Returns:
        homo_pts ([(b), N, 4] Union['np.ndarray', 'torch.tensor']): output point
            cloud
    Raises:
        ValueError: if the input tensor/array is not with the shape of [b, N, 3]
            or [N, 3]
        TypeError: if input is not either tensor or array
    """

    batch = False
    if len(pts.shape) == 3:
        pts = pts[0]
        batch = True
    elif len(pts.shape) == 2:
        pts = pts
    else:
        raise ValueError("only accept [b, n_pts, 3] or [n_pts, 3]")

    if isinstance(pts, torch.Tensor):
        ones = torch.ones_like(pts[:, 2:])
        homo_pts = torch.cat([pts, ones], axis=1)
        if batch:
            return homo_pts[None, :, :]
        else:
            return homo_pts
    elif isinstance(pts, np.ndarray):
        ones = np.ones_like(pts[:, 2:])
        homo_pts = np.concatenate([pts, ones], axis=1)
        if batch:
            return homo_pts[None, :, :]
        else:
            return homo_pts
    else:
        raise TypeError("wrong data type")


def get_corner_by_dims(dimensions):
    """get 8 corner points of 3D bbox defined by self.dimensions
    Returns:
        a np.ndarray with shape [8,3] to represent 8 corner points'
        position of the 3D bounding box.
    """

    l, h, w = dimensions[0], dimensions[1], dimensions[2]
    x_corners = [-l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2]
    y_corners = [-h / 2, -h / 2, h / 2, h / 2, -h / 2, -h / 2, h / 2, h / 2]
    z_corners = [-w / 2, -w / 2, -w / 2, -w / 2, w / 2, w / 2, w / 2, w / 2]
    corner_pts = np.array([x_corners, y_corners, z_corners], dtype=np.float32).T
    return corner_pts


def get_scannet_class_to_index():
    class_dict = {
        "chair": 0,
        "table": 1,
        "cabinet": 2,
        "trashbin": 3,
        "bookshelf": 4,
        "display": 5,
        "sofa": 6,
        "bathtub": 7,
        "bed": 8,
        "file cabinet": 9,
        "bag": 10,
        "printer": 11,
        "washer": 12,
        "lamp": 13,
        "microwave": 14,
        "stove": 15,
        "basket": 16,
        "bench": 17,
        "laptop": 18,
        "computer keyboard": 19,
        "other": 20,
    }
    return class_dict


def get_scannet_class_to_index_RayTran():
    # top 8
    class_dict = {
        "chair": 0,
        "table": 1,
        "cabinet": 2,
        "trashbin": 3,
        "bookshelf": 4,
        "display": 5,
        "sofa": 6,
        "bathtub": 7,
        "other": 8,
    }
    return class_dict


def get_point_cloud(depth_dir, intrinsics_depth):
    d = cv2.imread(depth_dir, -1).astype(np.float32)
    d /= 1000
    h, w = d.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xx = xx * d
    yy = yy * d
    pc = np.stack([xx, yy, d, np.ones_like(xx)], axis=2)
    pc = pc.reshape(-1, 4)
    T_camera_screen = np.linalg.inv(intrinsics_depth)

    T_camera_screen = torch.from_numpy(T_camera_screen).cuda()
    pc = torch.from_numpy(pc).cuda().float()

    pc = (T_camera_screen @ pc.T).T

    pos_z = torch.nonzero(pc[:, 2] > 0).squeeze(1)

    # pc = np.dot(T_camera_screen, pc.T).T
    # pos_z = np.nonzero(pc[:, 2] > 0)[0]

    point_cloud = pc[pos_z]
    return point_cloud[:, :3]


def get_catid_to_label_name(csv_filename):
    CARE_CLASSES = {
        "03211117": "display",
        "04379243": "table",
        "02808440": "bathtub",
        "02747177": "trashbin",
        "04256520": "sofa",
        "03001627": "chair",
        "02933112": "cabinet",
        "02871439": "bookshelf",
        "00000000": "other",
    }

    # catid2index = {}
    # path = os.path.join(csv_filename, "..", "scannetv2-labels.combined.tsv")
    # with path_manager.open(path) as csvfile:
    #     spamreader = csv.DictReader(csvfile, delimiter="\t")
    #     for row in spamreader:
    #         try:
    #             catid2index[row["wnsynsetid"][1:]] = (
    #                 int(row["nyu40id"]),
    #                 row["nyu40class"],
    #             )
    #         except:
    #             pass
    return CARE_CLASSES


def get_label(g_cad2scannet, catid_cads):
    labels = []
    for catid in catid_cads:
        try:
            name = g_cad2scannet[catid]
            labels.append(name)
        except KeyError:
            labels.append("other")
    return labels


def name2ids(labels, class_to_index):
    ids = []
    for label in labels:
        try:
            ids.append(class_to_index[label])
        except KeyError:
            ids.append(class_to_index["other"])
    return ids


def get_box3d_inside_fov(box3d_list, image_size, intrinsics_color):
    h, w, c = image_size
    T_screen_camera = torch.from_numpy(intrinsics_color).cuda()
    ratio_list = []
    ret_2d_box = []
    for ind, box3d in enumerate(box3d_list):
        ret_2d_box.append([])
        hom_box3d = torch.cat([box3d, torch.ones((8, 1)).cuda()], axis=1).float()
        projected_box3d = (T_screen_camera @ hom_box3d.T).T[:, :3]
        projected_box3d[:, 0] /= torch.maximum(
            projected_box3d[:, 2], torch.Tensor([1]).cuda()
        )
        projected_box3d[:, 1] /= torch.maximum(
            projected_box3d[:, 2], torch.Tensor([1]).cuda()
        )
        xmin, xmax = torch.min(projected_box3d[:, 0]), torch.max(projected_box3d[:, 0])
        ymin, ymax = torch.min(projected_box3d[:, 1]), torch.max(projected_box3d[:, 1])
        area = (xmax - xmin) * (ymax - ymin)
        cliped_xmin, cliped_xmax = torch.clip(xmin, 0, w - 1), torch.clip(
            xmax, 0, w - 1
        )
        cliped_ymin, cliped_ymax = torch.clip(ymin, 0, h - 1), torch.clip(
            ymax, 0, h - 1
        )
        ret_2d_box[ind] = [cliped_xmin, cliped_ymin, cliped_xmax, cliped_ymax]
        area_inside = (cliped_xmax - cliped_xmin) * (cliped_ymax - cliped_ymin)
        ratio = 1.0 * area_inside / torch.maximum(area, torch.Tensor([1]).cuda())
        ratio_list.append(ratio)
    return torch.cat(ratio_list), ret_2d_box


def get_point_cloud_inside_box3d(bbox3, pc):
    """
    Find point cloud inside a bounding box.
    :param bbox3:
    :param pc:
    :return:
    """
    point = copy.deepcopy(pc).unsqueeze(0)
    v45 = bbox3[:, 5] - bbox3[:, 4]
    v40 = bbox3[:, 0] - bbox3[:, 4]
    v47 = bbox3[:, 7] - bbox3[:, 4]
    point = point - bbox3[:, 4:5]
    m0 = torch.matmul(point, v45.unsqueeze(-1))
    m1 = torch.matmul(point, v40.unsqueeze(-1))
    m2 = torch.matmul(point, v47.unsqueeze(-1))
    cs = []
    for m, v in zip([m0, m1, m2], [v45, v40, v47]):
        c0 = 0 < m
        c1 = m < v.unsqueeze(1) @ v.unsqueeze(-1)
        c = c0 & c1
        cs.append(c)

    cs = cs[0] & cs[1] & cs[2]
    cs = cs.squeeze(-1)

    inside_number = cs.sum(-1)
    return inside_number, cs


def collate_fn(list_data):
    cam_pose, depth_im, _ = list_data
    # Concatenate all lists
    return cam_pose, depth_im, _


class ScanNetDataset(torch.utils.data.Dataset):
    """Pytorch Dataset for a single scene. getitem loads individual frames"""

    def __init__(self, n_imgs, scene, data_path, max_depth=3.0, id_list=None):
        """
        Args:
        """
        self.n_imgs = n_imgs
        self.scene = scene
        self.data_path = data_path
        self.max_depth = max_depth
        if id_list is None:
            self.id_list = [i for i in range(n_imgs)]
        else:
            self.id_list = id_list

    def __len__(self):
        return self.n_imgs

    def __getitem__(self, id):
        """
        Returns:
            dict of meta data and images for a single frame
        """
        id = self.id_list[id]
        path = os.path.join(
            self.data_path, self.scene, "pose", 'frame-{0:06d}.pose.txt'.format(id)
        )
        cam_pose = np.loadtxt(path, delimiter=" ")
        return cam_pose, None, None


def get_level(pc, trunc):
    difficulty_level = [
        {
            "point_cloud_num": 1000,
            "truncation_ratio": 0.85,
        },
        {
            "point_cloud_num": 500,
            "truncation_ratio": 0.70,
        },
        {
            "point_cloud_num": 100,
            "truncation_ratio": 0.50,
        },
    ]

    if (
        pc > difficulty_level[0]["point_cloud_num"]
        and trunc > difficulty_level[0]["truncation_ratio"]
    ):
        return 0
    elif (
        pc > difficulty_level[1]["point_cloud_num"]
        and trunc > difficulty_level[1]["truncation_ratio"]
    ):
        return 1
    elif (
        pc > difficulty_level[2]["point_cloud_num"]
        and trunc > difficulty_level[2]["truncation_ratio"]
    ):
        return 2
    else:
        return 3


def decode_image(value: Union[torch.ByteTensor, BytesIO]):
    """
    Decode a image from ByteTensor loaded by koski
        It supports all opencv format (e.g., 16-bit depth image) and keeps
        original image format.
    """
    if isinstance(value, torch.ByteTensor):
        value = value.numpy().view()
    else:
        value = np.asarray(bytearray(value.read()), dtype=np.uint8)
    return cv2.imdecode(value, cv2.IMREAD_UNCHANGED)


def view_selection(args, scene, cam_pose_list):
    all_ids = []
    ids = []
    count = 0
    last_pose = None
    for id in cam_pose_list.keys():
        cam_pose = cam_pose_list[id]

        if count == 0:
            ids.append(id)
            last_pose = cam_pose
            count += 1
        else:
            angle = np.arccos(
                (
                    (
                        np.linalg.inv(cam_pose[:3, :3])
                        @ last_pose[:3, :3]
                        @ np.array([0, 0, 1]).T
                    )
                    * np.array([0, 0, 1])
                ).sum()
            )
            dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
            if angle > (args.min_angle / 180) * np.pi or dis > args.min_distance:
                ids.append(id)
                last_pose = cam_pose
                count += 1
                if count == args.window_size:
                    all_ids.append(ids)
                    ids = []
                    count = 0
    return all_ids


def view_selection_w1(args, scene, cam_pose_list):
    all_ids = []
    ids = []
    count = 0
    last_pose = None
    last_id = 0
    for id in cam_pose_list.keys():
        cam_pose = cam_pose_list[id]
        last_id = id
        if count == 0:
            ids.append(id)
            last_pose = cam_pose
            count += 1
        else:
            angle = np.arccos(
                (
                    (
                        np.linalg.inv(cam_pose[:3, :3])
                        @ last_pose[:3, :3]
                        @ np.array([0, 0, 1]).T
                    )
                    * np.array([0, 0, 1])
                ).sum()
            )
            dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
            if angle > (args.min_angle / 180) * np.pi or dis > args.min_distance:
                ids.append(id)
                last_pose = cam_pose
                count += 1
    
    for id in ids:
        all_ids.append([id])
    return all_ids


def view_selection_overlap(args, scene, cam_pose_list):
    all_ids = []
    ids = []
    count = 0
    last_pose = None
    last_id = 0
    for id in cam_pose_list.keys():
        cam_pose = cam_pose_list[id]
        last_id = id
        if count == 0:
            ids.append(id)
            last_pose = cam_pose
            count += 1
        else:
            angle = np.arccos(
                (
                    (
                        np.linalg.inv(cam_pose[:3, :3])
                        @ last_pose[:3, :3]
                        @ np.array([0, 0, 1]).T
                    )
                    * np.array([0, 0, 1])
                ).sum()
            )
            dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
            if angle > (args.min_angle / 180) * np.pi or dis > args.min_distance:
                ids.append(id)
                last_pose = cam_pose
                count += 1

    for i in range(10):
        for j in range(count):
            if j + args.window_size <= count:
                ids_snippet = ids[j : j + args.window_size]
                if ids_snippet[-1] + i <= last_id:
                    ids_snippet = [
                        IS + i for IS in ids_snippet if IS + i in cam_pose_list.keys()
                    ]
                    if len(ids_snippet) == args.window_size:
                        all_ids.append(ids_snippet)

    # remove repeated data
    ids_iden = []
    for ids in all_ids:
        if ids not in ids_iden:
            ids_iden.append(ids)

    return ids_iden


def view_selection_allframes(args, scene, cam_pose_list):
    all_ids = []
    ids = []
    count = 0
    last_pose = None
    for id in cam_pose_list.keys():
        cam_pose = cam_pose_list[id]

        if count == 0:
            ids.append(id)
            last_pose = cam_pose
            count += 1
        else:
            angle = np.arccos(
                (
                    (
                        np.linalg.inv(cam_pose[:3, :3])
                        @ last_pose[:3, :3]
                        @ np.array([0, 0, 1]).T
                    )
                    * np.array([0, 0, 1])
                ).sum()
            )
            dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
            if angle > (args.min_angle / 180) * np.pi or dis > args.min_distance:
                ids.append(id)
                last_pose = cam_pose
                count += 1
                # if count == args.window_size:
                #     all_ids.append(ids)
                #     ids = []
                #     count = 0
    all_ids.append(ids)
    return all_ids


def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret
