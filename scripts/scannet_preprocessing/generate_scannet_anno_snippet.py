# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import glob
import os
import pickle

import numpy as np
import torch
from PIL import Image
import ray
import torch.multiprocessing

from processing_utils import (
    collate_fn,
    get_box3d_inside_fov,
    get_catid_to_label_name,
    get_homogeneous,
    get_label,
    get_level,
    get_point_cloud,
    get_point_cloud_inside_box3d,
    get_scannet_class_to_index_RayTran,
    name2ids,
    ScanNetDataset,
    split_list,
    view_selection,
    view_selection_overlap,
    view_selection_allframes,
    view_selection_w1,
)

torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scannet_path",
        help="the data path of the scannet dataset",
        default="/work/vig/Datasets/ScanNet/scans_uncomp",
    )
    parser.add_argument(
        "--out_filename",
        help="the data path of the output file",
        default="/work/vig/Datasets/ScanNet/deper_scannet_anno/scan2cad_box3d_anno_view3",
    )
    parser.add_argument("--window_size", default=3, type=int)
    parser.add_argument("--min_angle", default=15, type=float)
    parser.add_argument("--min_distance", default=0.1, type=float)
    parser.add_argument("--all_frames", action="store_true", help="if load all frames (keyframes)")
    
    # ray multi processes
    parser.add_argument('--n_proc', type=int, default=8, help='#processes launched to process scenes.')
    parser.add_argument('--n_gpu', type=int, default=1, help='#number of gpus')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--loader_num_workers', type=int, default=8)

    return parser.parse_args()


args = parse_args()


@ray.remote(num_cpus=args.num_workers + 1, num_gpus=(1 / args.n_proc))
def worker_thread(args, scenes):
    catid2name = get_catid_to_label_name(args.scannet_path)
    # read scannet validation scenes
    split_path = os.path.join(args.scannet_path, "..", "ScanNet", "Tasks", "Benchmark", "scannetv2_val.txt")
    scenes_val = open(split_path, "r").readlines()
    scenes_val = [l.strip() for l in scenes_val]

    for scene in scenes:
        if os.path.exists(
            os.path.join(args.out_filename, "image_anno_{}.pkl".format(scene))
        ):
            print(
                "{} exists, ignored".format(
                    os.path.join(args.out_filename, "image_anno_{}.pkl".format(scene))
                )
            )
            continue

        val = scene in scenes_val
        # if not val:
            # continue

        n_imgs = len(
            os.listdir(os.path.join(args.scannet_path, scene, "color"))
        )
        intrinsic_dir_depth = os.path.join(
            args.scannet_path,
            scene,
            "intrinsic",
            "intrinsic_depth.txt",
        )
        intrinsic_dir_color = os.path.join(
            args.scannet_path,
            scene,
            "intrinsic",
            "intrinsic_color.txt",
        )
        
        cam_intr_depth = np.loadtxt(intrinsic_dir_depth, delimiter=" ").astype(np.float32)

        cam_intr_color = np.loadtxt(intrinsic_dir_color, delimiter=" ").astype(np.float32)

        dataset = ScanNetDataset(n_imgs, scene, args.scannet_path)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            collate_fn=collate_fn,
            batch_sampler=None,
            num_workers=args.loader_num_workers,
        )

        cam_pose_all = {}
        for id, (cam_pose, _, _) in enumerate(dataloader):
            if id % 100 == 0:
                print("{}: read frame {}/{}".format(scene, str(id), str(n_imgs)))
            if (
                cam_pose[0][0] == np.inf
                or cam_pose[0][0] == -np.inf
                or cam_pose[0][0] == np.nan
            ):
                continue
            cam_pose_all.update({id: cam_pose})

        save_snippet_pkl(
            args,
            scene,
            cam_pose_all,
            cam_intr_depth,
            cam_intr_color,
            catid2name,
            val,
        )


def save_snippet_pkl(
    args, scene, cam_pose_list, cam_intr_depth, cam_intr_color, g_cad2scannet, val
):

    if scene.endswith("/"):
        scene = scene[:-1]

    # view selection
    print("segment: process scene {}".format(scene))
    if args.all_frames:
        all_ids = view_selection_allframes(args, scene, cam_pose_list)
    else:
        if val:
            if args.window_size == 1:
                all_ids = view_selection_w1(args, scene, cam_pose_list)
            else:
                all_ids = view_selection(args, scene, cam_pose_list)
        else:
            all_ids = view_selection_overlap(args, scene, cam_pose_list)

    try:
        with open(os.path.join(args.out_filename, scene + ".pkl"), "rb") as f:
            scene_anno_dict = pickle.load(f)
    except:
        # handle no oriented box for this scene
        return

    aligned_models = scene_anno_dict["aligned_models"]

    # camera coordinate
    catids = [_["catid_cad"] for _ in aligned_models]
    # catids -> cat name
    labels = get_label(g_cad2scannet, catids)
    # 20: others

    bboxes = [_["bboxes"] for _ in aligned_models]
    bbox_corners = [_["bbox_corners"] for _ in aligned_models]
    T_scan_object = [_["T_so"] for _ in aligned_models]
    sym = [_["sym"] for _ in aligned_models]
    bbox_corners = torch.from_numpy(np.array(bbox_corners)).cuda()

    roidb_scene = {
        "scene_name": scene,
        "bboxes": bboxes,
        "sym": sym,
        "T_scan_object": T_scan_object,
        "labels": labels,
        "snippets": [],
    }
    # save snippets
    for i, ids in enumerate(all_ids):
        img_shape = None

        inside_point_cloud_number_list = []
        trunc_ratio_list = []
        T_scan_camera_list = []
        cam_intr_color_list = []
        for id_ in ids:
            print("processing image {} scene {}".format(id_, scene))
            if img_shape is None:
                im_file = os.path.join(
                    args.scannet_path,
                    scene,
                    "color",
                    'frame-{0:06d}.color.jpg'.format(id_),
                )
                img = Image.open(im_file)
                img_shape = np.array(img).shape

            depth_file = os.path.join(
                args.scannet_path,
                scene,
                "depth",
                'frame-{0:06d}.depth.pgm'.format(id_),
            )
            T_scan_camera = cam_pose_list[id_]
            T_scan_camera_list.append(T_scan_camera)
            cam_intr_color_list.append(np.copy(cam_intr_color))

            if not args.all_frames:
                T_camera_scan = np.linalg.inv(T_scan_camera)

                T_camera_scan = torch.from_numpy(T_camera_scan).cuda()

                bboxes_camera = (
                    get_homogeneous(bbox_corners.view(-1, 3)) @ T_camera_scan.T
                )[:, :3].view(-1, 8, 3)

                # camera coordinate
                point_cloud = get_point_cloud(depth_file, cam_intr_depth)

                inside_point_cloud_number, cs_list = get_point_cloud_inside_box3d(
                    bboxes_camera, point_cloud
                )
                inside_point_cloud_number_list.append(inside_point_cloud_number)

                trunc_ratio, box2d_list = get_box3d_inside_fov(
                    bboxes_camera, img_shape, cam_intr_color
                )
                trunc_ratio_list.append(trunc_ratio)

        if args.all_frames:
            inside_point_cloud_number = trunc_ratio = None
        else:
            inside_point_cloud_number = torch.stack(inside_point_cloud_number_list).max(
                dim=0
            )[0]
            trunc_ratio = torch.stack(trunc_ratio_list).max(dim=0)[0]

        roidb_scene["snippets"].append(
            {
                "snippet_id": i,
                "image_ids": ids,
                "intrinsic": cam_intr_color_list,
                "T_scan_camera": T_scan_camera_list,
                "point_cloud_num_list": inside_point_cloud_number,
                "truncation_ratio_list": trunc_ratio,
            }
        )

    with open(
        os.path.join(args.out_filename, "image_anno_{}.pkl".format(scene)), "wb"
    ) as f:
        pickle.dump(roidb_scene, f)
    return


def get_roidb(args, split="train"):
    split_file = "scannetv2_train.txt" if split == "train" else "scannetv2_val.txt"
    print(split_file)
    split_path = os.path.join(args.scannet_path, "..", "ScanNet", "Tasks", "Benchmark", split_file)
    scenes = open(split_path, "r").readlines()
    scenes = [l.strip() for l in scenes]

    scannet_class_to_index = get_scannet_class_to_index_RayTran()

    roidb_filename = (
        os.path.join(args.out_filename, "scannet_train_gt_roidb.pkl")
        if "train" in split
        else os.path.join(args.out_filename, "scannet_val_gt_roidb.pkl")
    )
    scene_anno_path = os.path.join(args.out_filename, "scene_anno")
    if not os.path.exists(scene_anno_path):
        os.makedirs(scene_anno_path)
    # if os.path.exists(roidb_filename):
    #     item_list = pickle.load(open(roidb_filename, "rb"))
    #     print(
    #         "roidb loaded from {}, totally {} samples".format(
    #             roidb_filename, len(item_list)
    #         )
    #     )
    #     return item_list

    item_list = []
    roidb_filenames = glob.glob(os.path.join(args.out_filename, "image_anno*"))
    roidb_filenames.sort()
    print("total files: {}".format(len(roidb_filenames)))
    for roidb_file in roidb_filenames:
        with open(roidb_file, "rb") as f:
            roidb_scene = pickle.load(f)
        scene_name = roidb_scene["scene_name"]
        snippets = roidb_scene["snippets"]
        labels = roidb_scene["labels"]
        bboxes = roidb_scene["bboxes"]
        T_scan_object = roidb_scene["T_scan_object"]
        sym = roidb_scene["sym"]

        # cat name -> defined id (0, 1, 2, 3 ...)
        ids = name2ids(labels, scannet_class_to_index)
        print(labels)
        if scene_name not in scenes:
            continue
        print(scene_name)
        item_one_scene = {}
        for im_anno in snippets:
            point_cloud_num_list = im_anno["point_cloud_num_list"]
            trunc_ratio_list = im_anno["truncation_ratio_list"]

            num_obj = len(bboxes)
            valid_obj = []
            valid_difficulty = []
            for i in range(num_obj):
                if point_cloud_num_list is not None:
                    difficulty = get_level(
                        point_cloud_num_list[i],
                        trunc_ratio_list[i],
                    )
                    valid_difficulty.append(difficulty)
                    if difficulty >= 3:
                        continue
                valid_obj.append(i)

            if len(valid_obj) == 0:
                continue

            item_list.append(
                {
                    "scene_name": roidb_scene["scene_name"],
                    "snippet_id": im_anno["snippet_id"],
                }
            )

            item_one_scene[im_anno["snippet_id"]] = {
                "image_ids": im_anno["image_ids"],
                "T_scan_camera": im_anno["T_scan_camera"],
                "intrinsic": im_anno["intrinsic"],
                "annotations": {
                    "label": [ids[i] for i in valid_obj],
                    "bboxes": [bboxes[i] for i in valid_obj],
                    "sym": [sym[i] for i in valid_obj],
                    "T_scan_object": [T_scan_object[i] for i in valid_obj],
                    # "pc_num": [point_cloud_num_list[i] for i in valid_obj],
                    # "truncation": [trunc_ratio_list[i] for i in valid_obj],
                    # "difficulty": [valid_difficulty[i] for i in valid_obj],
                },
            }
        scene_save_path = os.path.join(scene_anno_path, scene_name + ".pkl")
        with open(scene_save_path, "wb") as fout:
            pickle.dump(item_one_scene, fout)

    with open(roidb_filename, "wb") as fout:
        pickle.dump(item_list, fout)
    print(
        "annotations written to file {}, totally {} samples".format(
            roidb_filename, len(item_list)
        )
    )
    return item_list


def main():
    all_proc = args.n_proc * args.n_gpu
    ray.init(num_cpus=all_proc * (args.num_workers + 1), num_gpus=args.n_gpu)
    
    # get scene name
    scenes = os.listdir(args.scannet_path)
    scenes.sort()

    # scenes = scenes[:1]
    # scenes = ["scene0599_02"] + scenes
    # worker_thread(args, scenes)
    print("totally {} scenes".format(len(scenes)))

    scenes = split_list(scenes, all_proc)

    ray_worker_ids = []
    for w_idx in range(all_proc):
        ray_worker_ids.append(worker_thread.remote(args, scenes[w_idx]))
        
    results = ray.get(ray_worker_ids)
    
    print("generate train data")
    get_roidb(args, split="train")

    print("generate val data")
    get_roidb(args, split="val")


if __name__ == "__main__":
    main()
