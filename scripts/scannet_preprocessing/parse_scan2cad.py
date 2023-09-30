# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import json
import os
import pickle

import numpy as np

from processing_utils import get_corner_by_dims, get_homogeneous, make_M_from_tqs


def generate_anno(args, full_anno):
    full_anno_list = []
    anno_num = len(full_anno)

    for j, anno in enumerate(full_anno):
        id_scan = anno["id_scan"]
        # if os.path.exists(os.path.join(args.out_filename, id_scan + ".pkl")):
        #     continue

        n_aligned_models = anno["n_aligned_models"]
        aligned_models = anno["aligned_models"]
        anno_dict = dict(
            id_scan=id_scan,
            n_aligned_models=n_aligned_models,
            aligned_models=[],
        )
        print(
            "processing bounding box {}, {} % {}, containing {} objs".format(
                id_scan, j, anno_num, n_aligned_models
            )
        )
        # T_world_scan
        T_ws = make_M_from_tqs(
            anno["trs"]["translation"],
            anno["trs"]["rotation"],
            anno["trs"]["scale"],
        )
        # T_scan_world
        T_sw = np.linalg.inv(T_ws)

        for i, model in enumerate(aligned_models):
            print("processing bounding box  {} % {}".format(i, n_aligned_models))
            t = model["trs"]["translation"]
            q = model["trs"]["rotation"]
            s = model["trs"]["scale"]

            sym = model["sym"]

            catid_cad = model["catid_cad"]
            id_cad = model["id_cad"]

            mat_off = np.eye(4)
            mat_off[:3, 3] = model["center"]

            if min(s) < 1e-3:
                continue
            scales = model["bbox"] * np.asarray(s) * 2
            T_wo = make_M_from_tqs(t, q, np.ones_like(s))
            T_so = T_sw @ T_wo @ mat_off
            bboxes = np.stack(
                [
                    -scales[0] / 2,
                    scales[0] / 2,
                    -scales[1] / 2,
                    scales[1] / 2,
                    -scales[2] / 2,
                    scales[2] / 2,
                ],
            )
            bbox_corners = get_corner_by_dims(scales)
            bbox_corners = (get_homogeneous(bbox_corners) @ T_so.T)[:, :3]

            model_box = dict(
                id_obj=i,
                catid_cad=catid_cad,
                id_cad=id_cad,
                bboxes=bboxes,
                bbox_corners=bbox_corners,
                T_so=T_so,
                sym=sym,
            )
            anno_dict["aligned_models"].append(model_box)

        with open(os.path.join(args.out_filename, id_scan + ".pkl"), "wb") as fout:
            pickle.dump(anno_dict, fout)
        print(
            "id_scan {} saved to {}".format(
                id_scan, os.path.join(args.out_filename, id_scan)
            )
        )
        full_anno_list.append(anno_dict)

    with open(
        os.path.join(args.out_filename, "scan2cad_bbox_3d_anno.pkl"), "wb"
    ) as fout:
        pickle.dump(full_anno_list, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan2cad_path",
        help="the data path of the scan2cad dataset",
        default="/work/vig/Datasets/ScanNet/scan2cad/full_annotations.json",
    )
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

    args = parser.parse_args()

    with open(args.scan2cad_path, "r") as f:
        full_anno = json.load(f)
    generate_anno(args, full_anno)
