# Copyright (c) Meta Platforms, Inc. and affiliates.
# This file is derived from [ODAM](https://github.com/likojack/ODAM/blob/main/src/utils/eval_utils.py 
# and https://github.com/likojack/ODAM/blob/main/src/utils/box_utils.py).
# Originating Author: Kejie Li

# Original header:
# MIT License Copyright (c) 2022 kejieli

from copy import deepcopy

import numpy as np

import numpy as np

import torch
from numba import jit

# from utils import load_list_from_folder, fileparts, mkdir_if_missing
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment as linear_assignment


CARE_CLASSES = {
    0: "chair",
    1: "table",
    2: "cabinet",
    3: "trash bin",
    4: "bookshelf",
    5: "display",
    6: "sofa",
    7: "bathtub",
    8: "other",
}


def match_sequence(
    total_gts,
    total_preds,
    total_tps,
    predictions,
    gts,
    threshold,
    scan_id,
):
    used_gts = []
    for gt in gts:
        total_gts[gt[0]] += 1
    for prediction in predictions:
        pred_class = prediction[0]
        pred_bbx = prediction[1]
        total_preds[pred_class] += 1
        for i, gt in enumerate(gts):
            gt_class = gt[0]
            gt_bbx = gt[1]
            if gt_class == pred_class:
                rotx_matrix = rotx(np.pi / 2)
                pred_bbx_rot = (rotx_matrix @ pred_bbx[[4, 0, 1, 5, 7, 3, 2, 6]].T).T
                gt_bbx_rot = (rotx_matrix @ gt_bbx[[4, 0, 1, 5, 7, 3, 2, 6]].T).T
                iou, _ = iou3d(gt_bbx_rot, pred_bbx_rot)
                if iou > threshold and i not in used_gts:
                    used_gts.append(i)
                    total_tps[pred_class] += 1


def rotx(t):
    """3D Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


@jit
def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def iou3d(corners1, corners2):
    """Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    """
    if np.isnan(corners1).any() or np.isnan(corners2).any():
        return 0, 0
    try:
        # corner points are in counter clockwise order
        rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
        rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
        area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
        area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
        inter, inter_area = convex_hull_intersection(rect1, rect2)
        iou_2d = inter_area / (area1 + area2 - inter_area)
        ymax = min(corners1[0, 1], corners2[0, 1])
        ymin = max(corners1[4, 1], corners2[4, 1])
        inter_vol = inter_area * max(0.0, ymax - ymin)
        vol1 = box3d_vol(corners1)
        vol2 = box3d_vol(corners2)
        iou = inter_vol / (vol1 + vol2 - inter_vol)
    except:
        print(np.isnan(corners1).any())
        print(np.isnan(corners2).any())
        return 0, 0
    return iou, iou_2d


@jit
def box3d_vol(corners):
    """corners: (8,3) no assumption on axis direction"""
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


@jit
def convex_hull_intersection(p1, p2):
    """Compute area of two convex hull's intersection area.
    p1,p2 are a list of (x,y) tuples of hull vertices.
    return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def polygon_clip(subjectPolygon, clipPolygon):
    """Clip a polygon with another polygon.
    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**
    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return outputList


def get_f1(gts, predictions, tps):
    total_gts = 0
    total_preds = 0
    total_tps = 0
    for c in CARE_CLASSES:
        if predictions[c] == 0:
            continue
        if gts[c] == 0:
            accu = 0
        else:
            accu = tps[c] / predictions[c]
        if gts[c] == 0:
            recall = 0
        else:
            recall = tps[c] / gts[c]
        print("class {}:".format(CARE_CLASSES[c]))
        print("accuracy: {}".format(accu))
        print("recall: {}".format(recall))
        f1 = 2 * accu * recall / (accu + recall) if accu + recall != 0 else 0
        print("F1: {}".format(f1))
        total_gts += gts[c]
        total_preds += predictions[c]
        total_tps += tps[c]
    if total_preds != 0:
        accuracy = total_tps / total_preds
    else:
        accuracy = 0
    if total_gts != 0:
        recall = total_tps / total_gts
    else:
        recall = 0
    if (accuracy + recall) != 0:
        f1 = 2 * accuracy * recall / (accuracy + recall)
    else:
        f1 = 0
    print("average accuracy: {}, recall: {}, F1: {}".format(accuracy, recall, f1))
    print("------------")
    return accuracy, recall, f1


def to_box(size, center, rotation):
    x_min, x_max, y_min, y_max, z_min, z_max = size
    c3o = np.stack(
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
        axis=-1,
    )
    c3o = c3o.reshape(*c3o.shape[:-1], 8, 3)
    c3o = c3o @ rotation.transpose(-1, -2) + center[np.newaxis]
    return c3o


class F1Calculator(object):
    """Calculating f1"""

    def __init__(
        self,
        conf_thresh,
        f1_iou_thresh=[0.25, 0.5, 0.7],
    ):
        """
        """
        self.f1_iou_thresh = f1_iou_thresh
        self.conf_thresh = conf_thresh
        self.iou_thresh = 0.1
        self.score_keep_diff_class = 0.9
        self.iou_average = 0.5
        self.num_trks_keep = 4
        self.preds = {}
        self.gts = {}
        self.pred_image_level = {}
        # iou computation assume up direction is negative Y
        self.rotx_matrix = rotx(np.pi / 2)
        self.reset()

    def step(self, outputs, gt_list):
        """
        """
        batch_pred_map_cls = self.parse_predictions(
            outputs,
            self.conf_thresh,
        )

        gts = self.make_gt_list(gt_list)

        batch_pred_map_cls = self.matching_pred(
            batch_pred_map_cls, outputs["scene_name"]
        )

        gts = self.matching_gt(gts, outputs["scene_name"])

    def matching_pred(self, detections, scene_names):
        """
        """
        for i, dets in enumerate(detections):
            scene_name = scene_names[i]
            if scene_name not in self.preds.keys():
                for trk_id in range(len(dets)):
                    dets[trk_id][-1] = trk_id
                self.preds[scene_name] = deepcopy(dets)
            else:
                trks = self.preds[scene_name]
                iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)

                for d, det in enumerate(dets):
                    for t, trk in enumerate(trks):
                        det_rot = (
                            self.rotx_matrix @ det[1][[4, 0, 1, 5, 7, 3, 2, 6]].T
                        ).T
                        trk_rot = (
                            self.rotx_matrix @ trk[1][[4, 0, 1, 5, 7, 3, 2, 6]].T
                        ).T
                        iou_matrix[d, t] = iou3d(det_rot, trk_rot)[
                            0
                        ]  # det: 8 x 3, trk: 8 x 3

                matched_indices = linear_assignment(
                    -iou_matrix
                )  # hougarian algorithm
                matched_indices = np.stack(matched_indices, axis=1)

                unmatched_detections = []
                for d, det in enumerate(dets):
                    if d not in matched_indices[:, 0]:
                        unmatched_detections.append(d)
                unmatched_trackers = []
                for t, trk in enumerate(trks):
                    if t not in matched_indices[:, 1]:
                        unmatched_trackers.append(t)

                # filter out matched with low IOU
                matches = []
                for m in matched_indices:
                    if iou_matrix[m[0], m[1]] < self.iou_thresh:
                        unmatched_detections.append(m[0])
                        unmatched_trackers.append(m[1])
                    else:
                        matches.append(m.reshape(1, 2))

                for match in matches:
                    dets[match[:, 0].item()][-1] = trks[match[:, 1].item()][-1]
                    if trks[match[:, 1].item()][2] < dets[match[:, 0].item()][2]:
                        trks[match[:, 1].item()] = dets[match[:, 0].item()]

                pre_trk_num = len(self.preds[scene_name])
                # add new detections
                for trk_id, um in enumerate(unmatched_detections):
                    dets[um][-1] = trk_id + pre_trk_num
                    trks.append(dets[um])
                self.preds[scene_name] = deepcopy(trks)
        return detections

    def make_gt_list(self, gt_list):
        batch_gt_map_cls = []
        for gt_dict_b in gt_list:
            gt_sem_cls_labels = gt_dict_b["labels"].data.cpu().numpy()
            gt_corners = gt_dict_b["gt_corners_world"].data.cpu().numpy()

            batch_gt_map_cls.append(
                [
                    (
                        gt_sem_cls_labels[j].item(),
                        gt_corners[j] + np.random.randn(1) * 0.001,
                        1, # score, placeholder
                    )
                    for j in range(gt_corners.shape[0])
                ]
            )
        return batch_gt_map_cls

    def parse_predictions(
        self,
        outputs,
        conf_thresh,
    ):
        """Parse predictions
        Args:
        Returns:
        """
        pred_corners = outputs["pred_corners_world"]
        sem_cls_probs = outputs["sem_cls_prob"]
        pred_mask = outputs["pred_mask"]

        pred_sem_cls_prob, pred_sem_cls = torch.max(sem_cls_probs, -1)
        scores = pred_sem_cls_prob

        pred_sem_cls = pred_sem_cls.data.cpu().numpy()
        pred_corners = pred_corners.data.cpu().numpy()
        pred_sem_cls_prob = pred_sem_cls_prob.data.cpu().numpy()
        scores = scores.data.cpu().numpy()

        bsize = pred_corners.shape[0]

        batch_pred_map_cls = (
            []
        ) 
        for i in range(bsize):
            batch_pred_map_cls.append(
                [
                    [
                        pred_sem_cls[i, j].item(),
                        pred_corners[i, j],
                        scores[i, j],
                        # trk_id, always last one
                        -1,
                    ]
                    for j in range(pred_corners.shape[1])
                    if pred_sem_cls[i, j].item() != 9
                    and scores[i, j] > conf_thresh
                    and pred_mask[i, j] == 1
                ]
            )
        return batch_pred_map_cls

    def matching_gt(self, gts, scene_names):
        """
        """
        gts_snippet = []
        for i, dets in enumerate(gts):
            scene_name = scene_names[i]
            if scene_name not in self.gts.keys():
                self.gts[scene_name] = dets
                gts_snippet.append(deepcopy(dets))
            else:
                trks = self.gts[scene_name]
                iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)

                for d, det in enumerate(dets):
                    for t, trk in enumerate(trks):
                        det_rot = (
                            self.rotx_matrix @ det[1][[4, 0, 1, 5, 7, 3, 2, 6]].T
                        ).T
                        trk_rot = (
                            self.rotx_matrix @ trk[1][[4, 0, 1, 5, 7, 3, 2, 6]].T
                        ).T
                        iou_matrix[d, t] = iou3d(det_rot, trk_rot)[
                            0
                        ]  # det: 8 x 3, trk: 8 x 3

                matched_indices = linear_assignment(-iou_matrix)  # hougarian algorithm
                matched_indices = np.stack(matched_indices, axis=1)

                unmatched_detections = []
                for d, det in enumerate(dets):
                    if d not in matched_indices[:, 0]:
                        unmatched_detections.append(d)
                unmatched_trackers = []
                for t, trk in enumerate(trks):
                    if t not in matched_indices[:, 1]:
                        unmatched_trackers.append(t)

                # filter out matched with low IOU
                matches = []
                for m in matched_indices:
                    if iou_matrix[m[0], m[1]] < self.iou_thresh:
                        unmatched_detections.append(m[0])
                        unmatched_trackers.append(m[1])
                    else:
                        matches.append(m.reshape(1, 2))

                for match in matches:
                    if trks[match[:, 1].item()][2] < dets[match[:, 0].item()][2]:
                        trks[match[:, 1].item()] = dets[match[:, 0].item()]

                # add new detections
                for um in unmatched_detections:
                    trks.append(dets[um])
                self.gts[scene_name] = trks
                gts_snippet.append(deepcopy(trks))
        return gts_snippet

    def compute_metrics(self):
        """Use accumulated predictions and groundtruths to compute Average Precision."""

        scenes = self.preds.keys()
        metrics = {}
        for threshold in self.f1_iou_thresh:
            total_gts = {k: 0 for k in CARE_CLASSES}
            total_preds = {k: 0 for k in CARE_CLASSES}
            total_tps = {k: 0 for k in CARE_CLASSES}

            for scene in scenes:
                preds = self.preds[scene]
                match_sequence(
                    total_gts,
                    total_preds,
                    total_tps,
                    preds,
                    self.gts[scene],
                    threshold,
                    scene,
                )
            print(total_gts)
            print(total_preds)
            print(total_tps)
            print("----------")
            accuracy, recall, f1 = get_f1(total_gts, total_preds, total_tps)
            metrics["{}_accuracy".format(threshold)] = accuracy
            metrics["{}_recall".format(threshold)] = recall
            metrics["{}_f1".format(threshold)] = f1
        return metrics

    def __str__(self):
        overall_ret = self.compute_metrics()
        return self.metrics_to_str(overall_ret)

    def metrics_to_str(self, overall_ret, per_class=True):
        mAP_strs = []
        AR_strs = []
        per_class_metrics = []
        for ap_iou_thresh in self.ap_iou_thresh:
            mAP = overall_ret[ap_iou_thresh]["mAP"] * 100
            mAP_strs.append(f"{mAP:.2f}")
            ar = overall_ret[ap_iou_thresh]["AR"] * 100
            AR_strs.append(f"{ar:.2f}")

            if per_class:
                # per-class metrics
                per_class_metrics.append("-" * 5)
                per_class_metrics.append(f"IOU Thresh={ap_iou_thresh}")
                for x in list(overall_ret[ap_iou_thresh].keys()):
                    if x == "mAP" or x == "AR":
                        pass
                    else:
                        met_str = f"{x}: {overall_ret[ap_iou_thresh][x]*100:.2f}"
                        per_class_metrics.append(met_str)

        ap_header = [f"mAP{x:.2f}" for x in self.ap_iou_thresh]
        ap_str = ", ".join(ap_header)
        ap_str += ": " + ", ".join(mAP_strs)
        ap_str += "\n"

        ar_header = [f"AR{x:.2f}" for x in self.ap_iou_thresh]
        ap_str += ", ".join(ar_header)
        ap_str += ": " + ", ".join(AR_strs)

        if per_class:
            per_class_metrics = "\n".join(per_class_metrics)
            ap_str += "\n"
            ap_str += per_class_metrics

        return ap_str

    def metrics_to_dict(self, overall_ret):
        metrics_dict = {}
        for ap_iou_thresh in self.ap_iou_thresh:
            metrics_dict[f"mAP_{ap_iou_thresh}"] = (
                overall_ret[ap_iou_thresh]["mAP"] * 100
            )
            metrics_dict[f"AR_{ap_iou_thresh}"] = overall_ret[ap_iou_thresh]["AR"] * 100
        return metrics_dict

    def reset(self):
        self.preds = {}
        self.gts = {}
        self.scan_cnt = 0
