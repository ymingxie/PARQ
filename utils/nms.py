"""
Copyright (c) Meta Platforms, Inc. and its affiliates. All Rights Reserved
This file is derived from [VoteNet](https://github.com/facebookresearch/votenet/blob/main/utils/nms.py).
Modified for [PARQ] by Yiming Xie.

Original header:
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
# boxes are axis aigned 2D boxes of shape (n,5) in FLOAT numbers with (x1,y1,x2,y2,score)
""" Ref: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
Ref: https://github.com/vickyboy47/nms-python/blob/master/nms.py
"""


def nms(
    obbs_pred, scores, num_semcls, overlap_threshold, nms_type="nms_3d_faster_samecls"
):
    # object to local
    pred_corners = obbs_pred.T_world_object.transform(obbs_pred.bb3corners_object)
    pred_corners = pred_corners.data.cpu().numpy()
    labels = obbs_pred.sem_id.data.cpu().numpy()
    scores = scores.data.cpu().numpy()
    pred_mask = run_nms(
        pred_corners, labels, scores, num_semcls, overlap_threshold, nms_type
    )
    return pred_mask


def run_nms(
    pred_corners,
    labels,
    scores,
    num_semcls,
    overlap_threshold,
    nms_type="nms_3d_faster_samecls",
):
    bs, K = pred_corners.shape[:2]
    # ---------- NMS input: pred_with_prob in (B,K,8) -----------
    pred_mask = np.zeros((bs, K)).astype(np.bool)
    for i in range(bs):
        boxes_3d_with_prob = np.zeros((K, 8))
        boxes_3d_with_prob[:, 0] = np.min(pred_corners[i, :, :, 0], axis=-1)
        boxes_3d_with_prob[:, 1] = np.min(pred_corners[i, :, :, 1], axis=-1)
        boxes_3d_with_prob[:, 2] = np.min(pred_corners[i, :, :, 2], axis=-1)
        boxes_3d_with_prob[:, 3] = np.max(pred_corners[i, :, :, 0], axis=-1)
        boxes_3d_with_prob[:, 4] = np.max(pred_corners[i, :, :, 1], axis=-1)
        boxes_3d_with_prob[:, 5] = np.max(pred_corners[i, :, :, 2], axis=-1)
        boxes_3d_with_prob[:, 6] = scores[i]
        boxes_3d_with_prob[:, 7] = labels[
            i, :, 0
        ]  # only suppress if the two boxes are of the same class!!
        bg_box_inds = np.where(labels[i, :, 0] != num_semcls)[0]
        if nms_type == "nms_3d_faster_samecls":
            pick = nms_3d_faster_samecls(
                boxes_3d_with_prob[bg_box_inds],
                overlap_threshold,
            )
        else:
            pick = nms_3d_faster(
                boxes_3d_with_prob[bg_box_inds],
                overlap_threshold,
            )
        pred_mask[i, bg_box_inds[pick]] = 1

    return pred_mask


def nms_2d(boxes, overlap_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    score = boxes[:, 4]
    area = (x2 - x1) * (y2 - y1)

    I = np.argsort(score)
    pick = []
    while I.size != 0:
        last = I.size
        i = I[-1]
        pick.append(i)
        suppress = [last - 1]
        for pos in range(last - 1):
            j = I[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            w = xx2 - xx1
            h = yy2 - yy1
            if w > 0 and h > 0:
                o = w * h / area[j]
                print("Overlap is", o)
                if o > overlap_threshold:
                    suppress.append(pos)
        I = np.delete(I, suppress)
    return pick


def nms_2d_faster(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    score = boxes[:, 4]
    area = (x2 - x1) * (y2 - y1)

    I = np.argsort(score)
    pick = []
    while I.size != 0:
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[: last - 1]])
        yy1 = np.maximum(y1[i], y1[I[: last - 1]])
        xx2 = np.minimum(x2[i], x2[I[: last - 1]])
        yy2 = np.minimum(y2[i], y2[I[: last - 1]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        if old_type:
            o = (w * h) / area[I[: last - 1]]
        else:
            inter = w * h
            o = inter / (area[i] + area[I[: last - 1]] - inter)

        I = np.delete(
            I, np.concatenate(([last - 1], np.where(o > overlap_threshold)[0]))
        )

    return pick


def nms_3d_faster(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    z1 = boxes[:, 2]
    x2 = boxes[:, 3]
    y2 = boxes[:, 4]
    z2 = boxes[:, 5]
    score = boxes[:, 6]
    area = (x2 - x1) * (y2 - y1) * (z2 - z1)

    I = np.argsort(score)
    pick = []
    while I.size != 0:
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[: last - 1]])
        yy1 = np.maximum(y1[i], y1[I[: last - 1]])
        zz1 = np.maximum(z1[i], z1[I[: last - 1]])
        xx2 = np.minimum(x2[i], x2[I[: last - 1]])
        yy2 = np.minimum(y2[i], y2[I[: last - 1]])
        zz2 = np.minimum(z2[i], z2[I[: last - 1]])

        l = np.maximum(0, xx2 - xx1)
        w = np.maximum(0, yy2 - yy1)
        h = np.maximum(0, zz2 - zz1)

        if old_type:
            o = (l * w * h) / area[I[: last - 1]]
        else:
            inter = l * w * h
            o = inter / (area[i] + area[I[: last - 1]] - inter)

        I = np.delete(
            I, np.concatenate(([last - 1], np.where(o > overlap_threshold)[0]))
        )

    return pick


def nms_3d_faster_samecls(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    z1 = boxes[:, 2]
    x2 = boxes[:, 3]
    y2 = boxes[:, 4]
    z2 = boxes[:, 5]
    score = boxes[:, 6]
    cls = boxes[:, 7]
    area = (x2 - x1) * (y2 - y1) * (z2 - z1)

    I = np.argsort(score)
    pick = []
    while I.size != 0:
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[: last - 1]])
        yy1 = np.maximum(y1[i], y1[I[: last - 1]])
        zz1 = np.maximum(z1[i], z1[I[: last - 1]])
        xx2 = np.minimum(x2[i], x2[I[: last - 1]])
        yy2 = np.minimum(y2[i], y2[I[: last - 1]])
        zz2 = np.minimum(z2[i], z2[I[: last - 1]])
        cls1 = cls[i]
        cls2 = cls[I[: last - 1]]

        l = np.maximum(0, xx2 - xx1)
        w = np.maximum(0, yy2 - yy1)
        h = np.maximum(0, zz2 - zz1)

        if old_type:
            o = (l * w * h) / area[I[: last - 1]]
        else:
            inter = l * w * h
            o = inter / (area[i] + area[I[: last - 1]] - inter)
        o = o * (cls1 == cls2)

        I = np.delete(
            I, np.concatenate(([last - 1], np.where(o > overlap_threshold)[0]))
        )

    return pick


if __name__ == "__main__":
    a = np.random.random((100, 5))
    print(nms_2d(a, 0.9))
    print(nms_2d_faster(a, 0.9))
