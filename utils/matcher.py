"""
Copyright (c) Meta Platforms, Inc. and its affiliates. All Rights Reserved
This file is derived from [DETR](https://github.com/facebookresearch/detr/blob/main/models/matcher.py).
Modified for [PARQ] by Yiming Xie.

Original header:
Copyright 2020 - present, Facebook, Inc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
        

class HungarianMatcherModified(nn.Module):
    """This is modified from the HungarianMatcher and Aside from Hungarian matching, we also match the GT box and 
    the predictions whose corresponding reference points are in close proximity to this GT box, since 
    for two adjacent reference points which have the similar queries, they should both detect nearby objects.
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, ratio=0.2, max_padding=10):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.ratio = ratio
        self.max_padding = max_padding
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0,  "all costs cant be 0"

    def forward(self, outputs, targets):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            out_prob = (
                outputs["pred_logits"].softmax(-1)
            )  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["coord_pos"] 

            indices = []
            punish_mask_list = []
            assert bs == len(targets)
            for batch_idx in range(bs):
                bz_boxes = out_bbox[batch_idx]
                bz_out_prob = out_prob[batch_idx]
                bz_tgt_ids = targets[batch_idx]["labels"]
                bz_tgt_center = targets[batch_idx]["center"]
                bz_tgt_corners = targets[batch_idx]["gt_corners"]
                num_insts = len(bz_tgt_ids)
                if num_insts == 0:  # empty object in key frame
                    non_valid = torch.zeros(bz_out_prob.shape[0]).to(bz_out_prob) > 0
                    indices_batchi = (non_valid, torch.arange(0, 0).to(bz_out_prob))
                    indices.append(indices_batchi)
                    continue
                
                # ---hungarian matching---
                # Compute the classification cost.
                cost_class = -bz_out_prob[:, bz_tgt_ids]
                cost_bbox = torch.cdist(bz_boxes, bz_tgt_center, p=1)
                cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class # + 100.0 * (~is_in_boxes_and_center)
                indices_batchi = linear_sum_assignment(cost.cpu())
                indices_batchi = list(indices_batchi)
                # -----------------------

                # ---match the GT box and the predictions whose corresponding reference points 
                # are in close proximity to this GT box---
                pred_indices = []
                gt_indices = []
                for j, box_j in enumerate(bz_tgt_corners):
                    inside_sph = cost_bbox[..., j] < self.ratio
                    pred_ind = torch.nonzero(inside_sph).squeeze(1).data.cpu().numpy()
                    punish_mask = torch.ones_like(inside_sph).bool()
                    punish_mask[pred_ind] = False
                    if pred_ind.shape[0] > self.max_padding:
                        choose = np.random.choice(pred_ind.shape[0], self.max_padding, replace=False)
                        pred_ind = pred_ind[choose]
                    punish_mask[pred_ind] = True
                    pred_indices.append(pred_ind)
                    gt_indices.append(np.ones_like(pred_ind) * j)
                pred_indices = np.concatenate(pred_indices)
                gt_indices = np.concatenate(gt_indices)
                # ----------------------
                
                indices_batchi[0] = np.concatenate([indices_batchi[0], pred_indices])
                indices_batchi[1] = np.concatenate([indices_batchi[1], gt_indices])
                
                # remove the redundant
                _, inverse_indices = np.unique(indices_batchi[0], return_index=True)
                indices_batchi[0] = indices_batchi[0][inverse_indices]
                indices_batchi[1] = indices_batchi[1][inverse_indices]

                indices.append(indices_batchi)
                punish_mask_list.append(punish_mask)

        return indices, punish_mask_list

