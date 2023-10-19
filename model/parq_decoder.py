# Copyright (c) Meta Platforms, Inc. and affiliates.
from functools import partial
from typing import List

import torch
from pytorch_lightning.utilities import rank_zero_only

from model.generic_mlp import GenericMLP
from model.transformer_parq import Transformer
from utils.parq_utils import (  # noqa
    BoxProcessor,
    compute_rotation_matrix_from_ortho6d,
    draw_detections,
    F1Calculator,
    HungarianMatcherModified,
    nms,
    rot_to_6d,
    roty,
)

from utils import Obb3D, Pose

import copy


def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PARQDecoder(torch.nn.Module):
    """
    PARQ Decoder, including one-layer transformer decoder and mlp heads
    """

    def __init__(
        self,
        cfg
    ):
        super(PARQDecoder, self).__init__()
        self.dim_in = cfg.DIM_IN
        self.num_queries = cfg.NUM_QUERIES
        self.num_semcls = cfg.NUM_SEMCLS
        self.loss_weight = cfg.LOSS_WEIGHT
        # NOTE following the official DETR rep0, bg_cls_weight means
        # relative classification weight of the no-object class.
        self.class_weight = torch.ones(self.num_semcls + 1)  # * class_weight
        # set background class as the last indice
        self.class_weight[self.num_semcls] = 0.1
        self.for_vis = cfg.FOR_VIS
        self.track_scale = cfg.TRACK_SCALE

        self.share_mlp_heads = cfg.SHARE_MLP_HEADS
        self.build_mlp_heads(
            self.num_semcls, self.dim_in, cfg.MEAN_SIZE_PATH, mlp_dropout=0.3
        )
        
        self.parq_module = Transformer(
                 dec_dim=cfg.TRANSFORMER.DEC_DIM, 
                 queries_dim=cfg.TRANSFORMER.QUERIES_DIM, 
                 dec_heads=cfg.TRANSFORMER.DEC_HEADS, 
                 dec_layers=cfg.TRANSFORMER.DEC_LAYERS, 
                 dec_ffn_dim=cfg.TRANSFORMER.DEC_FFN_DIM, 
                 dropout_rate=cfg.TRANSFORMER.DROPOUT_RATE,
                 scale=cfg.TRANSFORMER.SCALE,
                 share_weights=cfg.TRANSFORMER.SHARE_WEIGHTS)
        self.parq_module.decoder.mlp_heads = self.mlp_heads
        self.parq_module.decoder.box_processor = self.box_processor
        self.refpoint = torch.nn.Embedding(self.num_queries, 3)

        # match between the prediction and gt
        self.matcher = HungarianMatcherModified(cost_class=2, cost_bbox=0.25)

        if not isinstance(cfg.EVAL_TYPE, list):
            eval_type = [cfg.EVAL_TYPE]
        self.metrics_calculator = []
        for et in eval_type:
            if et == "f1":
                self.metrics_calculator.append(
                    F1Calculator(cfg.CONF_THRESH)
                )

        self.enable_nms = cfg.ENABLE_NMS

    def build_mlp_heads(
        self, num_semcls, decoder_dim, mean_size_path=None, mlp_dropout=None,
    ):
        """
        build mlp head to regress the parameters of the boxes
        """
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="ln",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=0.0,
            input_dim=decoder_dim,
        )
        mlp_func_small = partial(
            GenericMLP,
            norm_fn_name="ln",
            activation="relu",
            use_conv=True,
            hidden_dims=[],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        # add 1 for background/not-an-object class
        semcls_head = mlp_func_small(output_dim=num_semcls + 1)

        # geometry of the box
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func_small(output_dim=3)
        # ortho6d: https://arxiv.org/pdf/1812.07035.pdf
        rotation_head = mlp_func(output_dim=6)

        if not self.share_mlp_heads:
            semcls_head = _get_clones(semcls_head, self.model_cfg.dec_layers)
            center_head =  _get_clones(center_head, self.model_cfg.dec_layers)
            size_head = _get_clones(size_head, self.model_cfg.dec_layers)
            rotation_head = _get_clones(rotation_head, self.model_cfg.dec_layers)
            
        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("rotation_head", rotation_head),
        ]
        self.mlp_heads = torch.nn.ModuleDict(mlp_heads)
        self.box_processor = BoxProcessor(num_semcls, mean_size_path)
    
    def forward(self, intput_tokens, camera, T_camera_pseudoCam, T_world_pseudoCam, T_world_local):
        """
        aplly PARQ module, including transformer decoder and mlp heads
        input:
            input_tokens:       tensor, (B, N, C), the output of tokenization of (image features + ray positional encoding)
            camera:             camera intrinsics, (B, 6), width, height, fx, fy, cx, cy.
            T_camera_pseudoCam: pose, (B, 12), pose matrix from pseudo camera to camera
            T_world_pseudoCam:  pose, (B, 12), pose matrix from pseudo camera to world
            T_world_local:      pose, (B, 12), pose matrix from local to world
        output:
            box_prediction_list (parameters of 3d boxes): 
            [
                iteration 1 (dict):
                    pred_logits:          (B, N, num_semcls + 1), the classification logits (including background class)
                    center_unnormalized:  (B, N, 3), the predicted center of the box
                    size_unnormalized:    (B, N, 3), the predicted size of the box
                    ortho6d:              (B, N, 6), the predicted rotation of the box
                    sem_cls_prob:         (B, N, num_semcls + 1), the softmax of pred_logits
                    coord_pos:            (B, N, 3), the position of reference points, used for matcher
                iteration 2 (dict):
                    ...
                ...
            ]
        """
        meta_data = {"T_world_local": T_world_local,
                     "camera": camera,
                     "T_camera_pseudoCam": T_camera_pseudoCam,
                     "T_world_pseudoCam": T_world_pseudoCam}
        box_prediction_list= self.parq_module(intput_tokens, self.refpoint.weight, meta_data)
        return box_prediction_list

    def parse_target(
        self, bb3_target: Obb3D, T_world_local: Pose
    ):
        """
        parse the target 3d boxes to the format of the model
        """
        T_local_world = T_world_local.inverse()
        bb3_target_list = []
        sem_id = bb3_target.sem_id
        bs = sem_id.shape[0]
        for i in range(bs):
            bb3_target_i = bb3_target[i].remove_padding()
            # 3d boxes in object coords (center: 0,0,0; no rotation)
            # transform from object coord to local coord
            T_local_object = T_local_world[i] @ bb3_target_i.T_world_object
            size = bb3_target_i.bb3_size
            center_local = T_local_object.transform(
                bb3_target_i.bb3_center_object.unsqueeze(1)
            ).squeeze(1)
            bb3corners_local = T_local_object.transform(bb3_target_i.bb3corners_object)

            gt_ortho6d = rot_to_6d(T_local_object.R)

            gt_corners_world = bb3_target_i.T_world_object.transform(
                bb3_target_i.bb3corners_object
            )

            bb3_target_dict = {
                "labels": bb3_target_i.sem_id.squeeze(-1).long(),
                "center": center_local,
                "size": size,
                "T_rig_object": T_local_object.matrix.view(-1, 4, 4),
                "gt_corners": bb3corners_local,
                "gt_ortho6d": gt_ortho6d,
                "gt_corners_world": gt_corners_world,
                "T_world_object": bb3_target_i.T_world_object,
            }
            bb3_target_list.append(bb3_target_dict)
        return bb3_target_list

    def rotation_loss_with_sym(self, rot_predict, rot_target, sym_):
        """
          resolve symmetry for rotation loss
          sym_mapping = {
            "__SYM_NONE": 0,
            "__SYM_ROTATE_UP_2": 1,
            "__SYM_ROTATE_UP_4": 2,
            "__SYM_ROTATE_UP_INF": 3,
        }
        """
        rot = roty
        # --> resolve symmetry
        rot_loss_list = []
        for o in range(sym_.shape[0]):
            if sym_[o] == 1:
                m = 2
                tmp = [
                    torch.pow(
                        rot_predict[o]
                        - rot_target[o]
                        @ rot((k * 2.0 / m) * torch.pi, rot_predict.device),
                        2,
                    ).mean()
                    for k in range(m)
                ]
                rot_loss = torch.min(torch.stack(tmp))
            elif sym_[o] == 2:
                m = 4
                tmp = [
                    torch.pow(
                        rot_predict[o]
                        - rot_target[o]
                        @ rot((k * 2.0 / m) * torch.pi, rot_predict.device),
                        2,
                    ).mean()
                    for k in range(m)
                ]
                rot_loss = torch.min(torch.stack(tmp))
            elif sym_[o] == 3:
                m = 36
                tmp = [
                    torch.pow(
                        rot_predict[o]
                        - rot_target[o]
                        @ rot((k * 2.0 / m) * torch.pi, rot_predict.device),
                        2,
                    ).mean()
                    for k in range(m)
                ]
                rot_loss = torch.min(torch.stack(tmp))
            else:
                rot_loss = torch.pow(
                    rot_predict[o] - rot_target[o],
                    2,
                ).mean()
            rot_loss_list.append(rot_loss)
        rot_loss = torch.mean(torch.stack(rot_loss_list))
        return rot_loss

    def loss(
        self,
        out_dict_list,
        obbs_padded: Obb3D,
        T_world_local: Pose,
        sym: List = None,
        *argv,
    ):
        """
        input:
            out_dict_list:  predicted box3d parameters
            obbs_padded:    target box3d parameters
            T_world_local:  pose matrix from local to world
            sym:            symmetry of the object
        output:
            loss
        """
        assert obbs_padded.ndim == 3, f"{obbs_padded.shape}"

        loss_total = (
            out_dict_list[-1]["ortho6d"].sum()
            * out_dict_list[-1]["size_unnormalized"].sum()
            * out_dict_list[-1]["center_unnormalized"].sum()
            * out_dict_list[-1]["pred_logits"].sum()
            * 0
        )

        loss_dict = {"center_loss": 0, "size_loss": 0, "rot_loss": 0, "cat_loss": 0}
        valid_bs = 0

        bb3_target_list = self.parse_target(
                bb3_target=obbs_padded,
                T_world_local=T_world_local,
        )
                    
        for out_dict in out_dict_list:
            # parse target
            bs = obbs_padded.shape[0]
            indices, punish_mask = self.matcher(out_dict, bb3_target_list)
            
            # compute loss
            for i in range(bs):
                if len(indices[i][0]) != 0:
                    valid_bs += 1
                    # center loss: l1
                    center_predict = out_dict["center_unnormalized"][i][indices[i][0]]
                    center_target = bb3_target_list[i]["center"][indices[i][1]]
                    center_loss = (center_predict - center_target).abs().mean()
                    center_loss *= self.loss_weight[0]
                    loss_total += center_loss
                    loss_dict["center_loss"] += center_loss
                    # size loss: l1
                    size_predict = out_dict["size_unnormalized"][i][indices[i][0]]
                    size_target = bb3_target_list[i]["size"][indices[i][1]]
                    size_loss = (size_predict - size_target).abs().mean()
                    size_loss *= self.loss_weight[1]
                    loss_total += size_loss
                    loss_dict["size_loss"] += size_loss
                    # rotation loss: cosine similarity or rmat l2
                    # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                    rot_6d_predict = out_dict["ortho6d"][i][indices[i][0]]
                    rot_target = bb3_target_list[i]["T_rig_object"][:, :3, :3][
                        indices[i][1]
                    ]
                    rot_predict = compute_rotation_matrix_from_ortho6d(rot_6d_predict)
                    # loss rmat
                    if sym is not None:
                        sym_ = sym[i][indices[i][1]]
                        rot_loss = self.rotation_loss_with_sym(
                            rot_predict, rot_target, sym_
                        )
                    else:
                        rot_loss = torch.pow(rot_target - rot_predict, 2).mean()
                    rot_loss *= self.loss_weight[2]
                    loss_total += rot_loss
                    loss_dict["rot_loss"] += rot_loss
                    # category loss
                    classes_target_o = bb3_target_list[i]["labels"][indices[i][1]]
                    classes_target = torch.full(
                        out_dict["pred_logits"].shape[1:2],
                        self.num_semcls,
                        dtype=torch.int64,
                        device=out_dict["pred_logits"].device,
                    )
                    classes_target[indices[i][0]] = classes_target_o
                    if punish_mask is not None:
                        cross_entropy = torch.nn.CrossEntropyLoss(
                            self.class_weight.to(classes_target_o.device), reduce=False
                        )
                        cat_loss = cross_entropy(out_dict["pred_logits"][i], classes_target)
                    
                        cat_loss = (cat_loss * punish_mask[i]).sum() / punish_mask[i].sum()
                    else:
                        cross_entropy = torch.nn.CrossEntropyLoss(
                            self.class_weight.to(classes_target_o.device)
                        )
                        cat_loss = cross_entropy(out_dict["pred_logits"][i], classes_target)
                    cat_loss *= self.loss_weight[3]
                    loss_total += cat_loss
                    loss_dict["cat_loss"] += cat_loss
        if valid_bs != 0:
            loss_total = loss_total / valid_bs
            for key, value in loss_dict.items():
                loss_dict[key] = value / valid_bs
            
        loss_dict["total_loss"] = loss_total
        return loss_dict

    def parse_pred(self, out_dict):
        """
        reorganize the predicitions into OBB class. Also aplly some filters here, e.g. remove the predictions out the scope, nms
        """
        # only use the prediciton in the last iteration
        out_dict = out_dict[-1]
        size_predict = out_dict["size_unnormalized"]
        center_predict = out_dict["center_unnormalized"]
        logits = out_dict["sem_cls_prob"]
        scores, labels = torch.max(logits, -1)
        bs = logits.shape[0]
        rot_6d_predict = out_dict["ortho6d"]
        rot_matrix = compute_rotation_matrix_from_ortho6d(
            rot_6d_predict.contiguous().view(-1, 6)
        ).view(bs, -1, 3, 3)

        # t_matrix: T_local_object
        t_matrix = Pose.from_Rt(rot_matrix, center_predict)
        x_min, x_max = -size_predict[..., 0] / 2, size_predict[..., 0] / 2
        y_min, y_max = -size_predict[..., 1] / 2, size_predict[..., 1] / 2
        z_min, z_max = -size_predict[..., 2] / 2, size_predict[..., 2] / 2
        c3o = torch.stack(
            [x_min, x_max, y_min, y_max, z_min, z_max],
            dim=-1,
        )

        obbs_pred = Obb3D.separate_init(
            bb3_object=c3o.cpu(),
            T_world_object=t_matrix._data.cpu(),
            sem_id=labels.cpu(),
        )
        out_dict["obbs_pred"] = obbs_pred.cuda()

        # the model will preserve all outputs from one snippet for visualization without ant filtering, 
        # e.g. filter out far-away boxes or using nms
        if not self.for_vis:
            valid = (
                (center_predict[..., 0] > self.track_scale[0]).int()
                + (center_predict[..., 0] < self.track_scale[1]).int()
                + (center_predict[..., 2] > self.track_scale[4]).int()
                + (center_predict[..., 2] < self.track_scale[5]).int()
            )
            valid = valid == 4
        else:
            valid = torch.ones_like(center_predict[..., 0]).bool()
        if self.enable_nms:
            if not self.for_vis:
                pred_mask = nms(obbs_pred, scores, self.num_semcls, 0.1, "nms_3d_faster")
            else:
                pred_mask = nms(obbs_pred, scores, self.num_semcls, 0.2, "nms_3d_faster_samecls")
        out_dict["pred_mask"] = torch.tensor(pred_mask).to(valid.device) & valid

        return out_dict

    def update_metrics(
        self,
        out_dict,
        obbs_padded: Obb3D,
        T_world_local: Pose,
        scene_name: str = None,
    ):
        """
        prediction parse and accumulate for f1 evaluation
        input:
            out_dict: box parameters
            target: ["obbs_padded", "T_world_rig", "scene_name"]
        """
        assert obbs_padded.ndim == 3, f"{obbs_padded.shape}"

        # parse prediction
        out_dict = self.parse_pred(out_dict)

        # parse target
        bb3_target_list = self.parse_target(
            bb3_target=obbs_padded,
            T_world_local=T_world_local,
        )
        for key, value in out_dict.items():
            if value is not None:
                out_dict[key] = value.detach()
        # for hungarian matching
        obbs_pred = out_dict["obbs_pred"]
        out_dict["scene_name"] = scene_name
        out_dict["pred_corners_world"] = T_world_local.transform(
            obbs_pred.T_world_object.transform(obbs_pred.bb3corners_object)
        ).detach()
        for calculator in self.metrics_calculator:
            calculator.step(out_dict, bb3_target_list)

    def compute_metrics(self):
        metrics = {}
        for calculator in self.metrics_calculator:
            metrics.update(calculator.compute_metrics())
        return metrics

    def reset_metrics(self):
        for calculator in self.metrics_calculator:
            calculator.reset()

    @rank_zero_only
    def log_images(
        self,
        out_dict,
        obbs_padded: Obb3D,
        Ts_world_pseudoCam: Pose,
        Ts_world_local: Pose,
        T_camera_pseudoCam: Pose,
        rgb_img=None,
        calib_rgb=None,
        slaml_img=None,
        calib_slaml=None,
        slamr_img=None,
        calib_slamr=None,
    ):
        # only log batch 0
        imgs = [rgb_img[0]]
        calibs = [calib_rgb[0]]
        tags = ["rgb_img"]
        tags_gt = ["rgb_img_gt"]
        # scannet only has rgb
        if slaml_img is not None:
            imgs.append(slaml_img[0])
            calibs.append(calib_slaml[0])
            tags.append("slaml_img")
            tags_gt.append("slaml_img_gt")
        if slamr_img is not None:
            imgs.append(slamr_img[0])
            calibs.append(calib_slamr[0])
            tags.append("slamr_img")
            tags_gt.append("slamr_img_gt")

        out_dict = self.parse_pred(out_dict)
        obbs_pred = out_dict["obbs_pred"]
        pred_mask = out_dict["pred_mask"]
        T_local_object = obbs_pred.T_world_object
        T_world_object = Ts_world_local @ T_local_object
        T_pseudoCam_world = Ts_world_pseudoCam.inverse()
        imgs_pred = draw_detections(
            imgs,
            calibs,
            obbs_pred.bb3corners_object[0],
            T_world_object[0],
            T_pseudoCam_world[0],
            T_camera_pseudoCam[0],
            obbs_pred.sem_id.squeeze(-1)[0].int(),
            tags,
            self.num_semcls,
            mask=pred_mask[0],
        )

        if obbs_padded != None:
            obbs_padded = obbs_padded.remove_padding()
            imgs_gt = draw_detections(
                imgs,
                calibs,
                obbs_padded[0].bb3corners_object,
                obbs_padded[0].T_world_object,
                T_pseudoCam_world[0],
                T_camera_pseudoCam[0],
                obbs_padded[0].sem_id.squeeze(-1).int(),
                tags_gt,
                self.num_semcls,
            )
        else:
            imgs_gt = {}

        return {**imgs_pred, **imgs_gt}
