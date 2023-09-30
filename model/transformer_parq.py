
"""
Copyright (c) 2023 Yiming Xie
This file is derived from [DETR](https://github.com/facebookresearch/detr/blob/main/models/transformer.py).
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

DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.functional import grid_sample
import math


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    # https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/models/dense_heads/petr_head.py#L29
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_z = torch.stack(
        (pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb

        
class Transformer(nn.Module):

    def __init__(self, dec_dim=512, queries_dim=512, dec_heads=8,
                 dec_layers=6, dec_ffn_dim=2048, dropout_rate=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=True, 
                 scale=None,
                 share_weights=False):
        super().__init__()
        assert (
            queries_dim == dec_dim
        ), f"queries dim {queries_dim} needs to be equal to input enc_dim {dec_dim} for transformer encoder"
        
        decoder_layer = TransformerDecoderLayer(dec_dim, dec_heads, dec_ffn_dim,
                                                    dropout_rate, activation, normalize_before)
        decoder_norm = nn.LayerNorm(dec_dim)
        self.decoder = TransformerDecoder(decoder_layer, dec_layers, dec_dim, scale, decoder_norm,
                                              return_intermediate=return_intermediate_dec, share_weights=share_weights)
        self._reset_parameters()

        self.d_model = dec_dim
        self.nhead = dec_heads

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_tokens, query_tokens, meta_data, mask=None):
        """
        Decoder - PARQ module, including transformer decoder and mlp heads
        input:
            input_tokens:       tensor, (B, N, C), the output of tokenization of (image features + ray positional encoding)
            query_tokens:       tensor, (N', C), reference points, N' is the number of reference points
            meta_data:          dict, meta data of the scene, including:
                                    camera:             camera intrinsics, (B, 6), width, height, fx, fy, cx, cy.
                                    T_camera_pseudoCam: pose, (B, 12), pose matrix from pseudo camera to camera
                                    T_world_pseudoCam:  pose, (B, 12), pose matrix from pseudo camera to world
                                    T_world_local:      pose, (B, 12), pose matrix from local to world
        output:
            output_list (parameters of 3d boxes): 
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
        bs = input_tokens.shape[0]
        query_tokens = query_tokens.repeat(bs, 1, 1)

        output_list = self.decoder(None, input_tokens, memory_key_padding_mask=mask,
                          query_pos=query_tokens, meta_data=meta_data)
        return output_list


def project(memory_hw, query_pos, T_camera_local, camera):
    '''
    Project reference points onto multi-view image to fetch appearence features
    Bilinear interpolation is used to fetch features.
    Average pooling is used to aggregate features from different views.
    '''
    w, h = camera.size[0][0].data.cpu().numpy()
    # from local coord to camera coord
    query_pos_c = T_camera_local.transform(query_pos.unsqueeze(1))

    center_b_list = []
    valid_b_list = []
    for cam_b, pos_c_b in zip(camera, query_pos_c):
        center_im_b, center_valid_b = cam_b.project(pos_c_b)
        center_b_list.append(center_im_b)
        valid_b_list.append(center_valid_b)
    center_im = torch.stack(center_b_list)
    center_valid = torch.stack(valid_b_list)
        
    im_grid = torch.stack([2 * center_im[..., 0] / (w - 1) - 1, 2 * center_im[..., 1] / (h - 1) - 1], dim=-1)
    bs, num_view, num_query, _ = im_grid.shape
    im_grid = im_grid.view(bs * num_view, 1, -1, 2)
        
    features = grid_sample(memory_hw, im_grid, padding_mode='zeros', align_corners=True)
    features = features.view(bs, num_view, -1, num_query)
    features = features.permute(0, 1, 3, 2).contiguous()
    # average across different views
    features = features.sum(dim=1)
    mask = center_valid.sum(dim=1)
    invalid_mask = mask == 0
    mask[invalid_mask] = 1
    features /= mask.unsqueeze(-1)
    return features, center_im, center_valid


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, dim_in, scale, norm=None, return_intermediate=False, share_weights=False):
        super().__init__()
        if not share_weights:
            self.layers = _get_clones(decoder_layer, num_layers)
        else:
            self.layers = _get_clones(decoder_layer, 1)
        self.num_layers = num_layers
        self.share_weights = share_weights
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.position_encoder = torch.nn.Sequential(
                torch.nn.Linear(128 * 3, dim_in),
                torch.nn.ReLU(),
                torch.nn.Linear(dim_in, dim_in),
            )
        self.mlp_heads = None
        self.box_processor = None
        self.scale = scale

    def normalize(self, center_offset):
        center_offset1 = (center_offset[..., 0] - self.scale[0]) / (
                    self.scale[1] - self.scale[0]
                )
        center_offset2 = (center_offset[..., 1] - self.scale[2]) / (
                    self.scale[3] - self.scale[2]
                )
        center_offset3 = (center_offset[..., 2] - self.scale[4]) / (
                    self.scale[5] - self.scale[4]
                )
        center_offset = torch.stack([center_offset1, center_offset2, center_offset3], dim=-1)
        return center_offset
    
    def denormalize(self, center_offset):
        center_offset1 = (
                center_offset[..., 0] * (self.scale[1] - self.scale[0]) + self.scale[0]
            )
        center_offset2 = (
                center_offset[..., 1] * (self.scale[3] - self.scale[2]) + self.scale[2]
            )
        center_offset3 = (
                center_offset[..., 2] * (self.scale[5] - self.scale[4]) + self.scale[4]
            )
        center_offset = torch.stack([center_offset1, center_offset2, center_offset3], dim=-1)
        return center_offset
    
    def bbox3d_prediction(self, tokens, coord, layer_num):
        """
        Predict the paramers of the boxes via multiple MLP heads
        input:
            tokens: tensor, (B, N, C), output of transformer decoder
            coord: tensor, (B, N, 3), coordinates of the reference points
            layer_num: int, the layer number of the transformer decoder
        output:
            out_dict: parameters of 3d boxes
        """
        if tokens.dim() == 4:
            tokens_list = torch.split(tokens, 1, dim=0)
        else:
            tokens_list = [tokens.unsqueeze(0)]
        
        share_mlp_heads = True
        if isinstance(self.mlp_heads["sem_cls_head"], torch.nn.ModuleList):
            share_mlp_heads = False
            
        box_prediction_list = []
        for tokens in tokens_list:
            tokens = tokens[0]
            # tokens are B x nqueries x noutput, change to B x noutput x bqueries
            tokens = tokens.permute(0, 2, 1).contiguous()
            if share_mlp_heads:
                cls_logits = self.mlp_heads["sem_cls_head"](tokens).transpose(1, 2)
                center_offset = self.mlp_heads["center_head"](tokens).transpose(1, 2)
            else:
                cls_logits = self.mlp_heads["sem_cls_head"][layer_num](tokens).transpose(1, 2)
                center_offset = self.mlp_heads["center_head"][layer_num](tokens).transpose(1, 2)

            coord_pos = self.denormalize(coord)
            center_offset = center_offset + inverse_sigmoid(coord)
            center_offset = center_offset.sigmoid()
            center_offset = self.denormalize(center_offset)
            
            if share_mlp_heads:
                size_scale = self.mlp_heads["size_head"](tokens).transpose(1, 2)
                ortho6d = self.mlp_heads["rotation_head"](tokens).transpose(1, 2)
            else:
                size_scale = self.mlp_heads["size_head"][layer_num](tokens).transpose(1, 2)
                ortho6d = self.mlp_heads["rotation_head"][layer_num](tokens).transpose(1, 2)

            (
                center_normalized,
                center_unnormalized,
            ) = self.box_processor.compute_predicted_center(center_offset)

            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            with torch.no_grad():
                (
                    semcls_prob,
                    objectness_prob,
                ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits)

            size_unnormalized = self.box_processor.compute_predicted_size(
                size_scale, semcls_prob
            )

            box_prediction = {
                "pred_logits": cls_logits,
                "center_unnormalized": center_unnormalized,
                "size_unnormalized": size_unnormalized,
                "ortho6d": ortho6d,
                "sem_cls_prob": semcls_prob,
                # used in matcher
                "coord_pos": coord_pos, # use input reference point to match instead of output center
            }
            box_prediction_list.append(box_prediction)
        return box_prediction_list

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                meta_data = None):
        # parse meta data
        camera = meta_data["camera"]
        T_camera_pseudoCam = meta_data["T_camera_pseudoCam"]
        T_world_pseudoCam = meta_data["T_world_pseudoCam"]
        T_world_local = meta_data["T_world_local"]
        bs, num_view = T_camera_pseudoCam.shape[:2]

        T_camera_local = T_camera_pseudoCam @ (
                T_world_pseudoCam.inverse() @ T_world_local
            )
        w, h = camera.size[0][0].data.cpu().numpy()
        memory_hw = memory.view(bs * num_view, int(h), int(w), -1)
        memory_hw = memory_hw.permute(0, 3, 1, 2)
        
        memory = memory.permute(1, 0, 2)
        
        
        intermediate = []
        reference_points = query_pos.sigmoid()
        for layer_num in range(self.num_layers):
            if self.share_weights:
                layer = self.layers[0]
            else:
                layer = self.layers[layer_num]
            
            # positional encoding
            pos_feat = self.position_encoder(pos2posemb3d(reference_points))
            pos_feat = pos_feat.permute(1, 0, 2)

            # project
            pixel_aligned, center_im, center_valid = project(memory_hw, self.denormalize(reference_points), T_camera_local, camera)
            output = pixel_aligned.permute(1, 0, 2)

            output, attn = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=pos_feat)
            output = output.permute(1, 0, 2)
            output_dict = self.bbox3d_prediction(output, reference_points, layer_num)
            reference_points = self.normalize(output_dict[0]["center_unnormalized"])
            reference_points = reference_points.detach()
            
            if self.return_intermediate:
                intermediate += output_dict

        return intermediate


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, attn

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
