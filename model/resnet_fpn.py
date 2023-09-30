# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
from typing import Literal

import einops

import torch
from torchvision import transforms
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


class ResnetFPN(torch.nn.Module):
    """
    Extract a specified ResNet backbone with FPN on top
    resnet_name: type of resnet, e.g., resnet18, resnet34, resnet50, resnet101, resnet152
    layers:  name of layers to extract
    """

    def __init__(
        self,
        resnet_name: Literal[
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        ] = "resnet18",
        layer: str = "0",
        freeze: bool = True,
    ):
        super(ResnetFPN, self).__init__()
        assert resnet_name in [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ]
        self.resnet_fpn = resnet_fpn_backbone(
            resnet_name, pretrained=True, trainable_layers=5
        )
        self.layer = str(layer)
        self.transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )

        self.freeze = freeze
        if self.freeze:
            self.feature_extractor.eval()
        logger.info(f"Resnet FPN Preprocessor {resnet_name}, layers: {layer}")

    def forward(self, batch):
        B = batch["rgb_img"].shape[0]

        batch_img = batch["rgb_img"]

        if batch["rgb_img"].dim() == 5:
            batch_img = einops.rearrange(batch_img, "b t c h w -> (b t) c h w")

        # normalized mean and std
        batch_img = self.transform(batch_img)

        if self.freeze:
            with torch.no_grad():
                features = self.resnet_fpn(batch_img)
        else:
            features = self.resnet_fpn(batch_img)

        feature_list = []
        for layer in range(4):
            feature = features[str(layer)]
            feature = torch.nn.functional.interpolate(
                feature, features[self.layer].shape[-2:], mode="bilinear"
            )
            feature_list.append(feature)
        v = torch.cat(feature_list, dim=1)
            
        if batch["rgb_img"].dim() == 5:
            v = v.view(B, batch["rgb_img"].shape[1], *(v.size()[-(v.dim() - 1) :]))

        batch["all_features"] = v

        logger.debug(f"{self.layer} {v.shape}")
        camera = batch["camera"]
        scale_factor = 1 / (2 ** (int(self.layer) + 2))
        batch['camera_feature'] = camera.scale(scale_factor)
        return batch
