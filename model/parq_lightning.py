# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
import math
import os

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import LightningModule
from model.resnet_fpn import ResnetFPN
from model.ray_positional_encoding import AddRayPE
from model.parq_decoder import PARQDecoder
import sys
import einops
sys.path.append("..") # Adds higher directory to python modules path.


from utils.train_utils import CosineAnnealingWarmupRestarts

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# logger.setLevel(logging.DEBUG)

from utils.vis_utils import normalize, pca_compress


class PARQ(LightningModule):
    """
    PARQ Lightning Module

    Args:
        cfg:
            configure the lightning module
    """

    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.save_hyperparameters(cfg)
        # hparams == cfg; this supports loading from checkpoint
        self.cfg = self.hparams

        self.batch_size = self.cfg.DATAMODULE.BATCH_SIZE

        self.backbone2d = ResnetFPN(
            cfg.MODEL.BACKBONE2D.RESNET_NAME,
            cfg.MODEL.BACKBONE2D.LAYER,
            cfg.MODEL.BACKBONE2D.FREEZE)
        self.add_ray_pe = AddRayPE(
            cfg.MODEL.TOKENIZER.OUT_CHANNELS,
            cfg.MODEL.TOKENIZER.RAY_POINTS_SCALE,
            cfg.MODEL.TOKENIZER.NUM_SAMPLES,
            cfg.MODEL.TOKENIZER.MIN_DEPTH,
            cfg.MODEL.TOKENIZER.MAX_DEPTH)
        self.box3d_decoder = PARQDecoder(cfg.MODEL.DECODER)

    def prepare_data(self):
        rank = os.getenv("RANK")
        logger.info(f"prepare_data rank={rank}")

    def setup(self, stage=None):
        logger.info("setup done ")

    def forward(self, batch, batch_idx):
        # image feature extraction
        batch = self.backbone2d(batch)
        # generate ray positional encoding
        encoding = self.add_ray_pe(batch['all_features'], batch['camera_feature'], batch['T_camera_pseudoCam'], batch['T_world_pseudoCam'], batch['T_world_local'])

        # add ray positional encoding to image features
        images_feat = batch['all_features'] + encoding

        # tonkenize image features
        input_tokens = einops.rearrange(
            images_feat,
            "b (t dt) c (h dh) (w dw) -> b t h w (dt dh dw c)",
            dh=1,
            dw=1,
            dt=1,
        )
        input_tokens = einops.rearrange(input_tokens, "b t h w c -> b (t h w) c")

        # compute the outputs
        outputs = self.box3d_decoder(input_tokens, batch['camera_feature'], batch['T_camera_pseudoCam'], batch['T_world_pseudoCam'], batch['T_world_local'])
        
        # compute loss
        losses = self.box3d_decoder.loss(outputs, batch['obbs_padded'], batch['T_world_local'], batch['sym'])
        return losses, outputs
    
    def training_step(self, batch, batch_idx):
        losses, outputs = self.forward(batch, batch_idx)
        self.log_step(batch, losses, outputs, batch_idx=batch_idx, stage="train")
        return losses['total_loss']

    def validation_step(self, batch, batch_idx):
        losses, outputs = self.forward(batch, batch_idx)
        self.box3d_decoder.update_metrics(
                outputs, batch['obbs_padded'], batch['T_world_local'], batch['scene_name']
            )

        self.log_step(batch, losses, outputs, batch_idx=batch_idx, stage="val")
        return losses['total_loss']

    def on_validation_epoch_start(self) -> None:
        logger.debug("starting validation epoch; reset metrics")
        self.box3d_decoder.reset_metrics()

    def validation_epoch_end(self, outs):
        logger.debug("finishing validation epoch; evaluate metrics")
        metrics = {}
        metrics = self.box3d_decoder.compute_metrics()
        for key, value in metrics.items():
            if type(value) is np.ndarray:
                continue
            elif type(value) is Image.Image:
                if self.logger.experiment is not None:
                    self.logger.experiment.add_image(
                        "val/metrics/{}".format(key),
                        np.asarray(value).transpose(2, 0, 1),
                        self.trainer.current_epoch,
                    )
            else:
                self.log(
                    "val/metrics/{}".format(key),
                    value,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                    rank_zero_only=False,
                )
        logger.info(metrics)
        return metrics

    def test_step(self, batch, batch_idx=0):
        return self.forward(batch)

    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer):
        current_lr = [d["lr"] for d in optimizer.param_groups][0]
        logger.debug(f"Step: {self.global_step} LR: {current_lr:.4e}")

    def configure_optimizers(self):
        logger.info(f"configure optimizer {self.cfg.OPTIMIZER.NAME}")
        effective_batch_size = (
            self.cfg.DATAMODULE.BATCH_SIZE
            * self.cfg.TRAINER.NUM_NODES
            * self.cfg.TRAINER.GPUS
            * self.cfg.TRAINER.ACCUMULATE_GRAD_BATCHES
        )

        # "Accurate Large Minibatch SGD ..."
        if self.cfg.OPTIMIZER.AUTOSCALE_LR:
            lr = self.cfg.OPTIMIZER.LEARNING_RATE * effective_batch_size / 256.0
        else:
            lr = self.cfg.OPTIMIZER.LEARNING_RATE

        # use the multi_tensor implementation of AdamW to speed up optimizer.step()
        optimizer = torch.optim._multi_tensor.AdamW(
        self.get_model_params(),
            lr=lr,
        )
        warmup_epochs = self.cfg.OPTIMIZER.WARMUP_EPOCHS
        lr_min = self.cfg.OPTIMIZER.LEARNING_RATE
        if effective_batch_size <= 256:
            lr_min = lr_min / 256.0
        cycle_mult = self.cfg.OPTIMIZER.CYCLE_MULT
        num_restarts = self.cfg.OPTIMIZER.NUM_RESTARTS
        total_epochs = self.cfg.TRAINER.MAX_EPOCHS
        cycle_fractions = [pow(cycle_mult, i) for i in range(num_restarts)]
        epochs_cycle_0 = math.ceil(total_epochs / sum(cycle_fractions))
        logger.info(f"warm up epochs {warmup_epochs}")
        logger.info(f"cycle0 {epochs_cycle_0}")
        logger.info(f"total {total_epochs}")
        logger.info(f"cycle multiplier {cycle_mult}")
        logger.info(f"lr_min {lr_min} to lr_max {lr}")
        # cosine decay with warmup epochs
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            epochs_cycle_0,
            cycle_mult,
            lr,
            lr_min,
            warmup_epochs,
        )
        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            },
        )

    def get_model_params(self):
        return (
            filter(lambda p: p.requires_grad, self.parameters())
            if self.cfg.OPTIMIZER.IGNORE_FROZEN_PARAMS
            else self.parameters()
        )

    def log_image(self, img: torch.Tensor, tag: str, step: int = 0):
        if (
            self.cfg.LOG_IMAGES
            and self.logger is not None
            and self.logger.experiment is not None
        ):
            logger.debug(f"image to log {img.shape}")
            self.logger.experiment.add_image(tag, img, step)

    def log_step(self, batch, losses, outputs, batch_idx=0, stage="train"):
        if len(losses) > 0:
            for key, value in losses.items():
                self.log(
                    f"{stage}/{key}",
                    value,
                    on_step=True,
                    logger=True,
                    sync_dist=True,
                )

        if batch_idx == 0 or batch_idx % self.cfg.LOG_IMAGES_FREQUENCY == 0:
            logger.debug("logging images")
            log_ims = self.get_log_images(batch, outputs)
            for tag, im in log_ims.items():
                if isinstance(im, np.ndarray):
                    self.log_image(
                        im,
                        f"{stage}/{tag}",
                        self.global_step,
                    )
                else:
                    logger.warning(
                        f"not logging {stage}/{tag} because it unknown type {type(im)}"
                    )
    
    def log_input(
        self,
        images: torch.Tensor
    ):
        log_img = {}
        T = images.shape[1]
        input_snippet = einops.rearrange(images, "b t c h w -> (b t) c h w")
        input_snippet = einops.rearrange(
            input_snippet, "(b t) c h w -> b c (t h) w", t=T
        )
        log_img["input"] = input_snippet
        return log_img

    def get_log_images(self, batch, outputs):
        def convert_image(img):
            """convert logged images to format CxHxW (1<=C<=3) range 0 to 255 np.uin8"""
            if not isinstance(img, torch.Tensor):
                return img
            if img.dim() == 5:  # BxTxCxHxW
                img = img[0, 0, ...]
            elif img.dim() == 4:  # BxCxHxW
                img = img[0, ...]
            img = img.detach()
            if img.shape[0] > 3:  # C
                img = pca_compress(img.unsqueeze(0)).squeeze(0)
            img = normalize(img)
            # go from 0-1 to 0-255
            img = (img * 255).cpu().numpy()
            img = img.astype(np.uint8)
            if img.shape[0] == 1:
                img = img[0, :, :]
            if img.ndim == 2:
                img = np.stack([img, img, img])
            return img

        log_ims = {}
        # Log outputs.
        log_outputs = self.box3d_decoder.log_images(
            outputs,
            batch['obbs_padded'],
            batch['T_world_pseudoCam'],
            batch['T_world_local'],
            batch['T_camera_pseudoCam'],
            batch['rgb_img'],
            batch['camera'],
        )
        if log_outputs is not None:
            for tag, im in log_outputs.items():
                log_ims[f"{tag}"] = convert_image(im)

        # Log inputs.
        log_inputs = self.log_input(
            batch['rgb_img']
        )
        if log_inputs is not None:
            for tag, im in log_inputs.items():
                log_ims[f"{tag}"] = convert_image(im)

        return log_ims
