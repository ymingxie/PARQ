# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os

import time
from datetime import datetime
from os import path as osp
from typing import List
import argparse

from model import PARQ

import torch

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from datasets.scannet_dataset import ScanNetDataModule
from config import cfg, update_config


def setup_tensorboard(cfg):
    tensorboard_run = f"parq_{datetime.fromtimestamp(time.time()).strftime('%y-%m-%d-%H-%M-%S')}"

    tb_logger = TensorBoardLogger(
        save_dir=cfg.LOG_PATH,
        name=cfg.NAME,
        version="model_" + tensorboard_run,
        default_hp_metric=False,
    )
    log_path = osp.join(cfg.LOG_PATH, cfg.NAME, "model_" + tensorboard_run)
    return log_path, tensorboard_run, tb_logger


def train(cfg):
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.info("training started")
    # pring config
    rank = int(os.getenv("RANK", 0))
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.info(f"starting train function {rank}")

    seed_everything(cfg.SEED)

    # create datamodule
    data_module = ScanNetDataModule(cfg.DATAMODULE)

    # create model from scratch
    logger.info("Setting up Perceiver model")
    model = PARQ(cfg)

    if cfg.PRETRAINED_PATH is not None:
        # just reload the weights, without other parameters (e.g. optimizer, scheduler, lr, etc.)
        assert cfg.PRETRAINED_PATH is None
        logger.info("Load pretrained model.")
        checkpoint = torch.load(cfg.PRETRAINED_PATH, map_location=model.device)[
            "state_dict"
        ]
        model.load_state_dict(checkpoint, strict=False)

    logger.info("Setting up tb logger")
    log_path, tb_run, tb_logger = setup_tensorboard(cfg)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    callbacks.append(
        ModelCheckpoint(
            monitor=cfg.CALLBACK.MONITOR,
            save_top_k=cfg.CALLBACK.SAVE_TOP_K,
            save_last=cfg.CALLBACK.SAVE_LAST,
            verbose=cfg.CALLBACK.VERBOSE,
            auto_insert_metric_name=cfg.CALLBACK.AUTO_INSERT_METRIC_NAME,
            mode=cfg.CALLBACK.MODE,
            dirpath=log_path,
            filename=tb_run + "-epoch{epoch:04d}-val_loss{val/total_loss:.2f}-05_f1{val/metrics/0.5_f1:.3f}",
            )
    )
    callbacks.append(
        LearningRateMonitor(
            logging_interval= 'step'
            )
    )

    if cfg.CHECKPOINT_PATH is None:
        logger.info("no model checkpoint found - starting from scratch!")
    else:
        logger.info(f"loading model checkpoint from {cfg.CHECKPOINT_PATH}")

    effective_batch_size = (
        cfg.DATAMODULE.BATCH_SIZE
        * cfg.TRAINER.NUM_NODES
        * cfg.TRAINER.GPUS
        * cfg.TRAINER.ACCUMULATE_GRAD_BATCHES
    )

    strategy = DDPStrategy(find_unused_parameters=False, static_graph=True)
    trainer = Trainer(
        strategy=strategy,
        profiler=cfg.TRAINER.PROFILER,
        accelerator=cfg.TRAINER.ACCELERATOR,
        gpus=cfg.TRAINER.GPUS,
        num_nodes=cfg.TRAINER.NUM_NODES,
        accumulate_grad_batches=cfg.TRAINER.ACCUMULATE_GRAD_BATCHES,
        max_epochs=cfg.TRAINER.MAX_EPOCHS,
        log_every_n_steps=cfg.TRAINER.LOG_EVERY_N_STEPS,
        gradient_clip_val=cfg.TRAINER.GRADIENT_CLIP_VAL,
        reload_dataloaders_every_n_epochs=cfg.TRAINER.RELOAD_DATALOADERS_EVERY_N_EPOCHS,
        replace_sampler_ddp=cfg.TRAINER.REPLACE_SAMPLER_DDP,
        overfit_batches=cfg.TRAINER.OVERFIT_BATCHES,
        auto_scale_batch_size=cfg.TRAINER.AUTO_SCALE_BATCH_SIZE,
        check_val_every_n_epoch=cfg.TRAINER.CHECK_VAL_EVERY_N_EPOCH,
        precision=cfg.TRAINER.PRECISION,
        val_check_interval=cfg.TRAINER.VAL_CHECK_INTERVAL,
        resume_from_checkpoint=cfg.CHECKPOINT_PATH,
        limit_val_batches=8192 // effective_batch_size,
        callbacks=callbacks,
        logger=tb_logger,
    )

    # Training mode
    logger.info("starting training")
    trainer.fit(model=model, datamodule=data_module)
    logger.info(f"k best models: {trainer.checkpoint_callback.best_k_models}")
    logger.info(f"best model: {trainer.checkpoint_callback.best_model_path}")
    modelPath = trainer.checkpoint_callback.best_model_path
    if modelPath == "":
        logger.info(
            "starting testing the current model since no best model is saved yet."
        )
        retTest = trainer.test(model=model, datamodule=data_module)
        retVal = trainer.validate(model=model, datamodule=data_module)
    else:
        logger.info(f"starting testing model: {modelPath}")
        retTest = trainer.test(ckpt_path=modelPath, datamodule=data_module)
        retVal = trainer.validate(ckpt_path=modelPath, datamodule=data_module)

    logger.info("done training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A PyTorch Implementation of PARQ')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # parse arguments and check
    args = parser.parse_args()
    update_config(cfg, args)

    print("training with pytorch lightning")
    train(cfg)
