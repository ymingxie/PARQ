# Copyright (c) Meta Platforms, Inc. and its affiliates. All Rights Reserved
import argparse
import faulthandler
import logging
import time

from model import PARQ
import torch
from config import cfg, update_config
from datasets.scannet_dataset import ScanNetDataModule
from datasets.demo_dataset import DemoModule

# Improve multiprocess debugging
faulthandler.enable(all_threads=True)
logging.basicConfig(level=logging.INFO)


def test_model(cfg):

    # create datamodule
    if cfg.DEMO:
        data_module = DemoModule(cfg.DATAMODULE)
    else:
        data_module = ScanNetDataModule(cfg.DATAMODULE)
    loader = data_module.val_dataloader()
    model = PARQ(cfg)

    if cfg.CHECKPOINT_PATH is not None:
        state_dict = torch.load(cfg.CHECKPOINT_PATH)
        state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=True)

    model.cuda()
    model.eval()
    time_all = []
    gpu_mem_usage = []
    for i, batch in enumerate(loader):
        print(i)
        for key, value in batch.items():
            try:
                batch[key] = value.cuda()
            except:
                pass
        print(batch["scene_name"])
        start_time = time.time()
        with torch.no_grad():
            loss = model.validation_step(batch, 0)
        print("inference time {}".format(time.time() - start_time))
        time_all.append(time.time() - start_time)
        average = torch.tensor(time_all).mean()
        print("average time {}".format(average.item()))
        if not cfg.DEMO:
            print("loss {}".format(loss.data.cpu().item()))
            del loss
        # torch.cuda.empty_cache()
        # gpu_mem_usage.append(torch.cuda.memory_reserved())
        # summary_text = f"""
        #         average time: {average.item()} 
        #         Average GPU memory usage (GB): {sum(gpu_mem_usage) / len(gpu_mem_usage) / (1024 ** 3)} 
        #         Max GPU memory usage (GB): {max(gpu_mem_usage) / (1024 ** 3)} 
        #     """
        # print(summary_text)
    ap = model.validation_epoch_end(None)

    for key, value in ap.items():
        print(key)
        print(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A PyTorch Implementation of PARQ')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--CHECKPOINT_PATH',
                        help='checkpoint path',
                        type=str)

    parser.add_argument('--DEMO',
                        help='if use demo dataset',
                        type=bool,
                        default=False)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    
    # parse arguments and check
    args = parser.parse_args()
    update_config(cfg, args)

    print("training with pytorch lightning")
    test_model(cfg)
