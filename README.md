# Pixel-Aligned Recurrent Queries for Multi-View 3D Object Detection
### [Project Page](https://ymingxie.github.io/parq) | [Paper](https://arxiv.org/abs/2310.01401)

> Pixel-Aligned Recurrent Queries for Multi-View 3D Object Detection  
> [Yiming Xie](https://ymingxie.github.io), [Huaizu Jiang](https://jianghz.me/), [Georgia Gkioxari*](https://gkioxari.github.io/), [Julian Straub*](http://people.csail.mit.edu/jstraub/)  
> ICCV 2023

![real-time video](assets/parq_teaser.gif)

<br/>
<!-- ## TODO -->
<!-- - [x] ScanNet Dataset -->
<!-- - [ ] ARKitScenes Dataset -->

## How to use
### Installation
```
conda env create -f environment.yml
```

### Pretrained Model on ScanNet
Download the [pretrained weights](https://drive.google.com/file/d/1FuIf1jDPX-ooOx0x-tS69ejhdn9NFuXz/view?usp=sharing) and put it under 
`PROJECT_PATH/checkpoint/`.
You can also use [gdown](https://github.com/wkentaro/gdown) to download it in command line:
```bash
gdown --id 1FuIf1jDPX-ooOx0x-tS69ejhdn9NFuXz
```

### Data Preperation for ScanNet
Download and extract ScanNet by following the instructions provided at http://www.scan-net.org/.
<details>
  <summary>[Expected directory structure of ScanNet (click to expand)]</summary>
  
You can obtain the train/val/test split information from [here](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark).
```
PROJECT_PATH
└───data
|   └───scannet
|   │   └───scans
|   │   |   └───scene0000_00
|   │   |       └───color
|   │   |       │   │   0.jpg
|   │   |       │   │   1.jpg
|   │   |       │   │   ...
|   │   |       │   ...
|   │   └───scans_raw
|   │   |   └───scene0000_00
|   │   |       └───scene0000_00.aggregation.json
|   │   |       └───scene0000_00_vh_clean_2.labels.ply
|   │   |       └───scene0000_00_vh_clean_2.0.010000.segs.json
|   │   |       │   ...
|   |   └───scannetv2_test.txt
|   |   └───scannetv2_train.txt
|   |   └───scannetv2_val.txt
|   |   └───scannetv2-labels.combined.tsv
```
</details>

Next download the generated oriented boxes [annotations](https://drive.google.com/file/d/1lGNiUMcCe3fFOS7D3Zla_LwYJuQ3Sj96/view?usp=sharing) and put it under `PROJECT_PATH/data/scannet/`

OR you can run the [data preparation](scripts/scannet_preprocessing/README.md) script by yourself.


### Inference on ScanNet val-set
```bash
python eval.py --cfg ./config/eval.yaml CHECKPOINT_PATH ./checkpoint/parq_release.ckpt
```

### Training on ScanNet
Training with 8 gpus:
```bash
python train.py --cfg ./config/train.yaml TRAINER.GPUS 8
```


<!-- ## Coordinates illustration for ScanNet
World coordinate: ScanNet world coordinate  
Camera coordinate: Camera coodinate with OpenCV format (face forward +z, right: +x)  
PseudoCam coordinate: gravity-aligned camera coordinate (rotate camera coodinate to make the coordinate gravity-aligned)  
Local coordinate: the PseudoCam coordinate of middle frame   -->

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{xie2023parq,
  title={Pixel-Aligned Recurrent Queries for Multi-View {3D} Object Detection},
  author={Xie, Yiming and Jiang, Huaizu and Gkioxari, Georgia and Straub, Julian},
  booktitle={ICCV},
  year={2023}
}
```

## License
The majority of PARQ is relased under the MIT License. 
[LICENSE-MIT file](LICENSE-MIT) is for file `model/transformer_parq.py`.
[LICENSE file](LICENSE) is for other files.

## Acknowledgment
We want to thank the following contributors that our code is based on:
[DETR](https://github.com/facebookresearch/detr),
[VoteNet](https://github.com/facebookresearch/votenet),
[RotationContinuity](https://github.com/papagina/RotationContinuity),
[Pixloc](https://github.com/cvg/pixloc/blob/master/pixloc/pixlib/geometry/wrappers.py)
.
