from yacs.config import CfgNode as CN

_C = CN()


# general configs
_C.SEED = 100
_C.MEMORY_GB = 230
_C.CHECKPOINT_PATH = None
_C.PRETRAINED_PATH = None # Load the pretrained model, which may have different keys from the current model
_C.NAME = "release"
_C.LOG_PATH = "/work/vig/yimingx/parq_logs"
_C.TAG = ""
_C.LOG_IMAGES = True
_C.LOG_IMAGES_FREQUENCY = 4800
_C.LOG_RANK_ZERO_ONLY = True


# lightning trainer
_C.TRAINER = CN()
_C.TRAINER.PROFILER = 'simple'
_C.TRAINER.ACCELERATOR = 'gpu'
_C.TRAINER.GPUS = 2
_C.TRAINER.NUM_NODES = 1
_C.TRAINER.ACCUMULATE_GRAD_BATCHES = 1
_C.TRAINER.MAX_EPOCHS = 100
_C.TRAINER.LOG_EVERY_N_STEPS = 100
_C.TRAINER.GRADIENT_CLIP_VAL = 1.0
_C.TRAINER.RELOAD_DATALOADERS_EVERY_N_EPOCHS = 0
_C.TRAINER.REPLACE_SAMPLER_DDP = True
_C.TRAINER.OVERFIT_BATCHES = 0.0
_C.TRAINER.AUTO_SCALE_BATCH_SIZE = 'binsearch'
_C.TRAINER.CHECK_VAL_EVERY_N_EPOCH = 1
_C.TRAINER.PRECISION = 32
_C.TRAINER.VAL_CHECK_INTERVAL = 1.0
_C.TRAINER.LIMIT_VAL_BATCHES = 1.0
_C.TRAINER.LIMIT_TRAIN_BATCHES = 1.0


# callback
_C.CALLBACK = CN()
_C.CALLBACK.MONITOR = 'val/metrics/0.5_f1' # name of the logged metric which determines when model is improving
_C.CALLBACK.SAVE_TOP_K = 3 # save k best models (determined by above metric)
_C.CALLBACK.SAVE_LAST = True # additionaly always save model from last epoch
_C.CALLBACK.VERBOSE = False
_C.CALLBACK.DIRPATH = None
_C.CALLBACK.FILENAME = None
_C.CALLBACK.AUTO_INSERT_METRIC_NAME = False
_C.CALLBACK.MODE = 'max'

# optimization
_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = 'adamw'
_C.OPTIMIZER.LEARNING_RATE = 1e-4 # cosine-annealing with restarts; by default just a single cycle
_C.OPTIMIZER.CYCLE_MULT = 1
_C.OPTIMIZER.WARMUP_EPOCHS = 0
_C.OPTIMIZER.NUM_RESTARTS = 1
_C.OPTIMIZER.IGNORE_FROZEN_PARAMS = True # filter out all model parameters with requires_grad=False
_C.OPTIMIZER.AUTOSCALE_LR = True # Auto-scale the learning rate based on "Accurate Large Minibatch SGD ..."


# datamodule
_C.DATAMODULE = CN()
_C.DATAMODULE.DATA_PATH = './data/scannet/scans'
_C.DATAMODULE.TRAIN_ANNOTATION_PATH = './data/scannet/scan2cad_box3d_anno_view3_overlap/scannet_train_gt_roidb.pkl'
_C.DATAMODULE.VAL_ANNOTATION_PATH = './data/scannet/scan2cad_box3d_anno_view3_overlap/scannet_val_gt_roidb.pkl'
_C.DATAMODULE.BATCH_SIZE = 1
_C.DATAMODULE.NUM_WORKERS = 1
_C.DATAMODULE.NUM_FRAMES_PER_SNIPPET = 3
_C.DATAMODULE.SHUFFLE = True
_C.DATAMODULE.GRAVITY_ALIGNED = True

# model
feature_dim = 1024
_C.MODEL = CN()
# 2d backbone
_C.MODEL.BACKBONE2D = CN()
_C.MODEL.BACKBONE2D.RESNET_NAME = 'resnet50'
_C.MODEL.BACKBONE2D.LAYER = 0
_C.MODEL.BACKBONE2D.FREEZE = False

# tokenizer
_C.MODEL.TOKENIZER = CN()
_C.MODEL.TOKENIZER.OUT_CHANNELS = feature_dim
_C.MODEL.TOKENIZER.PATCH_SIZE = 1 # patch size when using tokenization
_C.MODEL.TOKENIZER.RAY_POINTS_SCALE = [-2, 2, -1.5, 0, 0.25, 4.25] # [min_x, max_x, min_y, max_y, min_z, max_z] used to normalize the points along each ray
_C.MODEL.TOKENIZER.NUM_SAMPLES = 64 # number of ray points
_C.MODEL.TOKENIZER.MIN_DEPTH = 0.25 # min depth of ray points
_C.MODEL.TOKENIZER.MAX_DEPTH = 5.25 # max depth of ray points

# PARQ decoder
_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.DIM_IN = feature_dim
_C.MODEL.DECODER.NUM_QUERIES = 128
_C.MODEL.DECODER.NUM_SEMCLS = 9 # number of semantic classes, e.g. 9 on scannet, 17 on ARKitScenes
_C.MODEL.DECODER.BOX_SIZE = [1, 1, 1]
_C.MODEL.DECODER.LOSS_WEIGHT = [5.0, 5.0, 5.0, 1.0] # center, size, rotation, category
_C.MODEL.DECODER.CONF_THRESH = 0.1 # confidence threshold, box will be filtered if its score is lower than this thresh
_C.MODEL.DECODER.MEAN_SIZE_PATH = None # the path to the file of the mean size for each object in scannet
_C.MODEL.DECODER.EVAL_TYPE = 'f1' # using f1 score to evaluate the model, follow ODAM https://arxiv.org/abs/2108.10165
_C.MODEL.DECODER.ENABLE_NMS = True # if True, the model will use nms to filter out potential duplicate boxes
_C.MODEL.DECODER.SHARE_MLP_HEADS = True # detection head share weight in each iteration
_C.MODEL.DECODER.FOR_VIS = False # if True, the model will preserve all outputs from one snippet for visualization without ant filtering, e.g. filter out far-away boxes or using nms
_C.MODEL.DECODER.TRACK_SCALE = [-1.5, 1.5, -2, 1, 0, 2] # only track boxes within this scale, because the quality of predictions is usually bad due to truncation... also used for other baselines

_C.MODEL.DECODER.TRANSFORMER = CN()
_C.MODEL.DECODER.TRANSFORMER.DEC_DIM = feature_dim # decoder dimension
_C.MODEL.DECODER.TRANSFORMER.DEC_HEADS = 4 # number of decoder heads
_C.MODEL.DECODER.TRANSFORMER.DEC_FFN_DIM = 768 # decoder feedforward dimension
_C.MODEL.DECODER.TRANSFORMER.DEC_LAYERS = 8 # number of decoder layers
_C.MODEL.DECODER.TRANSFORMER.DROPOUT_RATE = 0.1 # dropout_rate=0.0 disables dropout
_C.MODEL.DECODER.TRANSFORMER.QUERIES_DIM = feature_dim # same as dec_dim but needed for other configs
_C.MODEL.DECODER.TRANSFORMER.SCALE = _C.MODEL.TOKENIZER.RAY_POINTS_SCALE
_C.MODEL.DECODER.TRANSFORMER.SHARE_WEIGHTS = True




def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


def check_config(cfg):
    pass