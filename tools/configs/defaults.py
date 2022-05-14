import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.MODEL = CN()
_C.MODEL.SMOKE_ON = True
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHT = ""

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------

_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.HEIGHT_TRAIN = 384

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.HEIGHT_TRAIN = 384
# Maximum size of the side of the image during training
_C.INPUT.WIDTH_TRAIN = 1280
# Size of the smallest side of the image during testing
_C.INPUT.HEIGHT_TEST = 384
# Maximum size of the side of the image during testing
_C.INPUT.WIDTH_TEST = 1280

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# Root path of dataset files.
_C.DATASETS.ROOT = ""
# train split tor dataset
_C.DATASETS.TRAIN_SPLIT = ""
# train split tor dataset
_C.DATASETS.EVAL_SPLIT = ""
# test split for dataset
_C.DATASETS.TEST_SPLIT = ""
# class types
_C.DATASETS.DETECT_CLASSES = ("Car",)
_C.DATASETS.MAX_OBJECTS = 30

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
_C.MODEL.BACKBONE.CONV_BODY = "DLA-34-DCN"
_C.MODEL.BACKBONE.DOWN_RATIO = 4