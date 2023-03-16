import logging
import os
import tempfile
import shutil
import sys
from abc import ABC, abstractmethod
from monai.utils import first, set_determinism
import matplotlib.pyplot as plt
import SimpleITK as sitk  # noqa: N813
import numpy as np
import itk
from PIL import Image
import tempfile
from monai.data import ITKReader, PILReader
from monai.handlers.utils import from_engine
from Swin_VFTR import Swin_VFTR
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, MeanIoU, SSIMMetric
from monai.losses import DiceCELoss
from sklearn.model_selection import KFold
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
from monai.apps import CrossValidation
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, create_test_image_3d
from monai.engines import (
    EnsembleEvaluator,
    SupervisedEvaluator,
    SupervisedTrainer
)
from monai.handlers import MeanDice, StatsHandler, ValidationHandler, from_engine
from monai.inferers import SimpleInferer, SlidingWindowInferer, sliding_window_inference

from monai.losses import DiceLoss
from monai.networks.nets import UNet
import torch
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    Activationsd,
    EnsureChannelFirstd,
    AsDiscreted,
    Compose,
    Resized,
    LoadImaged,
    LoadImage,
    MeanEnsembled,
    RandCropByPosNegLabeld,
    RandRotate90d,
    RandCropByPosNegLabeld,
    SaveImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    VoteEnsembled,
    Flipd,
    ResizeWithPadOrCropd,
    Transposed,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ToTensor,
)
from monai.utils import set_determinism
%matplotlib inline
print_config()
