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
from swinvftr.swin_vftr import SwinVFTR
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
print_config()

def train():
    if not os.path.exists('spectralis_swin-vfstr_MSA_CM_weights'):
        os.makedirs('spectralis_swin-vfstr_MSA_CM_weights')
        
    os.environ['MONAI_DATA_DIRECTORY'] = '/nfs/cc-filer/home/khondkerfarihah/SwinVFTR/spectralis_swin-vfstr_MSA_CM_weights'
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)
    
    filenames = []
    labelnames = []
    with open('retouch_train_fold0.txt', 'r') as f:
        for line in f.readlines():
            filenames.append(line.strip('\n').split(' ')[0])
            labelnames.append(line.strip('\n').split(' ')[1])
    train_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(filenames, labelnames)
    ]

    filenames = []
    labelnames = []
    with open('retouch_test_fold0.txt', 'r') as f:
        for line in f.readlines():
            filenames.append(line.strip('\n').split(' ')[0])
            labelnames.append(line.strip('\n').split(' ')[1])
    test_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(filenames, labelnames)
    ]
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Transposed(keys=["image", "label"],indices=(0,2,1,3)),
            Resized(keys=["image", "label"],spatial_size=(496, 512, 49), size_mode='all', mode='area', align_corners=None, anti_aliasing=False, anti_aliasing_sigma=None),
            ScaleIntensityRanged(
                keys=["image"], a_min=0.0, a_max=65535.0,
                b_min=0.0, b_max=1.0
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Flipd(keys=["image", "label"],spatial_axis=0),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=[256, 256, 32],
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Resized(keys=["image", "label"],spatial_size=(496, 512, 49), size_mode='all', mode='area', align_corners=None, anti_aliasing=False, anti_aliasing_sigma=None),
            Transposed(keys=["image", "label"],indices=(0,2,1,3)),
            ScaleIntensityRanged(
                keys=["image"], a_min=0.0, a_max=65535.0,
                b_min=0.0, b_max=1.0
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Flipd(keys=["image", "label"],spatial_axis=0),
        ]
    )
    train_ds = CacheDataset(
    data=train_dicts, transform=train_transforms,
    cache_rate=1.0, num_workers=2)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)

    val_ds = CacheDataset(
        data=test_dicts, transform=val_transforms, cache_rate=1.0, num_workers=2)

    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)
    
    device = torch.device("cuda:0")


    model = SwinVFTR(
        img_size=(256, 256, 32),
        in_channels=1,
        out_channels=4,
        depths=(2, 2, 2),
        num_heads=(2,2,2,2),
        feature_size=24,
        use_checkpoint=False,
    ).to(device)

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)

    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    
    torch.autograd.set_detect_anomaly(True)
    max_epochs = 600
    val_interval = 2
    best_metric = -1
    #l1_loss = 0
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=4)])
    post_label = Compose([AsDiscrete(to_onehot=4)])
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs =  model(inputs,eval_bool=True)
            l1_loss =0
            #for i in range(3):
            #    l1_loss += l1_loss_function(outputs[1][0][i],outputs[2][0][i])
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            #epoch_loss += l1_loss.item()
            del outputs
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (496, 512, 32)
                    sw_batch_size = 1
                    #val_outputs =  model(val_inputs,eval_bool=True)
                    eval_bool={'eval_bool':True} # True
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model,**eval_bool)
                    #val_outputs= SimpleInferer(val_inputs, network=model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        root_dir, "best_spectralis_fold0.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
                
        plt.figure("train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        plt.xlabel("epoch")
        plt.plot(x, y)
        plt.subplot(1, 2, 2)
        plt.title("Val Mean Dice")
        x = [val_interval * (i + 1) for i in range(len(metric_values))]
        y = metric_values
        plt.xlabel("epoch")
        plt.plot(x, y)
        plt.savefig('train:fold1')
if __name__ == "__main__":
    train()

    
