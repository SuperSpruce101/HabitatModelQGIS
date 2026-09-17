import segmentation_models_pytorch as smp
import torch.nn as nn


def create_model(
    input_channels: int = 3,
    number_of_classes: int = 5,
) -> nn.Module:

    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=input_channels,
        classes=number_of_classes,
    )