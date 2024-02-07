"""Useful functions to initialize model and process"""

import random

import numpy as np
import torch
import torch.nn as nn
import torchvision


def set_seed(seed: int) -> None:
    """Function to freeze all libs random states

    Parameters
    ----------
    seed : int
        Random state
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_requires_grad(model: nn.Module, value: bool = False) -> None:
    """Function to turn on/off all model's weight's gradients

    Parameters
    ----------
    model : torch.nn.Module
        Model to change grads
    value : bool, optional
        Turn on/off gradients of weights, by default False
    """
    for param in model.parameters():
        param.requires_grad = value


def init_model(
    device: torch.device, num_classes: int, pretrained: bool = False
) -> nn.Module:
    """Function to initialize classifier by using backbone of pretrained ResNet50

    Parameters
    ----------
    device : torch.device
        User's device (cpu or gpu)
    num_classes : int
        Number of classed to classify
    pretrained : bool
        Download pretrained model

    Returns
    -------
    nn.Module
        New model
    """

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = torchvision.models.resnet50(weights=weights)

    # freeze all weight's gradients
    set_requires_grad(model, False)

    # new head classifier
    model.fc = nn.Sequential(
        *[
            nn.Dropout(0.735),
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        ]
    )

    model = model.to(device)
    return model
