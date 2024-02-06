"""Useful classes / functions for training process"""

import os
import time
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from Ipython.display import clear_output
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from tqdm import tqdm


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, torch.data.utils.DataLoader],
    criterion: nn.Module,
    optimizer,
    phases: List[str],
    device: torch.device,
    sheduler: Union[None, nn.Module] = None,
    num_epochs: int = 3,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Function to train custom image classifier

    Parameters
    ----------
    model : nn.Module
        Model with new classifier, backbone pretrained
    dataloaders : Dict[str, torch.data.utils.DataLoader]
        Loaders of train and test data
    criterion : nn.Module
        Function to optimize
    optimizer : _type_
        Selected optimized. Preferred Adam
    phases : List[str]
        Train and
    device : torch.device
        User device where to store data and model
    sheduler : Union[None, nn.Module], optional
        Learning rate sheduler, by default None
    num_epochs : int, optional
        Epochs to train, by default 3

    Returns
    -------
    Tuple[nn.Module, Dict[str, List[float]]]
        Trained model and metric history
    """

    start_time = time.time()

    metric_history = {k: list() for k in phases}
    loss_history = {k: list() for k in phases}

    for epoch in range(1, num_epochs + 1):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("-" * 10)

        # each epoch has a training and validation phase
        for phase in phases:
            if phase == "train":
                # set model to training mode
                model.train()
            else:
                # set model to evaluate mode
                model.eval()

            running_loss = 0.0
            phase_preds, phase_labels = [], []

            # tterate over data
            n_batches = len(dataloaders[phase])
            for inputs, labels in tqdm(dataloaders[phase], total=n_batches):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in train phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                phase_preds.extend(preds.detach().cpu().numpy())
                phase_labels.extend(labels.detach().cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_f1 = f1_score(phase_labels, phase_preds, average="macro")

            print("{} Loss: {:.4f} f1: {:.4f}".format(phase, epoch_loss, epoch_f1))
            loss_history[phase].append(epoch_loss)
            metric_history[phase].append(epoch_f1)

        # run sheduler after validation phase
        if sheduler is not None and phase == "val":
            sheduler.step()

        plot_train_process(
            cur_epoch_num=epoch,
            loss_history=loss_history,
            metric_history=metric_history,
        )

    time_elapsed = time.time() - start_time
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return model, metric_history


class ArtDataset(Dataset):
    def __init__(self, root_dir, csv_path=None, transform=None):
        self.transform = transform
        self.files = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]
        self.targets = None
        if csv_path:
            df = pd.read_csv(csv_path, sep="\t")
            self.targets = df["label_id"].tolist()
            self.files = [
                os.path.join(root_dir, fname) for fname in df["image_name"].tolist()
            ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert("RGB")
        target = self.targets[idx] if self.targets else -1
        if self.transform:
            image = self.transform(image)
        return image, target


def plot_train_process(
    cur_epoch_num: int,
    loss_history: Dict[str, List[float]],
    metric_history: Dict[str, List[float]],
) -> None:
    """Function to plot losses and metrics on training process

    Plots 2 graphics with loss and metrics on train and test data

    Parameters
    ----------
    cur_epoch_num : int
        Current epoch number
    loss_history : Dict[str, List[float]]
        Storage of train and validation loss history
    metric_history : Dict[str, List[float]]
        Storage of train and validation metric history
    """

    clear_output(wait=True)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 15))

    marker_style = {
        "marker": "o",
        "markersize": 5,
        "markerfacecolor": "black",
    }

    train_style = {
        "label": "Train value",
        "color": "b",
    } | marker_style

    val_style = {
        "label": "Test value",
        "color": "r",
    } | marker_style

    x_epoch = np.arange(1, cur_epoch_num + 1)

    sns.lineplot(ax=ax[0], x=x_epoch, y=loss_history["train"], **train_style)
    sns.lineplot(ax=ax[0], x=x_epoch, y=loss_history["val"], **val_style)

    sns.lineplot(ax=ax[1], x=x_epoch, y=metric_history["train"], **train_style)
    sns.lineplot(ax=ax[1], x=x_epoch, y=metric_history["val"], **val_style)

    ax[0].set_ylabel("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_title("Loss graph")
    ax[0].grid()

    ax[1].set_ylabel("Metric")
    ax[1].set_xlabel("Epoch")
    ax[1].set_title("F1-macro score")
    ax[1].grid()

    plt.show()
