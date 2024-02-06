"""Script for transfer learning ResNet50 with IMAGENET1K_V2 weights on Russian art dataset"""

"""
1. Import libraries, data, constants
1.1 Import libs
"""

import os
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms

from src.initial_model_utils import init_model, set_seed
from src.train_utils import ArtDataset, train_model

"""
1.2 Define constants
"""

# fixed seed in order to reproduce results
RANDOM_STATE = 139

# resnet input image size
IMG_SIZE = 224

# amount of art classes of dataset
NUM_CLASSES = 35

# loaders params
BATCH_SIZE = 32
NUM_WORKERS = 4

# set to true if first run to split data
SPLIT_DATA_FOLDERS = False

# path to original data
TRAIN_DATASET = "./data/train/"
ORIGIN_TRAIN_CSV = "./data/private_info/train.csv"

# path to new splitted data and information
# train path remains same
TEST_DATASET = "./data/test/"
TRAIN_CSV = "./data/private_info/new_train.csv"
TEST_CSV = "./data/private_info/new_test.csv"

# path to save model's weight
MODEL_WEIGHTS = "./data/weights/resnet50_tl_68.pt"

"""
1.3 Set libs setting
"""
# Fixed all random states
set_seed(RANDOM_STATE)

"""
1.4 Split initial data to train and test parts
"""

# Split default data into train and test files
if SPLIT_DATA_FOLDERS:
    # original dataset
    train_data_info = pd.read_csv(ORIGIN_TRAIN_CSV, sep="\t")

    # extract id, classes
    indices = train_data_info.index
    labels = train_data_info.label_id

    # stratified split due to class imbalance
    ind_train, ind_test, _, _ = train_test_split(
        indices, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )

    # new datasets info
    new_train, new_test = (
        train_data_info.loc[ind_train].reset_index(drop=True),
        train_data_info.loc[ind_test].reset_index(drop=True),
    )

    # move test photos from original train to new test folder
    source_dir = TRAIN_DATASET
    target_dir = TEST_DATASET
    file_names = new_test["image_name"].values

    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), target_dir)

    new_train.to_csv(TRAIN_CSV, index=False, sep="\t")
    new_test.to_csv(TEST_CSV, index=False, sep="\t")


"""
1.5 Import images data
"""

# define base transforms (from ResNet50 docs)
base_transforms = transforms.Compose(
    [
        transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
        transforms.Lambda(lambda x: np.array(x, dtype="float32") / 255),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# add some random augmentations in order to prevent overfitting
augmentations = transforms.RandomChoice(
    [
        transforms.Compose(
            [
                transforms.Resize(size=300, max_size=301),
                transforms.CenterCrop(size=300),
                transforms.RandomCrop(250),
            ]
        ),
        transforms.RandomRotation(degrees=(-25, 25)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
)

# final compose and datasets
train_transform = transforms.Compose([augmentations, base_transforms])
test_transform = base_transforms

train_dataset = ArtDataset(TRAIN_DATASET, TRAIN_CSV, train_transform)
test_dataset = ArtDataset(TEST_DATASET, TEST_CSV, test_transform)


"""
2. Training process
2.1 Resolve class imbalance by using weightened sampler
"""

# class2weight mapper
weight_mapper = compute_class_weight(
    "balanced", classes=np.unique(train_dataset.targets), y=train_dataset.targets
)

samples_weight = np.array([weight_mapper[t] for t in train_dataset.targets])
samples_weight = torch.from_numpy(samples_weight)

sampler = WeightedRandomSampler(
    samples_weight.type("torch.DoubleTensor"), len(samples_weight)
)

"""
2.2 Loaders, model and other pretraining objects
"""
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    BATCH_SIZE=BATCH_SIZE,
    NUM_WORKERS=NUM_WORKERS,
    sampler=sampler,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, BATCH_SIZE=BATCH_SIZE, shuffle=False, NUM_WORKERS=NUM_WORKERS
)

loaders = {"train": train_loader, "val": test_loader}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = init_model(device, num_classes=NUM_CLASSES)

optimizer = torch.optim.Adam(
    params=[param for param in model.parameters() if param.requires_grad],
    lr=1e-3,
    weight_decay=5e-4,
)

# simple, but actually works
sheduler = torch.optim.lr_scheduler.StepLR(
    optimizer=optimizer, step_size=10, gamma=0.5, verbose=True
)

criterion = nn.CrossEntropyLoss()

"""
2.3 Training and results
"""
train_results = train_model(
    model,
    loaders,
    criterion,
    optimizer,
    phases=["train", "val"],
    sheduler=sheduler,
    num_epochs=80,
)

torch.save(model.state_dict(), MODEL_WEIGHTS)
