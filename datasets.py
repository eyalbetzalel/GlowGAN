from pathlib import Path

import torch
import torch.nn.functional as F
import h5py
import numpy as np
from torchvision import transforms, datasets

n_bits = 8


def preprocess(x):
    # Follows:
    # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78

    x = x * 255  # undo ToTensor scaling to [0,1]

    n_bins = 2 ** n_bits
    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5

    return x


def postprocess(x):
    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    x = x * 2 ** n_bits
    return torch.clamp(x, 0, 255).byte()

def one_hot_encode(target):
    """
    One hot encode with fixed 10 classes
    Args: target           - the target labels to one-hot encode
    Retn: one_hot_encoding - the OHE of this tensor
    """
    num_classes = 10
    one_hot_encoding = F.one_hot(torch.tensor(target),num_classes)

    return one_hot_encoding


def get_CIFAR10(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10

    if augment:
        transformations = [
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])


    path = Path(dataroot) / "data" / "CIFAR10"
    train_dataset = datasets.CIFAR10(
        path,
        train=True,
        transform=train_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    test_dataset = datasets.CIFAR10(
        path,
        train=False,
        transform=test_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    return image_shape, num_classes, train_dataset, test_dataset


def get_SVHN(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1))]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])


    path = Path(dataroot) / "data" / "SVHN"
    train_dataset = datasets.SVHN(
        path,
        split="train",
        transform=train_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    test_dataset = datasets.SVHN(
        path,
        split="test",
        transform=test_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    return image_shape, num_classes, train_dataset, test_dataset

def transform_cluster_to_image(data):
    log_dir = '/home/dsi/eyalbetzalel/pytorch-generative-v6/image_test'
    pathToCluster = r"/home/dsi/eyalbetzalel/image-gpt/downloads/kmeans_centers.npy"
    clusters = torch.from_numpy(np.load(pathToCluster)).float()
    data = torch.reshape(torch.from_numpy(train), [-1, 32, 32])
    # train = train[:,None,:,:]
    sample = torch.reshape(torch.round(127.5 * (clusters[data.long()] + 1.0)), [data.shape[0], 3, 32, 32]).to('cuda')
    return sample

def get_GMMSD(path_train, path_test):

    with h5py.File(path_train, "r") as f:
        a_group_key = list(f.keys())[0]
        train = list(f[a_group_key])
    with h5py.File(path_test, "r") as f:
        a_group_key = list(f.keys())[0]
        test = list(f[a_group_key])
    test_images = transform_cluster_to_image(test)
    train_images = transform_cluster_to_image(train)
    import ipdb; ipdb.set_trace()
    return train_images, test_images

