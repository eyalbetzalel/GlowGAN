from pathlib import Path
import torch
import torch.nn.functional as F
import h5py
import numpy as np
from torchvision import transforms, datasets
from torchvision.utils import make_grid

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
    x = x / 2
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

#def gmmsd_save_grid(img):
#    import ipdb; ipdb.set_trace()
#    import matplotlib.pyplot as plt
#    img_p = postprocess(img)
#    yos = img_p[0]
#    #yos = yos.permute(1,2,0)
#    plt.imsave('test4.png', yos.cpu().numpy())
#    return

def transform_cluster_to_image(data_input):
    pathToCluster = r"/home/dsi/eyalbetzalel/image-gpt/downloads/kmeans_centers.npy"
    clusters = torch.from_numpy(np.load(pathToCluster)).float()
    data = torch.reshape(torch.from_numpy(data_input), [-1, 32, 32])
    # train = train[:,None,:,:]
    # sample = torch.reshape(torch.round(127.5 * (clusters[data.long()] + 1.0)), [data.shape[0],3 ,32, 32]).to('cuda')
    # sample = torch.reshape(clusters[data.long()], [data.shape[0],3 ,32, 32]).to('cuda')
    sample = clusters[data.long()].to('cuda')
    # sample = sample.permute(0,3,1,2)
    # gmmsd_save_grid(sample)
    return sample

    
def get_GMMSD(augment, dataroot, download, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_shape = (32, 32, 3)
    num_classes = 1  # Not real value. TODO : Check if this is relevant. 
    dataset = np.load(dataroot)
    images_cluster = dataset[:,:1024] # 1024 is the image, and 1025 is the log likelihood in this data structure (next line)
    ll = dataset[:,-1]
    images = transform_cluster_to_image(images_cluster)
    ind = int(images.shape[0]*0.7)
    # trainX = images[:ind]
    trainX = images[:1024]
    # trainY = torch.tensor(ll[:ind], device=device)
    trainY = torch.ones(trainX.shape[0])
    testX = images[ind+1:]
    # testY = torch.tensor(ll[ind+1:], device=device)
    testY = torch.ones(testX.shape[0])
    train_dataset = torch.utils.data.TensorDataset(trainX, trainY)
    test_dataset = torch.utils.data.TensorDataset(testX, testY)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size)
    return image_shape, num_classes, train_loader, test_loader

