################################################################################
# Imports: 

import wandb
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# WP - GAN :

import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from wpgan_model import Discriminator, Generator

# GLOW :

import json
import shutil
import random
from itertools import islice

import torch.optim as optim
import torch.utils.data as data

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss
from datasets import get_CIFAR10, get_SVHN, get_GMMSD, postprocess
from glow_model import Glow

################################################################################

# os.makedirs("images", exist_ok=True)

################################################################################

# Main loop : 

# TODO : Import models Descriminator and Generator . 

def main(
    # Glow : 
    dataset,
    dataroot,
    download,
    augment,
    batch_size,
    eval_batch_size,
    n_epochs,
    saved_model,
    seed,
    hidden_channels,
    K,
    L,
    actnorm_scale,
    flow_permutation,
    flow_coupling,
    LU_decomposed,
    learn_top,
    y_condition,
    y_weight,
    max_grad_clip,
    max_grad_norm,
    lr,
    n_workers,
    cuda,
    n_init_batches,
    output_dir,
    saved_optimizer,
    warmup,
    
    # WP-GAN:

    b1,
    b2,
    n_cpu,
    latent_dim,
    img_size,
    channels,
    n_critic,
    clip_value,
    sample_interval,

):
    # Main loop:
    
    ################################################################################
    # Configure data loader
    def check_dataset(dataset, dataroot, augment, download, batch_size):
        if dataset == "cifar10":
            cifar10 = get_CIFAR10(augment, dataroot, download)
            input_size, num_classes, train_dataset, test_dataset = cifar10
        if dataset == "svhn":
            svhn = get_SVHN(augment, dataroot, download)
            input_size, num_classes, train_dataset, test_dataset = svhn
        if dataset == "gmmsd":
            gmmsd = get_GMMSD(augment, dataroot, download, batch_size)
            input_size, num_classes, train_dataset, test_dataset = gmmsd
        return input_size, num_classes, train_dataset, test_dataset

    ds = check_dataset(dataset, dataroot, augment, download, batch_size)
    image_shape, num_classes, train_dataset, test_dataset = ds

    multi_class = False
    if dataset != 'gmmsd' :
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            drop_last=True,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=n_workers,
            drop_last=False,
        )
        
    else : 

        test_loader = test_dataset
        train_loader = train_dataset   

    ################################################################################
    # Train : 

    img_shape = (img_size, img_size, channels)

    cuda = True if torch.cuda.is_available() else False

    # Loss weight for gradient penalty
    lambda_gp = 10

    # Initialize generator and discriminator
    
    generator = Glow(
        image_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        num_classes,
        learn_top,
        y_condition,
    )
    
    
    
    wandb.config = {"learning_rate": lr, "epochs": n_epochs, "batch_size": 64}
    # Switch GLOW by regular generator : 
    generator = Generator(img_shape, latent_dim)
    discriminator = Discriminator(img_shape)

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Optimizers:
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(opt.b1, opt.b2))    
#    optimizer_G = optim.Adamax(generator.parameters(), lr=5e-4, weight_decay=5e-5)
#    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup)  # noqa
#    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
    
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def sample_from_glow(model):
#        # TODO : Check if Glow postprocess function fit WPGAN Generator output(dimensions / normalization etc)
#        model.set_actnorm_init()
#        model = model.eval()
#        with torch.no_grad():
        y = None
        images = model(y_onehot=y, temperature=1, reverse=True)
        images = images.permute(0,2,3,1)
        return images
         
    def compute_gradient_penalty(D, real_samples, fake_samples):
    
        if real_samples.shape[0] < fake_samples.shape[0]:
          fake_samples = fake_samples[:real_samples.shape[0]]
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(train_loader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
                        
            # Generate a batch of images
            # fake_imgs = sample_from_glow(generator)
            
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
    
            # Generate a batch of images
            fake_imgs = generator(z)
            
            # Real images
            real_validity = discriminator(real_imgs)
            
            # Fake images
            fake_validity = discriminator(fake_imgs)
            
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
            
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            
            wandb.log({"d_loss": d_loss})

            
            
            # Train the generator every n_critic steps
            if i % n_critic == 0:
                
                generator.train()
                
                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                # fake_imgs = sample_from_glow(generator)
                
                # Generate a batch of images
                fake_imgs = generator(z)
                
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
                )

                
                if batches_done % sample_interval == 0:
                    
                    # Save samples from generator : 
                    
                    fake_imgs = postprocess(fake_imgs).permute(0,3,1,2)
                    grid = make_grid(fake_imgs[:30], nrow=6).permute(1,2,0)
                    
                    plt.figure(figsize=(10,10))
                    plt.imsave("./images/sample_glow_batch_%d.png" % batches_done, grid.cpu().numpy())

                    caption_str = "Epoch : " + str(epoch)
                    images = wandb.Image(grid.cpu().numpy(), caption=caption_str)
                    wandb.log({"Generator:": images})
                    
                    # Save training batch as refernce : 
                    
                    real_imgs = postprocess(real_imgs).permute(0,3,1,2)
                    grid = make_grid(real_imgs[:30], nrow=6).permute(1,2,0)
                    
                    plt.figure(figsize=(10,10))
                    plt.imsave("./images/gmmsd_example_batch_%d.png" % batches_done, grid.cpu().numpy())
                
                    caption_str = "Epoch : " + str(epoch)
                    images = wandb.Image(grid.cpu().numpy(), caption=caption_str)
                    wandb.log({"GMMSD:": images})
                    
                    

                batches_done = batches_done + n_critic
                wandb.log({"g_loss": g_loss})

################################################################################

# Arguments : 

if __name__ == "__main__":
    
    # WP-GAN : 

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")

    # GLOW : 

    parser.add_argument(
        "--dataset",
        type=str,
        default="gmmsd",
        choices=["cifar10", "svhn", "gmmsd"],
        help="Type of the dataset to be used.",
    )

    parser.add_argument("--dataroot", type=str, default="/home/dsi/eyalbetzalel/GlowGAN/data/gmmsd.npy", help="path to dataset")

    parser.add_argument("--download", action="store_true", help="downloads dataset")

    parser.add_argument(
        "--no_augment",
        action="store_false",
        dest="augment",
        help="Augment training data",
    )

    parser.add_argument(
        "--hidden_channels", type=int, default=128, help="Number of hidden channels - 512"
    )

    parser.add_argument("--K", type=int, default=8, help="Number of layers per block - 32 ")

    parser.add_argument("--L", type=int, default=1, help="Number of blocks - 3")

    parser.add_argument(
        "--actnorm_scale", type=float, default=1.0, help="Act norm scale"
    )

    parser.add_argument(
        "--flow_permutation",
        type=str,
        default="invconv",
        choices=["invconv", "shuffle", "reverse"],
        help="Type of flow permutation",
    )

    parser.add_argument(
        "--flow_coupling",
        type=str,
        default="affine",
        choices=["additive", "affine"],
        help="Type of flow coupling",
    )

    parser.add_argument(
        "--no_LU_decomposed",
        action="store_false",
        dest="LU_decomposed",
        help="Train with LU decomposed 1x1 convs",
    )

    parser.add_argument(
        "--no_learn_top",
        action="store_false",
        help="Do not train top layer (prior)",
        dest="learn_top",
    )

    parser.add_argument(
        "--y_condition", action="store_true", help="Train using class condition"
    )

    parser.add_argument(
        "--y_weight", type=float, default=0.01, help="Weight for class condition loss"
    )

    parser.add_argument(
        "--max_grad_clip",
        type=float,
        default=0,
        help="Max gradient value (clip above - for off)",
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0,
        help="Max norm of gradient (clip above - 0 for off)",
    )

    parser.add_argument(
        "--n_workers", type=int, default=1, help="number of data loading workers"
    )

    # parser.add_argument(
    #     "--batch_size", type=int, default=64, help="batch size used during training"
    # )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=512,
        help="batch size used during evaluation",
    )

    # parser.add_argument(
    #     "--epochs", type=int, default=250, help="number of epochs to train for"
    # )

    # parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")

    parser.add_argument(
        "--warmup",
        type=float,
        default=5,
        help="Use this number of epochs to warmup learning rate linearly from zero to learning rate",  # noqa
    )

    parser.add_argument(
        "--n_init_batches",
        type=int,
        default=8,
        help="Number of batches to use for Act Norm initialisation",
    )

    parser.add_argument(
        "--no_cuda", action="store_false", dest="cuda", help="Disables cuda"
    )

    parser.add_argument(
        "--output_dir",
        default="output/",
        help="Directory to output logs and model checkpoints",
    )

    parser.add_argument(
        "--fresh", action="store_true", help="Remove output directory before starting"
    )

    parser.add_argument(
        "--saved_model",
        default="",
        help="Path to model to load for continuing training",
    )

    parser.add_argument(
        "--saved_optimizer",
        default="",
        help="Path to optimizer to load for continuing training",
    )

    wandb.init(project="GlowGAN", entity="eyalb")

    parser.add_argument("--seed", type=int, default=0, help="manual seed")

    opt = parser.parse_args()

    kwargs = vars(opt)
    del kwargs["fresh"]

    # with open(os.path.join(args.output_dir, "hparams.json"), "w") as fp:
    #     json.dump(kwargs, fp, sort_keys=True, indent=4)
    
    main(**kwargs)
    print(opt)