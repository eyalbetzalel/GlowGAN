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
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from wpgan_model import Discriminator, Generator

# REALNVP :

import argparse
import torch, torchvision
import torch.distributions as distributions
import torch.optim as optim
import torchvision.utils as utils
import torch.utils.data as data
import numpy as np
import realnvp, data_utils

# Dataset : 

from datasets import get_CIFAR10, get_GMMSD, get_SVHN, preprocess, postprocess


################################################################################

class Hyperparameters():
    def __init__(self, base_dim, res_blocks, bottleneck, 
        skip, weight_norm, coupling_bn, affine):
        """Instantiates a set of hyperparameters used for constructing layers.

        Args:
            base_dim: features in residual blocks of first few layers.
            res_blocks: number of residual blocks to use.
            bottleneck: True if use bottleneck, False otherwise.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            coupling_bn: True if batchnorm coupling layer output, False otherwise.
            affine: True if use affine coupling, False if use additive coupling.
        """
        self.base_dim = base_dim
        self.res_blocks = res_blocks
        self.bottleneck = bottleneck
        self.skip = skip
        self.weight_norm = weight_norm
        self.coupling_bn = coupling_bn
        self.affine = affine

################################################################################

# Main loop : 

# TODO : Import models Descriminator and Generator . 

def main(

    # realnvp : 
    base_dim, 
    res_blocks, 
    bottleneck, 
    skip, 
    weight_norm, 
    coupling_bn, 
    affine, 
    lr, 
    momentum, 
    decay,
    sample_size,
    n_epochs,
    
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
    
    # Dataset :
    dataroot,
    download,
    augment,
    dataset, 
    batch_size,
    n_workers,

):

    # Main loop:
    hps = Hyperparameters(
    base_dim = base_dim, 
    res_blocks = res_blocks, 
    bottleneck = bottleneck, 
    skip = skip, 
    weight_norm = weight_norm, 
    coupling_bn = coupling_bn, 
    affine = affine)
    
    scale_reg = 5e-5    # L2 regularization strength

    # prefix for images and checkpoints
    filename = 'bs%d_' % batch_size \
             + 'normal_' \
             + 'bd%d_' % hps.base_dim \
             + 'rb%d_' % hps.res_blocks \
             + 'bn%d_' % hps.bottleneck \
             + 'sk%d_' % hps.skip \
             + 'wn%d_' % hps.weight_norm \
             + 'cb%d_' % hps.coupling_bn \
             + 'af%d' % hps.affine \
    
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
            batch_size=batch_size,
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Loss weight for gradient penalty
    lambda_gp = 10

    # Initialize generator and discriminator

    prior = distributions.Normal(   # isotropic standard normal distribution
        torch.tensor(0.).to(device), 
        torch.tensor(1.).to(device)
        )

    data_info = data_utils.DataInfo("gmmsd", 3, 32)
    
    generator = realnvp.RealNVP(datainfo=data_info, 
    prior=prior, 
    hps=hps).to(device)

    wandb.config = {"learning_rate": lr, "epochs": n_epochs, "batch_size": 64}
    
    # Switch REALNVP by regular generator : 
    # generator = Generator(img_shape, latent_dim)
    
    discriminator = Discriminator(img_shape).to(device)

    # Optimizers:
    
#    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(opt.b1, opt.b2))    
    optimizer_G = optim.Adamax(generator.parameters(), lr=lr, betas=(momentum, decay), eps=1e-7)
    
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def sample_from_realnvp(model, batch_size):
#        # TODO : Check if Glow postprocess function fit WPGAN Generator output(dimensions / normalization etc)
        model = model.eval()
        with torch.no_grad():
            samples = model.sample(batch_size)
            samples, _ = data_utils.logit_transform(samples, reverse=True)
            samples = samples.permute(0,2,3,1)
        
            # SAVE IMAGES FROM GENERATOR : 
            # utils.save_image(utils.make_grid(samples),
            #     './samples/' + dataset + '/' + filename + '_ep%d.png' % epoch)
        return samples
         
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
            fake_imgs = sample_from_realnvp(generator, batch_size)

            #####################################################################################################################
            # Sample noise as generator input - WGANGP Generator:
            # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
            # Generate a batch of images - WGANGP Generator
            # fake_imgs = generator(z)
            #####################################################################################################################

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
            
                for k in range(1): 
                    
                    # -----------------
                    #  Train Generator
                    # -----------------
    
                    # Generate a batch of images
                    
                    
                    
                    fake_imgs = sample_from_realnvp(generator, batch_size)
                    
                    generator.train()

                    # This line is important due to the need for the gradient calculation in the generator for this step! :
                    
                    z = generator(fake_imgs.permute(0,3,1,2))
                    
                    
                    # Generate a batch of images - WGANGP Generator
                    # fake_imgs = generator(z)
                    
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
                    plt.imsave("./images2/sample_glow_batch_%d.png" % batches_done, grid.cpu().numpy())

                    caption_str = "Epoch : " + str(epoch)
                    images = wandb.Image(grid.cpu().numpy(), caption=caption_str)
                    wandb.log({"Generator:": images})
                    
                    # Save training batch as refernce : 
                    
                    real_imgs = postprocess(real_imgs).permute(0,3,1,2)
                    grid = make_grid(real_imgs[:30], nrow=6).permute(1,2,0)
                    
                    plt.figure(figsize=(10,10))
                    plt.imsave("./images2/gmmsd_example_batch_%d.png" % batches_done, grid.cpu().numpy())
                
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

    # REAL - NVP :
     
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=32)
    parser.add_argument('--base_dim',
                        help='features in residual blocks of first few layers.',
                        type=int,
                        default=64)
    parser.add_argument('--res_blocks',
                        help='number of residual blocks per group.',
                        type=int,
                        default=8)
    parser.add_argument('--bottleneck',
                        help='whether to use bottleneck in residual blocks.',
                        type=int,
                        default=0)
    parser.add_argument('--skip',
                        help='whether to use skip connection in coupling layers.',
                        type=int,
                        default=1)
    parser.add_argument('--weight_norm',
                        help='whether to apply weight normalization.',
                        type=int,
                        default=1)
    parser.add_argument('--coupling_bn',
                        help='whether to apply batchnorm after coupling layers.',
                        type=int,
                        default=1)
    parser.add_argument('--affine',
                        help='whether to use affine coupling.',
                        type=int,
                        default=1)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--momentum',
                        help='beta1 in Adam optimizer.',
                        type=float,
                        default=0.9)
    parser.add_argument('--decay',
                        help='beta2 in Adam optimizer.',
                        type=float,
                        default=0.999)
                        
    parser.add_argument("--dataset",
                        type=str,
                        default="gmmsd",
                        choices=["cifar10", "svhn", "gmmsd"],
                        help="Type of the dataset to be used.",)

    parser.add_argument("--dataroot", type=str, default="/home/dsi/eyalbetzalel/GlowGAN/data/gmmsd.npy", help="path to dataset")

    parser.add_argument("--download", action="store_true", help="downloads dataset")

    parser.add_argument("--no_augment",
                        action="store_false",
                        dest="augment",
                        help="Augment training data",)
                        
    parser.add_argument("--n_workers", type=int, default=1, help="number of data loading workers")

    wandb.init(project="GlowGAN", entity="eyalb")

    

    opt = parser.parse_args()

    kwargs = vars(opt)

    # with open(os.path.join(args.output_dir, "hparams.json"), "w") as fp:
    #     json.dump(kwargs, fp, sort_keys=True, indent=4)
    
    main(**kwargs)
    print(opt)