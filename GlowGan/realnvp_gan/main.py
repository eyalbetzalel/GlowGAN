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

from datasets import get_CIFAR10, get_GMMSD, get_SVHN, preprocess, postprocess, postprocess_fake

# ImageGPT : 

from imagegpt.imagegpt import ImageGPT
from gmpm import *
import pandas as pd
import tensorflow as tf
import time
from js_test import *
import gc

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
    
    # GAN:
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

    # ImageGPT:
    n_gpu,
    tf_device,
    imagegpt_artifact,
    

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

    
    # Train : 

    def sample_from_realnvp(model, batch_size):
        model.train()
        samples = model.sample(batch_size)
        samples, _ = data_utils.logit_transform(samples, reverse=True)
        samples = samples.permute(0,2,3,1)
        return samples
        
    def sample_from_realnvp_for_js(model, batch_size):
        samples = sample_from_realnvp(model, batch_size)
        samples = samples.permute(0,3,1,2)
        # samples2, _ = data_utils.logit_transform(samples, reverse=True)
        log_prob = model.log_prob(samples)
        log_prob = log_prob / batch_size
        prob = torch.exp(log_prob)
        return samples, prob

    img_shape = (img_size, img_size, channels)
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prior = distributions.Normal(   # isotropic standard normal distribution
        torch.tensor(0.).to(device), 
        torch.tensor(1.).to(device)
        )
    data_info = data_utils.DataInfo("gmmsd", 3, 32)
    wandb.config.learning_rate = lr 
    wandb.config.epochs = n_epochs 
    wandb.config.batch_size =  batch_size

    # Initialize generator and discriminator:
    generator = realnvp.RealNVP(
        datainfo=data_info, 
        prior=prior, 
        hps=hps
        ).to(device)
    
    # Switch REALNVP by regular generator :     
    discriminator = Discriminator(img_shape).to(device)

    # Loss function : 
    adversarial_loss = torch.nn.BCELoss()
    # adversarial_loss = adversarial_loss.to(device)
    # Optimizers:
    # optimizer_G = optim.Adamax(generator.parameters(), lr=lr, betas=(momentum, decay), eps=1e-7)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(opt.b1, opt.b2))
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------
    columns=['Epoch','KL', 'Total Variation Distance', 'chi p', 'alpha 0.25', 'alpha 0.5', 'alpha 0.75', 'JS', 'G_Loss', 'FID', 'IS']
    df_res = pd.DataFrame(columns = columns)
    res_table = wandb.Table(columns=columns)
    cosine_similarity = torch.nn.CosineSimilarity()
    batches_done = 0
    image_gpt = ImageGPT(
        batch_size= batch_size,
        devices= tf_device,
        ckpt_path='/home/dsi/eyalbetzalel/GlowGAN/GlowGan/realnvp_gan/imagegpt/artifacts/model.ckpt-1000000/model.ckpt-1000000',
        color_cluster_path='/home/dsi/eyalbetzalel/GlowGAN/GlowGan/realnvp_gan/imagegpt/artifacts/kmeans_centers.npy',
        )
        
    lambda_gp = 10
    def compute_gradient_penalty(D, real_samples, fake_samples):
    
      """Calculates the gradient penalty loss for WGAN GP"""
      # Random weight term for interpolation between real and fake samples
      alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
      alpha = alpha.to("cuda:0")
      # Get random interpolation between real and fake samples
      interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
      d_interpolates = D(interpolates)
      fake = torch.ones((real_samples.shape[0], 1))
      fake = fake.to("cuda:0")
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
    
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(train_loader):
            
            if imgs.size(0) != batch_size:
              continue
            
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            real_imgs = 0.5 * real_imgs + 0.5
            
            
            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            
            # Generate a batch of images
            fake_imgs = sample_from_realnvp(generator, imgs.size(0))
            
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
            
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()
            
            if i % opt.n_critic  == 0 and epoch>0:
            
                # Generate a batch of images
                
                fake_imgs = sample_from_realnvp(generator, imgs.size(0))
                
                fake_validity = discriminator(fake_imgs)
                
                g_loss = -torch.mean(fake_validity)
                
                g_loss.backward()
                optimizer_G.step()
    
               
                wandb.log({"d_loss": d_loss})
                wandb.log({"g_loss": g_loss})
            
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
                    )
                    
            if batches_done % sample_interval == 0 and epoch>0:

                # ---------------------
                #  Sampling
                # ---------------------
                
                # Save samples from generator : 
                fake_imgs = postprocess_fake(fake_imgs).permute(0,3,1,2)
                grid = make_grid(fake_imgs[:30], nrow=6).permute(1,2,0)
                
                f = plt.figure(figsize=(10,10))
                plt.imsave("./images3/sample_glow_batch_%d.png" % batches_done, grid.cpu().numpy())
                plt.close(f)
                
                caption_str = "Epoch : " + str(epoch)
                images = wandb.Image(grid.cpu().numpy(), caption=caption_str)
                wandb.log({"Generator:": images})
                
                # Save training batch as refernce : 
                
                real_imgs = postprocess_fake(real_imgs).permute(0,3,1,2)
                grid = make_grid(real_imgs[:30], nrow=6).permute(1,2,0)
                
                f = plt.figure(figsize=(10,10))
                plt.imsave("./images3/gmmsd_example_batch_%d.png" % batches_done, grid.cpu().numpy())
                plt.close(f)
                
                caption_str = "Epoch : " + str(epoch)
                images = wandb.Image(grid.cpu().numpy(), caption=caption_str)
                wandb.log({"GMMSD:": images})
                
                EPOCH = epoch
                PATH = "generator.pt"
                LOSS = g_loss.item()
                
                torch.save({
                            'epoch': EPOCH,
                            'model_state_dict': generator.state_dict(),
                            'optimizer_state_dict': optimizer_G.state_dict(),
                            'loss': LOSS,
                            }, PATH)
                            
                EPOCH = epoch
                PATH = "discriminator.pt"
                LOSS = g_loss.item()
                
                torch.save({
                            'epoch': EPOCH,
                            'model_state_dict': discriminator.state_dict(),
                            'optimizer_state_dict': optimizer_D.state_dict(),
                            'loss': LOSS,
                            }, PATH)
            batches_done = batches_done + n_critic
        
#            if batches_done > 100:
    #############################################################################################################################################        
#                # Create ImageGPT Model :
#                import ipdb; ipdb.set_trace() 
#                samples, p_res = sample_images_from_generator(generator, n_samples=100)
#                samples = samples.detach().cpu()  
#                p_res = p_res.detach().cpu()
#                samples_postproc = 2.0 * (samples - 0.5) # [0,1] --> [-1,1]
#                q_res = run_imagegpt_on_sampled_images(samples_postproc, image_gpt, batch_size)
#                # inception_score_res = measure_inception_score_on_sampled_images(samples_postproc) # Low GPU resources. 
#                samples_postproc = postprocess_fake2(samples, save_image_flag = True)
#                path = save_sampled_images_to_path(samples_postproc, path="/home/dsi/eyalbetzalel/GlowGAN/GlowGan/realnvp_gan/samples_temp_small_train_set")
#                fid_res = measure_fid_on_sampled_images(path_test_dst = path, gpu_num="1")
#                delete_sampled_images_from_path(path)
#                p_res = p_res.tolist()
#                q_res = q_res.tolist()
#                fdiv_res = measure_fdiv_on_sampled_images(p_res, q_res)
#                inception_score = 0.0
    #############################################################################################################################################
#        
#              # JS : 
#              
#                # Samples from realnvp : 
#                    
#                    # Sample from RealNVP:
#                
#                samples_realnvp, p_realnvp = sample_from_realnvp_for_js(generator, batch_size)
#                    
#                    # ImageGPT :
#                samples_realnvp_np = samples_realnvp.detach().cpu()
#                samples_realnvp_postproc = 2.0 * (samples_realnvp_np - 0.5) # [0,1] --> [-1,1] 
#                q_imagegpt = run_imagegpt_on_sampled_images(samples_realnvp_postproc, image_gpt, batch_size)
#                
#                # Samples from ImageGPT : 
#                
#                    # Sample from ImageGPT (train loader) :
#                samples_imagegpt1, _ = next(iter(train_loader)) # [-1, 1]
#                samples_imagegpt1 = samples_imagegpt1.detach().cpu()
#                samples_imagegpt2 = 0.5 * samples_imagegpt1 + 0.5 # [-1, 1] --> [0, 1]
#                
#                    # ImnageGPT : 
#                
#                p_imagegpt = run_imagegpt_on_sampled_images(samples_imagegpt1, image_gpt, batch_size) # samples_imagegpt1 --> [-1, 1]
#    
#                
#                    # RealNVP : 
#                log_q_realnvp = generator.log_prob(samples_imagegpt2.to("cuda:0").permute(0,3,1,2))
#                log_q_realnvp = log_q_realnvp / batch_size
#                q_realnvp = torch.exp(log_q_realnvp)
#                
#                
#                # Cals Jenson-Shannon Divergence :
#                js_div = calc_js_div(p_imagegpt, p_realnvp, q_imagegpt, q_realnvp)
#                # js_div = js_div.numpy()
#                
#                # Calc gradients : 
#                
#                # js : 
#                loss_js = -1.0 * torch.log(torch.tensor(4.0)) + 2.0 * js_div
#                
#                grad_js = calc_gradient(generator, loss_js)
#                # gan :
#                fake_imgs, _ = sample_images_from_generator(generator, batch_size, compute_grad=False)
#                fake_imgs = samples_realnvp.to("cuda:0")
#                fake_imgs = fake_imgs.to("cuda:0")
#                # valid = Variable(Tensor(fake_imgs.size(0), 1).fill_(1.0), requires_grad=False)
#                g_loss = adversarial_loss(discriminator(fake_imgs), valid)
#                # grad_gan = calc_gradient(generator, loss_js)
#                # output = cosine_similarity(grad_js, grad_gan)
#                
#                # Save all results: 
#                g_loss = g_loss.detach().cpu().numpy()
#        
#                # g_loss = 0.0
#                js_div = js_div.detach().cpu().numpy()
#                df_res, res_list = save_all_results_to_file(fdiv_res, g_loss, js_div, inception_score, fid_res, epoch, df_res, res_path="/home/dsi/eyalbetzalel/GlowGAN/GlowGan/realnvp_gan/results/res_small_train_set.csv")
#                epoch, kld_res, tvd_res, chi2p_res, alpha25_res, alpha50_res, alpha75_res, inception_score, fid = res_list
#                wandb.log({"table": df_res})
                
                


################################################################################

# Arguments : 

if __name__ == "__main__":
    
    # WP-GAN : 

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=20000, help="number of epochs of training")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--sample_interval", type=int, default=1000, help="interval betwen image samples")

    # REAL - NVP :
     
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=32)
    parser.add_argument('--base_dim',
                        help='features in residual blocks of first few layers.',
                        type=int,
                        default=32)
    parser.add_argument('--res_blocks',
                        help='number of residual blocks per group.',
                        type=int,
                        default=3)
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

    # ImageGPT 
    
    parser.add_argument('--n_gpu', default=1, type=int)
    parser.add_argument("--tf_device", nargs="+", type=int, default=[1, 3], help="GPU devices for tf")
    parser.add_argument('--imagegpt_artifact', default='/home/dsi/eyalbetzalel/GlowGAN/GlowGan/realnvp_gan/imagegpt/artifacts/')
    
    wandb.init(project="GlowGAN", entity="eyalb")

    opt = parser.parse_args()

    kwargs = vars(opt)

    # with open(os.path.join(args.output_dir, "hparams.json"), "w") as fp:
    #     json.dump(kwargs, fp, sort_keys=True, indent=4)
    
    main(**kwargs)
    print(opt)