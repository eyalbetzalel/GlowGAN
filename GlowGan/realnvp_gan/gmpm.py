import torch
import data_utils
import numpy as np
from torchvision.utils import save_image
import os
import shutil
from inception_score import inception_score 
from fdiv import *
import pandas as pd
import wandb

# Sample images and measure p (likelihood) on this images by RealNVP:
# TODO : Print timing for each step in the process. 

def sample_images_from_generator(model, n_samples, compute_grad=False):
    print("--- Start : sample 5K images ---")
    samples_all = torch.empty(0, 3, 32, 32).cpu()
    log_prob_all = torch.empty(0).cpu()
    count = 0
    model.eval()
    if not(compute_grad):
        with torch.no_grad():
            while count < n_samples:
                samples = model.sample(32)
                samples, _ = data_utils.logit_transform(samples, reverse=True)
                log_prob = model.log_prob(samples)
                log_prob /= 32
                samples_all = torch.cat((samples_all,samples.detach().cpu()), dim=0)
                log_prob_all = torch.cat((log_prob_all, log_prob.detach().cpu()), dim=0)
                count += 32
    else :    
        while count < n_samples:
            samples = model.sample(32)
            samples, _ = data_utils.logit_transform(samples, reverse=True)
            log_prob = model.log_prob(samples)
            log_prob /= 32
            samples_all = torch.cat((samples_all,samples.detach().cpu()), dim=0)
            log_prob_all = torch.cat((log_prob_all, log_prob.detach().cpu()), dim=0)
            count += 32

    print("--- Finish : sample 5K images ---")
    p = torch.exp(log_prob_all)
    samples_all = samples_all.permute(0,2,3,1)
    return samples_all, p

# Files handeling:

def save_sampled_images_to_path(images, path="./samples_temp"):
    
    if os.path.exists(path):
        shutil.rmtree(path)
    
    os.mkdir(path)

    for i in range(images.shape[0]):

        img = images[0]
        fname = "img_" + str(i) + ".png"
        img_path_str = os.path.join(path,fname)
        # save_image get numbers in [0,1] [C,H,W]:
        img = img.permute(2,0,1)
        img = img.float() 
        img /= 255
        save_image(img, img_path_str)
    
    return path

def delete_sampled_images_from_path(path):
    shutil.rmtree(path)
    return ()

# measure q (likelihood) on this images by ImageGPT

def postprocess_fake2(x, save_image_flag = False):
    
    # function :postprocess_fake : 
    # Input are samples in tensor format from realnvp [-0.05,1.05]
    # Output are sample in numpy format [-1,1] for imagegpt or 
    # if save_image_flag is True [0, 255] for imsave.

    n_bits = 8
    #x = x + 0.05 
    # x = np.clip(x, 0.0, 1.1) # x is now [0, 1.1]
    # x = x / 1.1 # x is now [0, 1]
   
    if save_image_flag:
        x = x * 2 ** n_bits
        x = torch.clamp(x, 0, 255).byte() # x is now int [0, 255]
    else :
        x = 2 * (x - 0.5) # x is now [-1, 1]
    return x

def run_imagegpt_on_sampled_images(images, imagegpt_class, batch_size):

    dataset = torch.utils.data.TensorDataset(images)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)
    nll = []
    for i, imgs in enumerate(data_loader):
        imgs = imgs[0]
        sampled_images_numpy = imgs.detach().cpu().numpy()
        # NOTE: expect channels last
        # clusters are in (-1, 1)
        clustered_sampled_images = imagegpt_class.color_quantize(sampled_images_numpy)
        data_nll = imagegpt_class.eval_model(clustered_sampled_images)
        data_nll = [j for i in data_nll for j in i] # flatten list
        nll.append(data_nll)
    nll = [j for i in nll for j in i] # flatten list
    nll_np = np.asarray(nll)
    q_res = np.exp(-1.0 * nll_np)
    return q_res

# measure FID

def measure_fid_on_sampled_images(path_test_dst = "./temp_folder", path_source_orig= "/home/dsi/eyalbetzalel/image-gpt/save", gpu_num="0"):
    
    # TODO : Add dest folder
    # TODO : Install fid-pytorch on local env
    
    # Call function
    stream = os.popen("CUDA_VISIBLE_DEVICES=" + gpu_num + " python -m pytorch_fid " + path_test_dst + " " + path_source_orig + " --device cuda:0") 
    output = stream.read()
    output_arr_split = output.split()
    
    # read results
    
    i=0
    for str_tmp in output_arr_split:
        if str_tmp == "FID:":
            i+=1
            fid_curr_val = float(output_arr_split[i])
            print("FID curr val : " + str(fid_curr_val))
            break
        i+=1

    # return results 
    return fid_curr_val

# measure IS

def measure_inception_score_on_sampled_images(images):
    # TODO : Check if this is ok on larger GPU. 
    # TODO : imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    images = images.permute(0, 3, 1, 2)
    dataset = torch.utils.data.TensorDataset(images)
    inception_score_res = inception_score(dataset, cuda=True, batch_size=32, resize=True, splits=1)

    return inception_score_res

# measure f-div

def measure_fdiv_on_sampled_images(p,q):
    
    kld_res, _ = kld(p, q)
    tvd_res, _ = tvd(p, q)
    chi2p_res, _ = chi2_pearson(p, q)
    alpha25_res, _ = alphadiv(p, q, alpha=0.25)
    alpha50_res, _ = alphadiv(p, q, alpha=0.5)
    alpha75_res, _ = alphadiv(p, q, alpha=0.75)
    fdiv_res = (kld_res, tvd_res, chi2p_res, alpha25_res, alpha50_res, alpha75_res)
    return fdiv_res

def save_all_results_to_file(fdiv_res, g_loss, js_div, inception_score, fid, epoch, df, res_path): 
    
    kld_res, tvd_res, chi2p_res, alpha25_res, alpha50_res, alpha75_res = fdiv_res
    
    res_list = (epoch, kld_res, tvd_res, chi2p_res, alpha25_res, alpha50_res, alpha75_res, inception_score, fid)
    
    df = df.append({
    'Epoch':epoch,
    'KL':kld_res, 
    'Total Variation Distance':tvd_res, 
    'chi p':chi2p_res, 
    'alpha 0.25':alpha25_res, 
    'alpha 0.5':alpha50_res, 
    'alpha 0.75':alpha75_res,
    'JS' :  js_div,
    'G_Loss': g_loss,
    'FID':fid, 
    'IS':inception_score
    }, ignore_index=True)
    
    df.to_csv(res_path)
    
    return df, res_list
