import torch
import data_utils
import numpy as np
from torchvision.utils import save_image
import os
import shutil
from inception_score import inception_score 
from fdiv import *

# Sample images and measure p (likelihood) on this images by RealNVP:
# TODO : Print timing for each step in the process. 

def sample_images_from_generator(model, n_samples):
    print("--- Start : sample 5K images ---")
    model.eval()
    with torch.no_grad():
        samples = model.sample(n_samples)
        samples, _ = data_utils.logit_transform(samples, reverse=True)
        samples = samples.permute(0,2,3,1)
        log_prob = model.log_prob(samples)
    print("--- Finish : sample 5K images ---")
    p = torch.exp(log_prob)
    return samples, p

# Files handeling:

def save_sampled_images_to_path(images, path="./samples_temp"):
    
    if os.path.exists(path):
        shutil.rmtree(path)
    
    os.mkdir(path)

    for i in range(images.shape[0]):

        img = images[0]
        fname = "img_" + str(i) + ".png"
        img_path_str = os.path.join(path,fname)
        save_image(img, img_path_str)
    
    return path

def delete_sampled_images_from_path(path):
    shutil.rmtree(path)
    return ()

# measure q (likelihood) on this images by ImageGPT

def postprocess_fake(x, save_image_flag = False):
    
    # function :postprocess_fake : 
    # Input are samples in tensor format from realnvp [-0.05,1.05]
    # Output are sample in numpy format [-1,1] for imagegpt or 
    # if save_image_flag is True [0, 255] for imsave.

    n_bits = 8
    x = x + 0.05 
    x = np.clip(x, 0.0, 1.1) # x is now [0, 1.1]
    x = x / 1.1 # x is now [0, 1]
   
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
    for i, (imgs, _) in enumerate(data_loader):
        sampled_images_numpy = imgs.permute(0, 2, 3, 1).detach().cpu().numpy()
        # NOTE: expect channels last
        # clusters are in (-1, 1)
        sampled_images_numpy = postprocess_fake(sampled_images_numpy)
        clustered_sampled_images = imagegpt_class.color_quantize(sampled_images_numpy)
        data_nll = imagegpt_class.eval_model(clustered_sampled_images)
        nll.append(data_nll)
    # TODO stack results
    # TODO from nll to probability  
    return q

# measure FID

def measure_fid_on_sampled_images(path_test_dst = "./temp_folder", path_source_orig= "./source_folder", gpu_num="0"):
    
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
    # TODO : In the original code dataset is created from torchvision not from torch.utils.dataset
    # check if this is working right
    # TODO : imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]

    dataset = torch.utils.data.TensorDataset(images)
    inception_score_res = inception_score(dataset, cuda=True, batch_size=32, resize=True, splits=1)

    return inception_score_res

# measure f-div

def measure_fdiv_on_sampled_images(p,q):
    
    kld, _ = kld(p, q)
    tvd, _ = tvd(p, q)
    chi2p, _ = chi2_pearson(p, q)
    alpha25, _ = alphadiv(p, q, alpha=0.25)
    alpha50, _ = alphadiv(p, q, alpha=0.5)
    alpha75, _ = alphadiv(p, q, alpha=0.75)
    fdiv_res = (kld, tvd, chi2p, alpha25, alpha50, alpha75)
    return fdiv_res

def save_all_results_to_file(fdiv_res, inception_score, fid, epoch, path="./"):
    kld, tvd, chi2p, alpha25, alpha50, alpha75 = fdiv_res
    # TODO 
    return () 
