import json

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from datasets import get_GMMSD, postprocess
from model import Glow

def sample(model):
    with torch.no_grad():

        y = None
        images = postprocess(model(y_onehot=y, temperature=1, reverse=True))
    return images.cpu()
    
device = torch.device("cuda")

output_folder = 'output/'
model_name = 'glow_checkpoint_26256.pt'

with open(output_folder + 'hparams.json') as json_file:  
    hparams = json.load(json_file)
 
image_shape, num_classes, _, test_svhn = get_GMMSD(hparams['augment'], hparams['dataroot'], hparams['download'], hparams['batch_size'])
            

model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
             hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
             hparams['learn_top'], hparams['y_condition'])

model.load_state_dict(torch.load(output_folder + model_name)['model'])
model.set_actnorm_init()

model = model.to(device)

model = model.eval()

images = sample(model)
grid = make_grid(images[:30], nrow=6).permute(1,2,0)

plt.figure(figsize=(10,10))
import ipdb; ipdb.set_trace() 
plt.imsave('./images/sample_glow.png', grid.numpy())
plt.axis('off')