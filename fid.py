import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from scipy import linalg
from utils import get_testset
from dcgan import weights_init, Generator, Discriminator
from torchvision.models import inception_v3
import torchvision.models as models
import math
import warnings
import torch.nn.functional as F
###
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from multiprocessing import cpu_count
import torchvision.transforms as TF
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x
    
params = {
    "bsize" : 128,# Batch size during training.
    'imsize' : 128,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
    'nz' : 200,# Size of the Z latent vector (the input to the generator).
    'ngf' : 128,# Size of feature maps in the generator. The depth will be multiples of this.
    'ndf' : 128, # Size of features maps in the discriminator. The depth will be multiples of this.
    'nepochs' : 1,# Number of training epochs.
    'lr' : 0.0002,# Learning rate for optimizers
    'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
    'save_epoch' : 2}# Save step.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

dataloader = get_testset(params)
checkpoint =torch.load('./model_backup/dcgan_final/model_final.pth')
# Create the generator.
netG = Generator(params).to(device)
netG.load_state_dict(checkpoint['generator'])
comparison_size = 256
fixed_noise = torch.randn(comparison_size, params['nz'], 1, 1, device=device)
class FID():
    '''
    Code for FID Calculation taken from TA's piazza post
    '''
    def __init__(self, cache_dir='./Cache', device='cpu', transform_input=True):
        os.environ["TORCH_HOME"] = "./Cache"
        self.device=device
        self.transform_input = transform_input
        self.InceptionV3 = models.inception_v3(pretrained=True, transform_input=False, aux_logits=False).to(device=self.device)
        self.InceptionV3.eval()
    
    def build_maps(self, x):
        # Resize to Fit InceptionV3
        if list(x.shape[-2:]) != [299,299]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x = F.interpolate(x, size=[299,299], mode='bilinear')
        # Transform Input to InceptionV3 Standards
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # Run Through Partial InceptionV3 Model
        with torch.no_grad():
            # N x 3 x 299 x 299
            x = self.InceptionV3.Conv2d_1a_3x3(x)
            # N x 32 x 149 x 149
            x = self.InceptionV3.Conv2d_2a_3x3(x)
            # N x 32 x 147 x 147
            x = self.InceptionV3.Conv2d_2b_3x3(x)
            # N x 64 x 147 x 147
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            # N x 64 x 73 x 73
            x = self.InceptionV3.Conv2d_3b_1x1(x)
            # N x 80 x 73 x 73
            x = self.InceptionV3.Conv2d_4a_3x3(x)
            # N x 192 x 71 x 71
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            # N x 192 x 35 x 35
            x = self.InceptionV3.Mixed_5b(x)
            # N x 256 x 35 x 35
            x = self.InceptionV3.Mixed_5c(x)
            # N x 288 x 35 x 35
            x = self.InceptionV3.Mixed_5d(x)
            # N x 288 x 35 x 35
            x = self.InceptionV3.Mixed_6a(x)
            # N x 768 x 17 x 17
            x = self.InceptionV3.Mixed_6b(x)
            # N x 768 x 17 x 17
            x = self.InceptionV3.Mixed_6c(x)
            # N x 768 x 17 x 17
            x = self.InceptionV3.Mixed_6d(x)
            # N x 768 x 17 x 17
            x = self.InceptionV3.Mixed_6e(x)
            # N x 768 x 17 x 17
            x = self.InceptionV3.Mixed_7a(x)
            # N x 1280 x 8 x 8
            x = self.InceptionV3.Mixed_7b(x)
            # N x 2048 x 8 x 8
            x = self.InceptionV3.Mixed_7c(x)
            # N x 2048 x 8 x 8
            # Adaptive average pooling
            x = F.adaptive_avg_pool2d(x, (1, 1))
            # N x 2048 x 1 x 1
            return x
    
    def compute_fid(self, real_images, generated_images, batch_size=64):
        # Ensure Set Sizes are the Same
        assert(real_images.shape[0] == generated_images.shape[0])
        # Build Random Sampling Orders
        real_images = real_images[np.random.permutation(real_images.shape[0])]
        generated_images = generated_images[np.random.permutation(generated_images.shape[0])]
        # Lists of Maps per Batch
        real_maps = []
        generated_maps = []
        # Build Maps
#        for s in tqdm(range(int(math.ceil(real_images.shape[0]/batch_size))), desc='Evaluation', leave=False):
        for s in range(int(math.ceil(real_images.shape[0]/batch_size))):
            sidx = np.arange(batch_size*s, min(batch_size*(s+1), real_images.shape[0]))
            real_maps.append(self.build_maps(real_images[sidx].to(device=self.device)).detach().to(device='cpu'))
#            real_maps.append(self.build_maps(real_images[sidx]).detach())
            generated_maps.append(self.build_maps(generated_images[sidx].to(device=self.device)).detach().to(device='cpu'))
#            generated_maps.append(self.build_maps(generated_images[sidx]).detach())
        # Concatenate Maps
        real_maps = np.squeeze(torch.cat(real_maps).numpy())
        generated_maps = np.squeeze(torch.cat(generated_maps).numpy())
        # Calculate FID
        # Activation Statistics
        mu_g = np.mean(generated_maps, axis=0)
        mu_x = np.mean(real_maps, axis=0)
        sigma_g = np.cov(generated_maps, rowvar=False)
        sigma_x = np.cov(real_maps, rowvar=False)
        # Sum of Squared Differences
        ssd = np.sum((mu_g - mu_x)**2)
        # Square Root of Product of Covariances
        covmean = linalg.sqrtm(sigma_g.dot(sigma_x), disp=False)[0]
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # Final FID Computation
        return ssd + np.trace(sigma_g + sigma_x - 2*covmean)

fid_obj = FID()

with torch.no_grad():
    rand_sampler = torch.utils.data.RandomSampler(dataloader.dataset, num_samples=comparison_size, replacement=True)
    test_sampler = torch.utils.data.DataLoader(dataloader.dataset, batch_size=comparison_size, sampler=rand_sampler)
    for i,data in enumerate(test_sampler, 0):
        real_data = data[0]
        break
    print(len(real_data))
    fake_data = netG(fixed_noise).detach().cpu()
    fid_val = fid_obj.compute_fid(real_data, fake_data)
    print('fid',fid_val)
