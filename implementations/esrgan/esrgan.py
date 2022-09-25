"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 esrgan.py'
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images/training", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument('-test','--test',type=bool,help='do test',default=False)
    parser.add_argument('-train','--train',type=bool,help='do train',default=True)
    parser.add_argument('-loadweight','--load-weight',type=bool,help='load weight or not',default=False)
    parser.add_argument('-imgdir','--img-dir',help='train image dir',default=r"C:\factory_data\2022-08-26\f_384_2min\crops_ori\line")
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="line", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
    parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
    parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
    parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
    parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
    opt = parser.parse_args()
    print(opt)
    return opt



import torchvision
def load_data(args):
    size = (args.img_size,args.img_size)
    img_data = torchvision.datasets.ImageFolder(args.img_dir,
                                                transform=transforms.Compose([
                                                transforms.Resize(size),
                                                #transforms.RandomHorizontalFlip(),
                                                #transforms.Scale(64),
                                                transforms.CenterCrop(size),                                                 
                                                transforms.ToTensor(),
                                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #GANomaly parameter
                                                ])
                                                )
    data_loader = torch.utils.data.DataLoader(img_data, batch_size=args.batch_size,shuffle=True,drop_last=True)
    print('data_loader length : {}'.format(len(data_loader)))
    return data_loader

def load_data_test(args):
    size = (args.img_size,args.img_size)
    img_data = torchvision.datasets.ImageFolder(args.img_dir,
                                                transform=transforms.Compose([
                                                transforms.Resize(size),
                                                #transforms.RandomHorizontalFlip(),
                                                #transforms.Scale(64),
                                                transforms.CenterCrop(size),                                                 
                                                transforms.ToTensor(),
                                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #GANomaly parameter
                                                ])
                                                )
    data_loader = torch.utils.data.DataLoader(img_data, batch_size=args.test_batch_size,shuffle=False,drop_last=True)
    print('data_loader length : {}'.format(len(data_loader)))
    return data_loader


# ----------
#  Training
# ----------
def train(data_loader,opt):
    
    
    # Initialize generator and discriminator
    generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
    discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)
    #feature_extractor = FeatureExtractor().to(device)

    # Set feature extractor to inference mode
    #feature_extractor.eval()

    # Losses
    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_content = torch.nn.L1Loss().to(device)
    criterion_pixel = torch.nn.L1Loss().to(device)

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
        discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    
    SAVE_MODEL_G_DIR = "./runs/train/"
    if not os.path.exists(SAVE_MODEL_G_DIR): os.makedirs(SAVE_MODEL_G_DIR)
    
    SAVE_MODEL_G_DIR = "./runs/train/"
    SAVE_MODEL_G_PATH = os.path.join(SAVE_MODEL_G_DIR,"g_net.pt")
    SAVE_MODEL_D_PATH = os.path.join(SAVE_MODEL_G_DIR,"d_net.pt")
    #generator.load_state_dict(torch.load(SAVE_MODEL_G_PATH))
    #discriminator.load_state_dict(torch.load(SAVE_MODEL_D_PATH))
    if opt.load_weight:
        if os.path.exists(SAVE_MODEL_G_PATH):
            generator = torch.load(SAVE_MODEL_G_PATH)
            discriminator = torch.load(SAVE_MODEL_D_PATH)
            print('load weight G: {}'.format(SAVE_MODEL_G_PATH))
            print('load weight D: {}'.format(SAVE_MODEL_D_PATH))
    
    _lowest_loss = 100000
    for epoch in range(opt.epoch, opt.n_epochs):
        train_loss = 0
        for i, imgs in enumerate(dataloader):
    
            batches_done = epoch * len(dataloader) + i
    
            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))
    
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
    
            # ------------------
            #  Train Generators
            # ------------------
    
            optimizer_G.zero_grad()
    
            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)
    
            # Measure pixel-wise loss against ground truth
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)
    
            if batches_done < opt.warmup_batches:
                # Warm-up (pixel-wise loss only)
                loss_pixel.backward()
                optimizer_G.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), loss_pixel.item())
                )
                continue
    
            # Extract validity predictions from discriminator
            pred_real = discriminator(imgs_hr).detach()
            pred_fake = discriminator(gen_hr)
    
            # Adversarial loss (relativistic average GAN)
            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
    
            # Content loss
            #gen_features = feature_extractor(gen_hr)
            #real_features = feature_extractor(imgs_hr).detach()
            #loss_content = criterion_content(gen_features, real_features)
    
            # Total generator loss
            #loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel
            loss_G = opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            optimizer_D.zero_grad()
    
            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_hr.detach())
    
            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)
    
            # Total loss
            loss_D = (loss_real + loss_fake) / 2
    
            loss_D.backward()
            optimizer_D.step()
    
            # --------------
            #  Log Progress
            # --------------
    
            print(
                #"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, pixel: %f]"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    #loss_content.item(),
                    loss_GAN.item(),
                    loss_pixel.item(),
                )
            )
    
            if batches_done % opt.sample_interval == 0:
                # Save image grid with upsampled inputs and ESRGAN outputs
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                img_grid = denormalize(torch.cat((imgs_lr, gen_hr), -1))
                save_image(img_grid, "images/training/%d.png" % batches_done, nrow=1, normalize=False)
    
            if batches_done % opt.checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
                torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" %epoch)
            
            loss = loss_D + loss_G
            train_loss += loss.item()*imgs.size(0)
        train_loss = train_loss/len(dataloader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        print('_lowest_loss = {}'.format(_lowest_loss))
        if train_loss < _lowest_loss:
            _lowest_loss = train_loss
             
        print('Start save model !')
        #torch.save(generator.state_dict(), SAVE_MODEL_G_PATH)
        #torch.save(discriminator.state_dict(), SAVE_MODEL_D_PATH)
        torch.save(generator, SAVE_MODEL_G_PATH)
        torch.save(discriminator, SAVE_MODEL_D_PATH)
        print('save model weights complete with loss : %.3f' %(train_loss))    
            

if __name__=="__main__":
    opt = get_opt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hr_shape = (opt.hr_height, opt.hr_width)

    
    
    
    CelebAa= False
    custom_data = True
    
    if CelebAa:
        dataloader = DataLoader(
            ImageDataset("../../data/%s" % opt.dataset_name, hr_shape=hr_shape),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
        )
    elif custom_data:
        dataloader = DataLoader(
            ImageDataset(opt.img_dir, hr_shape=hr_shape),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
        )
        #dataloader = load_data(opt)
        #dataloader_test = load_data_test(opt)
    TRAIN = opt.train
    if TRAIN:
        train(dataloader,opt)