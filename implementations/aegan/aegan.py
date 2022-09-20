#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:15:25 2022

@author: ali
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 11:42:50 2022
@author: User
"""

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument('-test','--test',type=bool,help='do test',default=True)
parser.add_argument('-train','--train',type=bool,help='do train',default=False)
parser.add_argument('-loadweight','--load-weight',type=bool,help='load weight or not',default=True)
parser.add_argument('-imgdir','--img-dir',help='train image dir',default=r"/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/f_384_2min/crops_ori")
parser.add_argument('-imgdirtest','--img-dirtest',help='test image dir',default=r"/home/ali/GitHub_Code/YOLO/YOLOV5/runs/detect/f_384_2min/crops_ori")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--test_batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
parser.add_argument("--sample_interval_2", type=int, default=10, help="interval between image sampling")
parser.add_argument("--residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument("--n_critic", type=int, default=5, help="number of training iterations for WGAN discriminator")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
print(cuda)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), res_blocks=9):
        super(GeneratorResNet, self).__init__()
        channels, img_size, _ = img_shape

        # Initial convolution block
        model = [
            nn.Conv2d(channels, 64, 7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        curr_dim = 64
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim *= 2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(curr_dim)]

        # Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim = curr_dim // 2

        # Output layer
        model += [nn.Conv2d(curr_dim, channels, 7, stride=1, padding=3), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        #c = c.view(c.size(0), c.size(1), 1, 1)
        #c = c.repeat(1, 1, x.size(2), x.size(3))
        #x = torch.cat((x, c), 1)
        return self.model(x)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), n_strided=6):
        super(Discriminator, self).__init__()
        channels, img_size, _ = img_shape

        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1), nn.LeakyReLU(0.01)]
            return layers

        layers = discriminator_block(channels, 64)
        curr_dim = 64
        for _ in range(n_strided - 1):
            layers.extend(discriminator_block(curr_dim, curr_dim * 2))
            curr_dim *= 2

        self.model = nn.Sequential(*layers)

        # Output 1: PatchGAN
        self.out1 = nn.Conv2d(curr_dim, 1, 3, padding=1, bias=False)
        # Output 2: Class prediction
        #kernel_size = img_size // 2 ** n_strided
        #self.out2 = nn.Conv2d(curr_dim, c_dim, kernel_size, bias=False)
        ds_size = opt.img_size // 2 ** (n_strided - 2)
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
    def forward(self, img):
        feature_repr = self.model(img)
        #out_adv = self.out1(feature_repr)
        #out_cls = self.out2(feature_repr)
        
        out = feature_repr.view(feature_repr.shape[0], -1)
        validity = self.adv_layer(out)

        return validity,feature_repr
        
        #return out_adv


# Loss function
adversarial_loss = torch.nn.BCELoss()
imgs_loss = torch.nn.MSELoss()
feature_loss = torch.nn.MSELoss()
# Initialize generator and discriminator
img_shape = (opt.channels, opt.img_height, opt.img_width)
generator = GeneratorResNet(img_shape=img_shape, res_blocks=opt.residual_blocks)
discriminator = Discriminator(img_shape=img_shape)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    #criterion_cycle.cuda()
'''
if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch))
else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
'''
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    
TRAIN=opt.train
if TRAIN:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Configure data loader
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
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #GANomaly parameter
                                                ])
                                                )
    data_loader = torch.utils.data.DataLoader(img_data, batch_size=args.batch_size,shuffle=True,drop_last=True)
    print('data_loader length : {}'.format(len(data_loader)))
    return data_loader

def load_data_test(args):
    size = (args.img_size,args.img_size)
    img_data = torchvision.datasets.ImageFolder(args.img_dirtest,
                                                transform=transforms.Compose([
                                                transforms.Resize(size),
                                                #transforms.RandomHorizontalFlip(),
                                                #transforms.Scale(64),
                                                transforms.CenterCrop(size),                                                 
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #GANomaly parameter
                                                ])
                                                )
    data_loader = torch.utils.data.DataLoader(img_data, batch_size=args.test_batch_size,shuffle=False,drop_last=True)
    print('data_loader length : {}'.format(len(data_loader)))
    return data_loader


mnist = False
custom_data = True
if mnist:
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )
elif custom_data:
    dataloader = load_data(opt)
    dataloader_test = load_data_test(opt)
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
if TRAIN:
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
    _lowest_loss = 10000
    for epoch in range(opt.n_epochs):
        train_loss = 0
        for i, (imgs, _) in enumerate(dataloader):
    
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
    
            # Configure input
            real_imgs = Variable(imgs.type(Tensor).to("cuda"))
            
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Generate fake batch of images
            fake_imgs = generator(real_imgs)
            
            optimizer_D.zero_grad()
    
            # Measure discriminator's ability to classify real from generated samples
            real_validity,_=discriminator(real_imgs)
            fake_validity,_=discriminator(fake_imgs.detach())
            real_loss = adversarial_loss(real_validity, valid)
            fake_loss = adversarial_loss(fake_validity, fake)
            d_loss = (real_loss + fake_loss) / 2
            
            # Real images
            #real_validity = discriminator(real_imgs)
            # Fake images
            #fake_validity = discriminator(fake_imgs.detach())
         
            # Adversarial loss
            #loss_D_adv = -torch.mean(real_validity) + torch.mean(fake_validity) 
           
            # Total loss
            #d_loss = loss_D_adv
            if d_loss>=50.0:
                discriminator.apply(weights_init_normal)
            d_loss.backward()
            optimizer_D.step()
            
            
            # Every n_critic times update generator
            if i % opt.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
        
                optimizer_G.zero_grad()
        
                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        
                # Generate a batch of images
                gen_imgs = generator(real_imgs)
        
                # Loss measures generator's ability to fool the discriminator
                #g_loss = adversarial_loss(discriminator(gen_imgs), valid)
                # Discriminator evaluates translated image
                real_imgs_validate,real_feature = discriminator(real_imgs)
                gen_imgs_validate,gen_feature = discriminator(gen_imgs)
                loss_G_adv = feature_loss(gen_feature, real_feature)
                #fake_validity = discriminator(gen_imgs)
                #g_loss1 = adversarial_loss(fake_validity, valid)
                # Adversarial loss
                #loss_G_adv = -torch.mean(fake_validity)
                loss_G_con = imgs_loss(gen_imgs,real_imgs)
                
                g_loss = loss_G_adv*1 + loss_G_con*50
                
                g_loss.backward()
                optimizer_G.step()
            
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
    
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:36], "images/%d.png" % batches_done, nrow=6, normalize=True)
           
            loss = d_loss + g_loss
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
        
'''generate model unable to generate best image'''
TEST=opt.test
if TEST:
    os.makedirs("images_2",exist_ok=True)
    os.makedirs("images_abnormal",exist_ok=True)
    os.makedirs("images_normal",exist_ok=True)
    SAVE_MODEL_G_DIR = "./runs/train/"
    SAVE_MODEL_G_PATH = os.path.join(SAVE_MODEL_G_DIR,"g_net.pt")
    SAVE_MODEL_D_PATH = os.path.join(SAVE_MODEL_G_DIR,"d_net.pt")
    #generator.load_state_dict(torch.load(SAVE_MODEL_G_PATH))
    #discriminator.load_state_dict(torch.load(SAVE_MODEL_D_PATH))
    
    generator = torch.load(SAVE_MODEL_G_PATH)
    discriminator = torch.load(SAVE_MODEL_D_PATH)
    #generator.load_state_dict(torch.load(SAVE_MODEL_G_PATH))
    #discriminator.load_state_dict(torch.load(SAVE_MODEL_D_PATH))
    
    generator = torch.load(SAVE_MODEL_G_PATH)
    discriminator = torch.load(SAVE_MODEL_D_PATH)
    
    #for epoch in range(opt.n_epochs):
    train_loss = 0
    with torch.no_grad():
        for i, (imgs, _) in enumerate(dataloader_test):
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
            # -----------------
            #  Inference Generator
            # -----------------
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
            # Generate a batch of images
            real_imgs = Variable(imgs.type(Tensor).to("cuda"))
            gen_imgs = generator(real_imgs)
            # Loss measures generator's ability to fool the discriminator
            fake_validity,_ = discriminator(gen_imgs)
            fake_loss = adversarial_loss(fake_validity, valid)
            
            print(
                "[Batch %d/%d] [G loss: %f]"
                % (i, len(dataloader), fake_loss.item())
            )
            batches_done = len(dataloader) + i
            #if batches_done % opt.sample_interval_2 == 0:
            if fake_loss>2.0:
                save_image(gen_imgs.data[:1], "images_abnormal/%d.png" % batches_done, nrow=1, normalize=True)
            else:
                save_image(gen_imgs.data[:1], "images_normal/%d.png" % batches_done, nrow=1, normalize=True)