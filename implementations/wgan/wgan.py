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
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('-test','--test',type=bool,help='do test',default=False)
parser.add_argument('-train','--train',type=bool,help='do train',default=True)
parser.add_argument('-loadweight','--load-weight',type=bool,help='load weight or not',default=False)
parser.add_argument('-imgdir','--img-dir',help='train image dir',default=r"/home/ali/YOLOV5/runs/detect/f_384_2min/crops_ori")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

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
                                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #GANomaly parameter
                                                ])
                                                )
    data_loader = torch.utils.data.DataLoader(img_data, batch_size=args.batch_size,shuffle=True,drop_last=True)
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
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )
elif custom_data:
    dataloader = load_data(opt)
# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
TRAIN=True
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
    batches_done = 0
    for epoch in range(opt.n_epochs):
        train_loss = 0
        for i, (imgs, _) in enumerate(dataloader):
    
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            optimizer_D.zero_grad()
    
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
    
            # Generate a batch of images
            fake_imgs = generator(z).detach()
            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
    
            loss_D.backward()
            optimizer_D.step()
    
            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)
    
            # Train the generator every n_critic iterations
            if i % opt.n_critic == 0:
    
                # -----------------
                #  Train Generator
                # -----------------
    
                optimizer_G.zero_grad()
    
                # Generate a batch of images
                gen_imgs = generator(z)
                # Adversarial loss
                loss_G = -torch.mean(discriminator(gen_imgs))
    
                loss_G.backward()
                optimizer_G.step()
    
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
                )
    
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:16], "images/%d.png" % batches_done, nrow=4, normalize=True)
            batches_done += 1
            
            loss = loss_D + loss_G
            train_loss += loss.item()*imgs.size(0)
        train_loss = train_loss/len(dataloader)
        torch.save(generator, SAVE_MODEL_G_PATH)
        torch.save(discriminator, SAVE_MODEL_D_PATH)
        print('save model weights complete with loss : %.3f' %(train_loss))