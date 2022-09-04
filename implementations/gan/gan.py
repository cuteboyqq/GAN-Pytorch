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
parser.add_argument('-imgdir','--img-dir',help='train image dir',default=r"C:\factory_data\2022-08-26\f_384_2min\crops_ori")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
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
        #print('G_F_img before view {}'.format(img.shape))
        img = img.view(img.size(0), *img_shape)
        #print('G_F_img {}'.format(img.shape))
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
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        #print('D_F img_flat {}'.format(img_flat.shape))
        validity = self.model(img_flat)

        return validity

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



# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
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

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
SAVE_MODEL_G_DIR = "./runs/train/"
if not os.path.exists(SAVE_MODEL_G_DIR): os.makedirs(SAVE_MODEL_G_DIR)

SAVE_MODEL_G_PATH = os.path.join(SAVE_MODEL_G_DIR,"g_net.pt")
SAVE_MODEL_D_PATH = os.path.join(SAVE_MODEL_G_DIR,"d_net.pt")
_lowest_loss = 10000
for epoch in range(opt.n_epochs):
    train_loss = 0
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        #print('valid {}'.format(valid.shape))
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
        #print('fake {}'.format(fake.shape))
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        #print('real_imgs {}'.format(real_imgs.shape))
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        #print('imgs.shape[0] {}'.format(imgs.shape[0]))
        #print('imgs.shape[1] {}'.format(imgs.shape[1]))
        #print('imgs.shape[2] {}'.format(imgs.shape[2]))
        #print('imgs.shape[3] {}'.format(imgs.shape[3]))
        #print('z {}'.format(z.shape))
        # Generate a batch of images
        gen_imgs = generator(z)
        #print('gen_imgs {}'.format(gen_imgs.shape))
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        
        loss = d_loss + g_loss
        train_loss += loss.item()*imgs.size(0)
        
    train_loss = train_loss/len(dataloader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    
    if train_loss < _lowest_loss:
        _lowest_loss = train_loss
        print('Start save model !')
        torch.save(generator.state_dict(), SAVE_MODEL_G_PATH)
        torch.save(discriminator.state_dict(), SAVE_MODEL_D_PATH)
        print('save model weights complete with loss : %.3f' %(train_loss))
