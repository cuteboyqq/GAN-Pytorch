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
parser.add_argument('-imgdir','--img-dir',help='train image dir',default=r"/home/ali/YOLOV5/runs/detect/f_384_2min/crops_ori")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--test_batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=400, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
parser.add_argument("--sample_interval_2", type=int, default=10, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))
        

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    
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
            real_imgs = Variable(imgs.type(Tensor))
    
            # -----------------
            #  Train Generator
            # -----------------
    
            optimizer_G.zero_grad()
    
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
    
            # Generate a batch of images
            gen_imgs = generator(z)
    
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
    SAVE_MODEL_G_DIR = "./runs/train/"
    SAVE_MODEL_G_PATH = os.path.join(SAVE_MODEL_G_DIR,"g_net.pt")
    SAVE_MODEL_D_PATH = os.path.join(SAVE_MODEL_G_DIR,"d_net.pt")
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
            gen_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), g_loss.item())
            )
            batches_done = epoch * len(dataloader) + i
            #if batches_done % opt.sample_interval_2 == 0:
            save_image(gen_imgs.data[:1], "images_2/%d.png" % batches_done, nrow=1, normalize=True)