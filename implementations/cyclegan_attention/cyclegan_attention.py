import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import ImageDataset
from models import Gen, Dis, Attn
from losses import realTargetLoss, fakeTargetLoss, cycleLoss
import datetime
import time
from torchutils import toZeroThreshold, weights_init, Plotter, save_checkpoint
import itertools
from PIL import Image
from torchvision.utils import save_image, make_grid
import argparse
import sys
import os
from torch.autograd import Variable

def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    genA2B.eval()
    genB2A.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = genA2B(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = genB2A(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=10, normalize=True)
    real_B = make_grid(real_B, nrow=10, normalize=True)
    fake_A = make_grid(fake_A, nrow=10, normalize=True)
    fake_B = make_grid(fake_B, nrow=10, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done),nrow=6,normalize=False)


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test','--test',type=bool,help='do test',default=False)
    parser.add_argument('-train','--train',type=bool,help='do train',default=True)
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--n_epochs_infer", type=int, default=2, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="fake_snow_scene", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--valbatch_size", type=int, default=6, help="size of the batches")
    parser.add_argument('--LRgen', type=float, default=1e-4, help='learning rate for gen')
    parser.add_argument('--LRdis', type=float, default=1e-4, help='learning rate for dis')
    parser.add_argument('--LRattn', type=float, default=1e-5, help='learning rate fir attention module')
    parser.add_argument('--dataroot', type=str, default='datasets/', help='root of the images')
    parser.add_argument('--resume', type=str, default='None', help='file to resume')
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=512, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")

    opt = parser.parse_args()
    return opt

def train(opt):
    prev_time = time.time()
    passDisWhole = True
    countAdvLossA = 0.0
    countAdvLossB = 0.0
    countLossCycleA = 0.0
    countLossCycleB = 0.0
    countDisLossA = 0.0
    countDisLossB = 0.0
    lrScheduler = torch.optim.lr_scheduler.MultiStepLR(optAttn, milestones=[30], gamma=0.1, last_epoch=startEpoch -1)
    for epoch in range(startEpoch, nofEpoch):

        # Pass only the transformed fg after epoch 30 as per paper
        if epoch >=30:
            passDisWhole = False
            print('epoch -- >', epoch)

            # reset counters for logging & plotting 
            countAdvLossA = 0.0
            countAdvLossB = 0.0
            countLossCycleA = 0.0
            countLossCycleB = 0.0
            countDisLossA = 0.0
            countDisLossB = 0.0

        for i, batch in enumerate(dataloader):
            if i % 100 == 0:
                print(i)
            realA = Variable(batch['A'].type(Tensor))
            realB = Variable(batch['B'].type(Tensor))

            # optgen zero
            # optattn zero 
            optG.zero_grad()
            optAttn.zero_grad()
            '''
            Train Generator A 
            '''
            # A --> A'' 
            attnMapA = toZeroThreshold(AttnA(realA))
            fgA   = attnMapA * realA
            bgA   = (1 - attnMapA) * realA
            genB  = Variable(genA2B(fgA).type(Tensor))
            fakeB = Variable(((attnMapA * genB) + bgA).type(Tensor))
            # fakeBcopy = fakeB.clone()
            attnMapfakeB = toZeroThreshold(AttnB(fakeB))
            fgfakeB      = attnMapfakeB * fakeB
            bgfakeB      = (1 - attnMapfakeB) * fakeB
            genA_        = Variable((genB2A(fgfakeB)).type(Tensor))
            A_           = Variable(((attnMapfakeB * genA_) + bgfakeB).type(Tensor))
            '''
            Train Generator B
            '''
            # B --> B''
            attnMapB = toZeroThreshold(AttnB(realB))
            fgB = attnMapB * realB
            bgB = (1 - attnMapB) * realB
            genA = Variable((genB2A(fgB)).type(Tensor))
            fakeA = Variable(((attnMapB * genA) + bgB).type(Tensor))
            # fakeAcopy = fakeA.clone()
            attnMapfakeA = toZeroThreshold(AttnA(fakeA))
            fgfakeA = attnMapfakeA * fakeA
            bgfakeA = (1 - attnMapfakeA) * fakeA
            genB_ = Variable((genA2B(fgfakeA)).type(Tensor))
            B_ =    Variable(((attnMapfakeA * genB_) + bgfakeA).type(Tensor))

            # Gen , Attn and cyclic loss
            if passDisWhole:
                AdvLossA = realTargetLoss(disA(fakeA)) + realTargetLoss(disA(A_))
                AdvLossB = realTargetLoss(disB(fakeB)) + realTargetLoss(disB(B_))
            else:
                AdvLossA = realTargetLoss(disA(genA)) + realTargetLoss(disA(genA_))
                AdvLossB = realTargetLoss(disB(genB)) + realTargetLoss(disB(genB_))
            
            LossCycleA = cycleLoss(realA, A_) 
            LossCycleB = cycleLoss(realB, B_) 
            G_totalloss = AdvLossA + AdvLossB + LossCycleA + LossCycleB

            loss_cycle = LossCycleA + LossCycleB
            loss_GAN = AdvLossA + AdvLossB
            loss_G = G_totalloss

            G_totalloss.backward(retain_graph=True)
            optG.step()
            optAttn.step()
            '''
            Train discriminator A & B
            '''
            # Dis Loss and update
            optD.zero_grad()
            if passDisWhole:
                DisLossA = fakeTargetLoss(disA(fakeA)) + fakeTargetLoss(disA(A_)) + 2*realTargetLoss(disA(realA))
                DisLossB = fakeTargetLoss(disB(fakeB)) + fakeTargetLoss(disB(B_)) + 2*realTargetLoss(disA(realB))
            else:
                DisLossA = fakeTargetLoss(disA(genA)) + fakeTargetLoss(disA(genA_)) + 2*realTargetLoss(disA(realA))
                DisLossB = fakeTargetLoss(disB(genB)) + fakeTargetLoss(disB(genB_)) + 2*realTargetLoss(disA(realB))
            # if passDisWhole:
            #     DisLossA = fakeTargetLoss(disA(fakeA)) 
            #     DisLossB = fakeTargetLoss(disB(fakeB)) 
            # else:
            #     DisLossA = fakeTargetLoss(disA(genA)) 
            #     DisLossB = fakeTargetLoss(disB(genB))

            D_totalloss = DisLossA + DisLossB
            D_totalloss.backward(retain_graph=True)
            optD.step()
            
            # update counter
            countAdvLossA += AdvLossA.item()
            countAdvLossB += AdvLossB.item()
            countLossCycleA += LossCycleA.item()
            countLossCycleB += LossCycleB.item()
            countDisLossA += DisLossA.item()
            countDisLossB += DisLossB.item()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()


            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    D_totalloss.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    time_left,
                )
            )
    
            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)


        plotter.log('AdvLossA', countAdvLossA / (i + 1))
        plotter.log('AdvLossB', countAdvLossB / (i + 1))
        plotter.log('LossCycleA', countLossCycleA / (i + 1))
        plotter.log('LossCycleB', countLossCycleB / (i + 1))
        plotter.log('DisLossA', countDisLossA / (i + 1))
        plotter.log('DisLossB', countDisLossB / (i + 1))
        
        if (epoch + 1) % plotEvery == 0:
            plotter.plot('AdvLosses', ['AdvLossA', 'AdvLossB'], filename='AdvLosses.png')
            plotter.plot('CycleLosses', ['LossCycleA', 'LossCycleB'], filename='CycleLosses.png', ymax=1.0)
            plotter.plot('DisLosses', ['DisLossA', 'DisLossB'], filename='DisLosses.png')

        if (epoch + 1) % saveEvery == 0:
            save_checkpoint({
                'epoch' : epoch + 1,
                'optG' : optG.state_dict(),
                'optD' : optD.state_dict(),
                'optAttn' : optAttn.state_dict(),
                'plotter' : plotter,
                'genA2B' : genA2B.state_dict(),
                'genB2A' : genB2A.state_dict(),
                'disA' : disA.state_dict(),
                'disB' : disB.state_dict(),
                'AttnA' : AttnA.state_dict(),
                'AttnB' : AttnB.state_dict()
                }, 
                filename='models/checkpoint_'+str(epoch)+'.pth.tar'
                )

        lrScheduler.step()


if __name__=="__main__":
    opt = get_opt()
    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

    '''
    Tensor use cuda or not
    '''
    cudaAvailable = False
    if torch.cuda.is_available():
        cudaAvailable = True
    Tensor = torch.cuda.FloatTensor if cudaAvailable else torch.Tensor

    '''
    initial gen & dis network
    '''
    # Generators and Discriminators
    genA2B = Gen() 
    genB2A = Gen()
    disA = Dis()
    disB = Dis()
    # Attention Modules
    AttnA = Attn()
    AttnB = Attn()

    '''
    init gen/dis weight
    '''
    genA2B.apply(weights_init)
    genB2A.apply(weights_init)
    disA.apply(weights_init)
    disB.apply(weights_init)
    AttnA.apply(weights_init)
    AttnB.apply(weights_init)
    '''
        gen/dis network use cuda or not
    '''
    if cudaAvailable:
        genA2B.cuda()
        genB2A.cuda()

        disA.cuda()
        disB.cuda()

        AttnA.cuda()
        AttnB.cuda()

    '''
    create optimizer
    '''
    optG = torch.optim.Adam(itertools.chain(genA2B.parameters(), genB2A.parameters()),lr=opt.LRgen)
    optD = torch.optim.Adam(itertools.chain(disA.parameters(), disB.parameters()),lr=opt.LRdis)
    optAttn = torch.optim.Adam(itertools.chain(AttnA.parameters(), AttnB.parameters()),lr=opt.LRattn)

    # attributes to plot and its freq
    attributes =[('AdvLossA', 1),
                ('AdvLossB', 1),
                ('LossCycleA', 1),
                ('LossCycleB', 1),
                ('DisLossA', 1),
                ('DisLossB', 1)
                ]       
    # Custom Plotter module
    plotter = Plotter(attributes)

    dataroot = opt.dataroot
    batchSize = opt.batch_size
    n_cpu = 4
    size = 256
    '''
    create transfomer
    '''
    if opt.train:
        transforms_ = [ transforms.Resize((int(opt.img_height*1.12), int(opt.img_width*1.12)), Image.BICUBIC), 
                        # transforms.RandomCrop(size), 
                        transforms.RandomCrop((opt.img_height, opt.img_width)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    elif opt.test:
        transforms_ = [
                        transforms.RandomCrop((opt.img_height, opt.img_width)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    '''
    create dataloader
    '''
    if opt.train:
        dataloader = DataLoader(ImageDataset(dataroot, transforms_=transforms_, unaligned=True), 
            batch_size=batchSize, shuffle=True, num_workers=n_cpu)
        val_dataloader = DataLoader(ImageDataset(dataroot, transforms_=transforms_, unaligned=True), 
            batch_size=opt.valbatch_size, shuffle=True, num_workers=n_cpu)
    elif opt.test:
        dataloader = DataLoader(ImageDataset(dataroot, transforms_=transforms_, unaligned=True), 
            batch_size=1, shuffle=True, num_workers=n_cpu)
    
    startEpoch = 0
    nofEpoch = 100

    plotEvery = 1
    saveEvery = 2
    '''
    load model
    '''
    if opt.resume != 'None':
        checkpoint = torch.load(opt.resume)
        startEpoch = checkpoint['epoch']
        
        genA2B.load_state_dict(checkpoint['genA2B'])
        genB2A.load_state_dict(checkpoint['genB2A'])
        disA.load_state_dict(checkpoint['disA'])
        disB.load_state_dict(checkpoint['disB'])
        AttnA.load_state_dict(checkpoint['AttnA'])
        AttnB.load_state_dict(checkpoint['AttnB'])
        optG.load_state_dict(checkpoint['optG'])
        optD.load_state_dict(checkpoint['optD'])
        optAttn.load_state_dict(checkpoint['optAttn'])


        plotter = checkpoint['plotter']
        print('resumed from epoch ',startEpoch)
        del(checkpoint)
    '''
    train/ infer model
    '''
    if opt.train:
        train(opt)
    
        
    # To pass the whole image or only the fg to the discriminator
    