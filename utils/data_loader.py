# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 15:41:59 2022

@author: User
"""
import torch
import torchvision.transforms as transforms
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