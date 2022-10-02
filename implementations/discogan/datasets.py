import glob
import os
import torch

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root1,root2, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)

        #self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))
        
        self.files1 = sorted(glob.glob(root1 + '/*.*'))
        self.files2 = sorted(glob.glob(root2 + '/*.*'))

    def __getitem__(self, index):

        #img = Image.open(self.files[index % len(self.files)])
        img1 = Image.open(self.files1[index % len(self.files1)])
        img2 = Image.open(self.files2[index % len(self.files2)])
        #w, h = img.size
        w, h = img1.size
        #img_A = img.crop((0, 0, w/2, h))
        #img_B = img.crop((w/2, 0, w, h))
        
        img_A = img1.crop((0, 0, w, h))
        img_B = img2.crop((0, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.files1)
