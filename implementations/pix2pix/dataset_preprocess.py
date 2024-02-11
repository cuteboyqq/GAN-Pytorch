import glob
import random
import os
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cv2
import glob
import argparse
from torch.autograd import Variable
import numpy as np
def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgdir','--img-dir',help='image dir',default=r"C:/datasets/snow_images")
    parser.add_argument('-savedir','--save-dir',help='save image dir',default=r"pix2pix_images")
    parser.add_argument("--dataset_name", type=str, default="snow_scene", help="name of the dataset")
    parser.add_argument("--img_height", type=int, default=720, help="size of image height")
    parser.add_argument("--img_width", type=int, default=1280, help="size of image width")
    parser.add_argument("--skip_frame", type=int, default=2, help="skip frame")
    parser.add_argument("--show_img", type=bool, default=False, help="show image")
    opt = parser.parse_args()
    print(opt)
    return opt

class DataPreprocess:
    def __init__(self, opt):
        self.img_dir = opt.img_dir
        self.img_list = glob.glob(os.path.join(self.img_dir,"**","*.jpg"))
        self.show_img = opt.show_img
        self.dataset_name = opt.dataset_name
        self.img_width = opt.img_width
        self.img_height = opt.img_height
        self.save_pix2piximg = True
        self.save_dir = opt.save_dir
        self.skip_frame = opt.skip_frame

    def generate_pix2pix_dataset(self):
        os.makedirs(os.path.join("images",self.dataset_name), exist_ok=True)
        cuda = torch.cuda.is_available()
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        for i in range(len(self.img_list)):
            if i%self.skip_frame==0:
                print(f"{i} : {self.img_list[i]}")
                img = cv2.imread(self.img_list[i])
                img = cv2.resize(img,(self.img_width,self.img_height),interpolation=cv2.INTER_AREA)
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                avg = np.average(gray)
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                canny = cv2.Canny(gray, int(avg/3.0), int(avg*1.5))

                
                img2 = np.zeros_like(img)
                img2[:,:,0] = canny
                img2[:,:,1] = canny
                img2[:,:,2] = canny

                pix2pix_img = np.zeros((self.img_height,self.img_width*2,3), np.uint8)
                pix2pix_img[:,0:(self.img_width),:] = img[:,:,:]      # [h,w,c]
                pix2pix_img[:,(self.img_width):(self.img_width*2),:] = img2[:,:,:]
                if self.show_img:
                    # cv2.imshow("img",img)
                    # cv2.imshow("canny",img2)
                    cv2.imshow("pix2pix_img",pix2pix_img)
                    cv2.waitKey(100)

                if self.save_pix2piximg:
                    os.makedirs(self.save_dir,exist_ok=True)
                    img_file = str(i) + ".jpg"
                    img_path = os.path.join(self.save_dir,img_file)
                    cv2.imwrite(img_path,pix2pix_img)
          


        return NotImplemented

if __name__=="__main__":
    opt = get_opts()
    dp = DataPreprocess(opt)
    dp.generate_pix2pix_dataset()
   