import numpy as np
import tqdm
from ..losses.dsm import anneal_dsm_score_estimation
from ..losses.sliced_sm import anneal_sliced_score_estimation_vr
import torch.nn.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from ..datasets.celeba import CelebA
from ..models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.utils import save_image, make_grid
from PIL import Image
import glob
from natsort import natsorted
import cv2
import torch.nn as nn
from skimage.measure import compare_psnr,compare_ssim

#CUDA_VISIBLE_DEVICES=1 python3.5 separate_ImageNet.py --model ncsn --runner Test_6ch_inpainting_rgbrgb_p --config anneal_bedroom_6ch.yml --doc bedroom_rgbrgb_6ch --test --image_folder output/output_inpainter_6ch_rgbrgb_p/linear/result                #bedroom_rgbrgb_6ch  bedroom_6ch_64

__all__ = ['Test_6ch_inpainting_rgbrgb_p']

def compute_mask(array,rate=0.2):
    zeros_num = int(array.size * rate)
    new_array = np.ones(array.size)
    new_array[:zeros_num] = 0
    np.random.shuffle(new_array)
    re_array = new_array.reshape(array.shape)
    return re_array

def write_Data(result_all,i):
    with open(os.path.join('./output/output_inpainter_6ch_rgbrgb_p/linear/result/',"psnr"+".txt"),"w+") as f:
        #print(len(result_all))
        for i in range(len(result_all)):
            f.writelines('current image {} PSNR : '.format(i) + str(result_all[i][0]) + \
            "    SSIM : " + str(result_all[i][1]))
            f.write('\n')

class Test_6ch_inpainting_rgbrgb_p():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def test(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device) #checkpoint_240000
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet)
        scorenet.load_state_dict(states[0])
        scorenet.eval()
        
        samples = 1
        batch_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        files_list = glob.glob('./test_bedroom_256/*.png')
        files_list = natsorted(files_list)
        length = len(files_list)
        result_all = np.zeros([101,2])
        
        for j,file_path in enumerate(files_list):
            image_gt = cv2.imread(file_path)
            image_inter = cv2.imread('./output/output_inpainter_3ch_rgb/linear/img_{}_result.png'.format(j))

            mask = cv2.imread('./mask_256/linear.png')/255.0

            '''arr = np.ones((256,256))
            mask = compute_mask(arr,rate=0.65)
            mask = mask[:,:,np.newaxis]
            mask = np.concatenate((mask,mask,mask),2)'''
            
            image_inpainter = np.zeros(image_gt.shape).astype(np.float32)
            image_inpainter =  image_gt * mask
            
            mask = (torch.tensor(mask.transpose(2,0,1),dtype=torch.float32).unsqueeze(0)).cuda()
            image_inpainter = (torch.tensor(image_inpainter.transpose(2,0,1),dtype=torch.float32).unsqueeze(0) / 255.0) .cuda()
            image_inter = (torch.tensor(image_inter.transpose(2,0,1),dtype=torch.float32).unsqueeze(0) / 255.0).cuda()
            
            x0 = nn.Parameter(torch.Tensor(samples,6,mask.shape[2],mask.shape[3]).uniform_(0,1)).cuda()
            x01 = x0.clone()
            
            step_lr=0.8*0.00003    # 0.03*0.00006   0.1*0.00003
            #linear curvemask3_256  bedroom_6ch_64 0.8*0.00003
            #block  random  wordmask2_256  bedroom_rgbrgb_6ch checkpoint_240000  0.1*0.00003
            #Noise amounts
            sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,
                               0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
            n_steps_each = 100
            max_psnr = 0
            max_ssim = 0
            list_1=[]
            for idx, sigma in enumerate(sigmas):
                print(idx)
                lambda_recon = 1./sigma**2
                labels = torch.ones(1, device=x0.device) * idx
                labels = labels.long()
                
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                
                print('sigma = {}'.format(sigma))
                for step in range(n_steps_each):
                    
                    noise_x = torch.randn_like(x0) * np.sqrt(step_size * 2)
                    grad_x0 = scorenet(x01, labels).detach()
                    x0 = x0 + step_size * grad_x0 # + noise_x
                    x01 = x0.clone() + noise_x

                    x0 = (x0[:,0:3,...]+x0[:,3:6,...])/2

                    error_x0 = x0 - image_inter
                    x0 = x0 - step_size * lambda_recon * error_x0   
                    x0 = x0 - x0 * mask  + image_inpainter

                    x_rec = x0                   
                    x0 = torch.cat((x0,x0),1)

                    x_rec = x_rec.cpu().detach().numpy().transpose(0,2,3,1)
                    x_rec = np.clip(x_rec,0,1)
                    
                    original_image = np.array(image_gt,dtype = np.float32)
                    for i in range(x_rec.shape[0]):
                        psnr = compare_psnr(x_rec[i,...]*255.0,original_image,data_range=255)
                        ssim = compare_ssim(x_rec[i,...],original_image/255.0,data_range=1,multichannel=True)
                        print("current {} step".format(step),'PSNR :', psnr,'SSIM :', ssim)                    
                    #x_rec = np.transpose(x_rec,[1,2,0]) 
                    if max_psnr < psnr :
                        result_all[j,0] = psnr
                        max_psnr = psnr
                        result_all[length,0] = sum(result_all[:length,0])/length
                    if max_ssim < ssim:
                        result_all[j,1] = ssim
                        max_ssim = ssim
                        result_all[length,1] = sum(result_all[:length,1])/length
                    write_Data(result_all,j)
                    #print("current {} step".format(step),'PSNR :', psnr,'SSIM :', ssim)

            #imageio.mimwrite("./output-va/m.gif",list_1)

            x_save = x_rec
            x_save_R = x_save[:,:,:,2:3]
            x_save_G = x_save[:,:,:,1:2]
            x_save_B = x_save[:,:,:,0:1]
            x_save = np.concatenate((x_save_R,x_save_G,x_save_B),3)
            self.write_images(torch.tensor(x_save.transpose(0,3,1,2)), '_result.png',samples,j)
    def write_images(self, x,name,n=1,z=0):
        x = x.numpy().transpose(0, 2, 3, 1)
        d = x.shape[1]
        panel = np.zeros([1*d,n*d,3],dtype=np.uint8)
        for i in range(1):
            for j in range(n):
                panel[i*d:(i+1)*d,j*d:(j+1)*d,:] = (256*(x[i*n+j])).clip(0,255).astype(np.uint8)[:,:,::-1]

        cv2.imwrite(os.path.join(self.args.image_folder, 'img_{}'.format(z) + name), panel)
