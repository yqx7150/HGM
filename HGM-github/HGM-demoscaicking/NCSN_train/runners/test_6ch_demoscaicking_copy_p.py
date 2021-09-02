import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import math
from NCSN_train.models.cond_refinenet_dilated import CondRefineNetDilated
from NCSN_train.runners.Patcher6D import *
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr,compare_ssim
import glob
import h5py
from skimage import img_as_float, img_as_ubyte, io
from scipy.misc import imread,imsave
from scipy.linalg import norm,orth
import time
from natsort import natsorted

#CUDA_VISIBLE_DEVICES=1 python3.5 separate_ImageNet.py --model ncsn --runner Test_6ch_demoscaicking_copy_p --config anneal.yml --doc MSR_64_6ch --test --image_folder try_output_copy_p/Kodak_output

#CUDA_VISIBLE_DEVICES=0 python3.5 separate_ImageNet.py --model ncsn --runner Test_6ch_demoscaicking_copy_p --config anneal.yml --doc MSR_64_6ch --test --image_folder try_output_copy_p/McM_output

#import imutils
__all__ = ['Test_6ch_demoscaicking_copy_p']
def show(image):
    plt.figure(1)
    plt.imshow(np.abs(image),cmap='gray')
    plt.show()
def compute_mask(im_shape):
        pattern = 'bayer_rggb'
        # code from https://github.com/VLOGroup/joint-demosaicing-denoising-sem
        if pattern == 'bayer_rggb':
            r_mask = np.zeros(im_shape)
            r_mask[0::2, 0::2] = 1

            g_mask = np.zeros(im_shape)
            g_mask[::2, 1::2] = 1
            g_mask[1::2, ::2] = 1

            b_mask = np.zeros(im_shape)
            b_mask[1::2, 1::2] = 1
            mask = np.zeros(im_shape +(3,))
            mask[:, :, 0] = r_mask
            mask[:, :, 1] = g_mask
            mask[:, :, 2] = b_mask
        return mask
def write_Data(result_all,i):
    #with open(os.path.join('./try_output_copy_p/Kodak_output/',"psnr_Kodak"+".txt"),"w+") as f:
    with open(os.path.join('./try_output_copy_p/McM_output/',"psnr_McM"+".txt"),"w+") as f:
        #print(len(result_all))
        for i in range(len(result_all)):
            f.writelines('current image {} PSNR : '.format(i) + str(result_all[i][0]) + \
            "    SSIM : " + str(result_all[i][1]))
            f.write('\n')

class Test_6ch_demoscaicking_copy_p():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        
    def test(self):

        # Load the score network
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet, device_ids=[0])
        scorenet.load_state_dict(states[0])
        scorenet.eval()

        # get data 
        #files_list = glob.glob('./reference_image/Kodak/*.png')
        files_list = glob.glob('./reference_image/McM/*.tif')
        files_list = natsorted(files_list)
        #files_list = glob.glob('./McM/*.tif')
        #files_list.sort()
        length = len(files_list)
        #result_all = np.zeros([25,3])
        result_all = np.zeros([19,3])

        imsizer,imrizec,patchsize = 500,500,300
        
        Ptchr=Patcher3D(imsize=[imsizer,imrizec,3],patchsize=patchsize,step=int(256), nopartials=True, contatedges=True)     

        nopatches=len(Ptchr.genpatchsizes)
        batchsize = 1
        print("KCT-INFO: there will be in total " + str(nopatches) + " patches.")
        #assert False

        #files_list.sort()
        for j,file_path in enumerate(files_list):
            #test_img = io.imread(file_path, )
            #image_gt = imread('./groundtruth/5.png')
            image_gt = cv2.imread(file_path)
            #img_inter = cv2.imread('./output_MSR/output_3ch_Kodak/img_{}_Rec_.png'.format(j))  #(j+15)
            img_inter = cv2.imread('./output_MSR/output_3ch_McM/img_{}_Rec_.png'.format(j))   #(j+10)
            b, g, r = cv2.split(image_gt)
            image_gt = cv2.merge([r, g, b])
            b1, g1, r1 = cv2.split(img_inter)
            img_inter = cv2.merge([r1, g1, b1])
            
            img_inter = np.transpose(img_inter,[2,0,1])/255.0
            
            ptchs_ = np.array(Ptchr.im2patches(image_gt))
            print('np.array(Ptchr.im2patches(image_gt))',ptchs_.shape)
            mask = compute_mask(image_gt.shape[:2])
            mask = mask.astype(np.int32)

            image_mosaic = np.zeros(image_gt.shape).astype(np.int32)
            image_mosaic = mask * image_gt

            #print(image_mosaic.dtype)
            image_input = np.sum(image_mosaic, axis=2, dtype='uint16')
            #image_mosaic = preprocess(image_input)
            
            image_gt = img_as_ubyte(image_gt)
            image_input = img_as_ubyte(image_mosaic)
            image_gt = image_gt/255.0
            image_input = image_input/255.0

            image_shape = list((batchsize,)+(image_input.shape[2],)+image_input.shape[0:2])

            #x_all = nn.Parameter(torch.Tensor(np.zeros([1,3,imsizer,imrizec])).uniform_(-1,1)).cuda()

            #x0 = nn.Parameter(torch.Tensor(np.zeros([batchsize,3,patchsize,patchsize])).uniform_(-1,1)).cuda()
            
            x0 = nn.Parameter(torch.Tensor(np.zeros([nopatches,6,patchsize,patchsize])).uniform_(-1,1)).cuda()

            x01 = x0
            image_input = np.transpose(image_input,[2,0,1])
            mask = np.transpose(mask,[2,0,1])
            step_lr=0.05*0.00003    #MSR 0.25**0.00003             #0.035*0.00003 

            # Noise amounts
            sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,
                               0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
            n_steps_each = 80
            max_psnr = 0
            max_ssim = 0
            min_hfen = 100
            x0_patches = []
            for idx, sigma in enumerate(sigmas):
                print(idx)
                lambda_recon = 1./sigma**2
                labels = torch.ones(batchsize, device=x0.device) * idx
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                
                print('sigma = {}'.format(sigma))
                for step in range(n_steps_each):
                    x0_patches = []
                    start_in = time.time()
                    for i in range(nopatches):
                        noise1 = torch.rand_like(x0[i:i+1,:,:,:])* np.sqrt(step_size * 2)
                        grad1 = scorenet(x01[i:i+1,:,:,:], labels).detach()
                        x0[i:i+1,:,:,:] = x0[i:i+1,:,:,:] + step_size * grad1
                        x01[i:i+1,:,:,:] = x0[i:i+1,:,:,:] + noise1
                    x_patches = x0.clone().cpu().detach()
                    for i in range(nopatches): 
                        x_patch = x_patches[i,:,:,:]
                        #print(x_patch_gpu.shape,x_patch_gpu.dtype)
                        x_patch=np.array(x_patch.cpu().detach(),dtype = np.float32)
                        x0_patches.append(np.transpose(x_patch,[1,2,0]))
                    #x0_temp = x0.squeeze() 
                    x0_temp = Ptchr.patches2im(x0_patches)
                    x0_temp = np.transpose(x0_temp,[2,0,1])

                    x0_temp = (x0_temp[0:3,...] + x0_temp[3:6,...])/2
                    error_x0 = x0_temp - img_inter
                    x0_temp = x0_temp - step_size * lambda_recon * error_x0

                    x0_temp = x0_temp - (mask * x0_temp - image_input)
                      

                    x0_temp   = np.clip(x0_temp,0,1)
                    x_rec   = x0_temp

                    x_rec = np.transpose(x_rec,[1,2,0]) 
                    psnr = compare_psnr(255*abs(x_rec),255*abs(image_gt),data_range=255)
                    ssim = compare_ssim(abs(x_rec),abs(image_gt),data_range=1,multichannel=True)
                    
                    #x_rec = np.transpose(x_rec,[1,2,0]) 
                    if max_psnr < psnr :
                        result_all[j,0] = psnr
                        max_psnr = psnr
                        result_all[length,0] = sum(result_all[:length,0])/length
                        
                        #imsave('./try_output_copy_p/Kodak_output/'+'img_{}_Rec_'.format(j)+'.png',(255.0*x_rec).astype(np.uint8))
                        imsave('./try_output_copy_p/McM_output/'+'img_{}_Rec_'.format(j)+'.png',(255.0*x_rec).astype(np.uint8))
                    if max_ssim < ssim:
                        result_all[j,1] = ssim
                        max_ssim = ssim
                        result_all[length,1] = sum(result_all[:length,1])/length
                    
                    write_Data(result_all,j)
                    print("current {} step".format(step),'PSNR :', psnr,'SSIM :', ssim)
                    x_mid = np.zeros([nopatches,3,patchsize,patchsize],dtype=np.float32)
                    ptchs = Ptchr.im2patches(x_rec)
                    for i,patch in enumerate(ptchs):
                       x_mid[i,...] =  np.transpose(patch,[2,0,1])
                    x0 = torch.tensor(np.concatenate((x_mid,x_mid),1),dtype=torch.float32).cuda()
                   
    def write_images(self, x,image_save_path):
        x = np.array(x,dtype=np.uint8)
        cv2.imwrite(image_save_path, x)
