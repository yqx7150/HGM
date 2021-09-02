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
import imageio
from skimage.measure import compare_psnr,compare_ssim

__all__ = ['Test_3ch_inpainting_256']

#CUDA_VISIBLE_DEVICES=1 python3.5 separate_ImageNet.py --model ncsn --runner Test_3ch_inpainting_256 --config anneal_bedroom_3ch.yml --doc bedroom_3ch_64 --test --image_folder ./aaa/3ch


def compute_mask(array,rate=0.2):
    '''按照数组模板生成对应的 0-1 矩阵，默认rate=0.2'''
    zeros_num = int(array.size * rate)#根据0的比率来得到 0的个数
    new_array = np.ones(array.size)#生成与原来模板相同的矩阵，全为1
    new_array[:zeros_num] = 0 #将一部分换为0
    np.random.shuffle(new_array)#将0和1的顺序打乱
    re_array = new_array.reshape(array.shape)#重新定义矩阵的维度，与模板相同
    return re_array

def write_Data(result_all,i):
    with open(os.path.join('./aaa/3ch/',"psnr"+".txt"),"w+") as f:
        #print(len(result_all))
        for i in range(len(result_all)):
            f.writelines('current image {} PSNR : '.format(i) + str(result_all[i][0]) + \
            "    SSIM : " + str(result_all[i][1]))
            f.write('\n')

class Test_3ch_inpainting_256():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def test(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet)
        scorenet.load_state_dict(states[0])
        scorenet.eval()
        
        samples = 1
        batch_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        files_list = glob.glob('./test_bedroom_256/000027.png')
        files_list = natsorted(files_list)
        length = len(files_list)
        result_all = np.zeros([1000,2])
        
        for j,file_path in enumerate(files_list):
            image_gt = cv2.imread(file_path)
            #cv2.imwrite('./output_inpainter_rgb/image_gt.png',image_gt)
            
            mask = cv2.imread('./mask_256/7.png')/255.0

            '''arr = np.ones((256,256))
            mask = compute_mask(arr,rate=0.90)
            mask = mask[:,:,np.newaxis]
            mask = np.concatenate((mask,mask,mask),2)'''

            #print(mask)
            #assert False
            #mask = mask[:,:,np.newaxis]
            image_inpainter = np.zeros(image_gt.shape).astype(np.float32)
            image_inpainter =  mask * image_gt
            
            #cv2.imwrite('./output_inpainter_3ch_rgb/minist/3/image_inpainter/image_inpainter_{}.png'.format(j),image_inpainter)
                   
            img = torch.tensor(image_gt.transpose(2,0,1),dtype=torch.float32).unsqueeze(0) / 255.0
            x_stack = torch.zeros([img.shape[0]*samples,img.shape[1],img.shape[2],img.shape[3]],dtype=torch.float32)

            for i in range(samples):
                x_stack[i*batch_size:(i+1)*batch_size,...] = img
            img = x_stack
            
            mask = torch.tensor(mask.transpose(2,0,1),dtype=torch.float32).unsqueeze(0)
            image_inpainter = torch.tensor(image_inpainter.transpose(2,0,1),dtype=torch.float32).unsqueeze(0) / 255.0
            
            x0 = nn.Parameter(torch.Tensor(samples,3,img.shape[2],img.shape[3]).uniform_(0,1)).cuda()
            x01 = x0.clone()
            
            step_lr=0.1*0.00003     # MSR 0.5**0.00003

            # Noise amounts
            sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,
                               0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
            n_steps_each = 100
            max_psnr = 0
            max_ssim = 0
            list_1 = []
            for idx, sigma in enumerate(sigmas):
                print(idx)
                lambda_recon = 1./sigma**2
                labels = torch.ones(1, device=x0.device) * idx
                labels = labels.long()
                
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                #y = image_inpainter.to(device) + torch.randn_like(image_inpainter.to(device)) * sigma
                
                print('sigma = {}'.format(sigma))
                for step in range(n_steps_each):
                    
                    noise_x = torch.randn_like(x0) * np.sqrt(step_size * 2)
                    grad_x0 = scorenet(x01, labels).detach()
                    x0 = x0 + step_size * grad_x0   #  + noise_x
                    x01 = x0.clone() + noise_x
                    
                    '''import matplotlib
                    import matplotlib.pyplot as plt
                    x_ =  x0.squeeze(0).cpu().detach().numpy().transpose(1,2,0)
                    plt.ion()
                    plt.imshow(x_)
                    plt.show()
                    plt.pause(0.3)
                    plt.clf()
                    plt.close()
                    list_1.append(x_.reshape((1,1,256,256,3)).transpose(0,2,1,3,4).reshape(256,256,3))'''
                    
                    
                    
                    #x0 = y
                    #x0 = x0 - (mask.to(device) * x0  + y)
                    x0 = x0 - x0 * mask.to(device)  + image_inpainter.to(device)
                    #x0 = torch.clamp(x0,0,1) 
                    x_rec = x0.cpu().detach().numpy().transpose(0,2,3,1)

                    original_image = np.array(img,dtype = np.float32).squeeze().transpose(1,2,0)
                    for i in range(x_rec.shape[0]):
                        psnr = compare_psnr(x_rec[i,...]*255.0,original_image*255.0,data_range=255)
                        ssim = compare_ssim(x_rec[i,...],original_image,data_range=1,multichannel=True)
                        print("current {} step".format(step),'PSNR :', psnr,'SSIM :', ssim)
                    #result_all[idx*100+step,0] = psnr
                    #result_all[idx*100+step,1] = ssim                    
                    #x_rec = np.transpose(x_rec,[1,2,0]) 
                    if max_psnr < psnr :
                        result_all[j,0] = psnr
                        max_psnr = psnr
                        result_all[length,0] = sum(result_all[:length,0])/length
                    if max_ssim < ssim:
                        result_all[j,1] = ssim
                        max_ssim = ssim
                        result_all[length,1] = sum(result_all[:length,1])/length
                    #write_Data(result_all,idx*10+step)
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
