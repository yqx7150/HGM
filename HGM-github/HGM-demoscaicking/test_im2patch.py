import numpy as np
from NCSN_train.runners.Patcher import *
from scipy.misc import imread,imsave
import matplotlib.pyplot as plt


[imsizer,imrizec] = [220,132]
img = imread('image_gt.png')[:,:,0]
Ptchr=Patcher(imsize=[imsizer,imrizec],patchsize=64,step=int(44), nopartials=True, contatedges=True)   

nopatches=len(Ptchr.genpatchsizes)
print("KCT-INFO: there will be in total " + str(nopatches) + " patches.")

ptchs = Ptchr.im2patches(img)
#print()
for patch in ptchs:
    print(patch.shape)
    plt.figure(1)
    plt.imshow(patch)
    plt.show()
img2 = Ptchr.patches2im(ptchs)
print(img.shape)
plt.figure(1)
plt.imshow(img)
plt.show()
err = abs(img - img2)
print(np.max(err))
