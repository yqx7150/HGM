import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.fftpack import fft2, ifft2

#from src.utility.psf2otf import psf2otf

SCALE_INPUT = 0.5

def psf2otf(psf, outSize):
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    padSize = outSize - psfSize
    psf = np.pad(psf, ((0, padSize[0]), (0, padSize[1])), 'constant')
    for i in range(len(psfSize)):
        psf = np.roll(psf, -int(psfSize[i] / 2), i)
    otf = np.fft.fftn(psf)
    nElem = np.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * np.log2(psfSize[k]) * nffts
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= nOps * np.finfo(np.float32).eps:
        otf = np.real(otf)
    return otf


def get_h_input(S):
    SCALE_INPUT = 0.5
    h = np.diff(S, axis=1)
    last_col = S[:, 0, :] - S[:, -1, :]
    last_col = last_col[:, np.newaxis, :]

    h = np.hstack([h, last_col])
    return h


def get_v_input(S):
    v = np.diff(S, axis=0)
    last_row = S[0, ...] - S[-1, ...]
    last_row = last_row[np.newaxis, ...]

    v = np.vstack([v, last_row])
    return v

def from_grad_get_image(S,h,v,beta):  #from_grad_get_image
    #S = cv2.imread('./images/000034.png') / 255.
    #beta = 8.388608e+1 / 2.
    S_in = S

    psf = np.asarray([[-1, 1]])
    out_size = (S.shape[0], S.shape[1])
    otfx = psf2otf(psf, out_size)
    psf = np.asarray([[-1], [1]])
    otfy = psf2otf(psf, out_size)

    Normin1 = fft2(np.squeeze(S), axes=(0, 1))
    Denormin2 = np.square(abs(otfx)) + np.square(abs(otfy))
    Denormin2 = Denormin2[..., np.newaxis]
    Denormin2 = np.repeat(Denormin2, 3, axis=2)
    Denormin = 1 + beta * Denormin2

    h_diff = -np.diff(h, axis=1)
    first_col = h[:, -1, :] - h[:, 0, :]
    first_col = first_col[:, np.newaxis, :]
    h_diff = np.hstack([first_col, h_diff])

    v_diff = -np.diff(v, axis=0)
    first_row = v[-1, ...] - v[0, ...]
    first_row = first_row[np.newaxis, ...]
    v_diff = np.vstack([first_row, v_diff])

    Normin2 = h_diff + v_diff
    Normin2 = beta * np.fft.fft2(Normin2, axes=(0, 1))
    #Normin2 = beta * fft2(Normin2, axes=(0, 1))

    Normin1 = fft2(np.squeeze(S), axes=(0, 1))
    FS = np.divide(np.squeeze(Normin1) + np.squeeze(Normin2),
                   Denormin)
    S = np.real(np.fft.ifft2(FS, axes=(0, 1)))
    #S = np.real(ifft2(FS, axes=(0, 1)))

    S = np.squeeze(S)
    #S = np.clip(S, 0, 1)
    #S = S * 255
    #S = S.astype(np.float32)
    #cv2.imwrite('output.png', S)
    return S

    '''S = cv2.cvtColor(S, cv2.COLOR_BGR2RGB)
    S_in = S_in * 255
    S_in = S_in.astype(np.uint8)
    S_in = cv2.cvtColor(S_in, cv2.COLOR_BGR2RGB)
    plt.imshow(np.hstack((S_in, S)))
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()'''

if __name__ == "__main__":
    S = cv2.imread('./216053.jpg') / 255.
    main(S)
