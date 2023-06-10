import torch 
import secrets
import tifffile as tff

import numpy as np
import scipy.signal as sig


kx = np.array([[-1, 0, 1]])
ky = np.array([[-1], [0], [1]])


def read_left(filename):
    rgb = tff.imread(filename)
    rdx, rdy = sig.convolve2d(rgb[:, :, 2], kx, 'same'), sig.convolve2d(rgb[:, :, 2], ky, 'same')
    gdx, gdy = sig.convolve2d(rgb[:, :, 1], kx, 'same'), sig.convolve2d(rgb[:, :, 1], ky, 'same')
    bdx, bdy = sig.convolve2d(rgb[:, :, 0], kx, 'same'), sig.convolve2d(rgb[:, :, 0], ky, 'same')
    dx = np.array([bdx, gdx, rdx])
    dy = np.array([bdy, gdy, rdy])
    rgb = rgb.astype('float32') / 127.5 - 1.0
    dx = dx.astype('float32') / 127.5
    dy = dy.astype('float32') / 127.5
    return rgb, dx, dy

if __name__=="__main__":
    init_size=1024
    patch_size=512
    dataset='/tmp/PairwiseSTEREO/Train-Track2-RGB-Val-GT/Track2-RGB-1'
    gt='/tmp/PairwiseSTEREO/Train-Track2-RGB-Val-GT/Track2-Truth'
    leftimgname=dataset+"/"+"JAX_004_009_007_LEFT_RGB.tif"
    rightimgname=dataset+"/"+"JAX_004_009_007_RIGHT_RGB.tif"
    dispGT=gt+"/"+"JAX_004_009_007_LEFT_DSP.tif"
    # Randomly pick a region from the left image and compute the corresponding offset in the right image 
    x_upl=secrets.randbelow(init_size-patch_size)
    y_upl=secrets.randbelow(init_size-patch_size)
    # Read images
    imL=tff.imread(leftimgname)[y_upl:y_upl+patch_size,x_upl:x_upl+patch_size,:]
    imR=tff.imread(rightimgname)[y_upl:y_upl+patch_size,x_upl:x_upl+patch_size,:]
    imDisp=tff.imread(dispGT)[y_upl:y_upl+patch_size,x_upl:x_upl+patch_size]
    tff.imwrite("./TileL.tif",imL)
    tff.imwrite("./TileR.tif",imR)
    tff.imwrite("./DispL.tif",imDisp)
    # Check dataReader 
    rgb,dx,dy=read_left(leftimgname)
    tff.imwrite("./gradx.tif",dx)
    