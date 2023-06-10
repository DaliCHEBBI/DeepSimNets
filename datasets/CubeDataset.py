from copyreg import dispatch_table
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
#import kornia.augmentation as AugEngine
import os
import timeit 
import tifffile as tff 
import secrets
import re
import warnings
"""import rasterio 
from rasterio.windows import Window
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
import cv2"""
from PIL import Image
import random
import torchvision.transforms as T
import torch.nn.functional as F

def Write2File(aListOfAbsNames,NameFile):
    with open(NameFile,'w+') as fl:
        fl.writelines([el + '\n' for el in aListOfAbsNames])
SCALES={
    'ENSHEDE':256.0,
    'EUROSDR':256.0,
    'VAHINGEN':192,
    'TOULOUSE_METRO':256.0,
    }

"""GeometricRandomBank=T.compose([
 T.RandomAffine(0,shear=5),
 T.RandomVerticalFlip(p=0.5)]
    )
RadiometricRandomBank=T.compose([
 T.RandomEqualize(),
 T.RandomAutocontrast()]
)"""
def ReadFileD(filename):
    dims=[]
    file=open(filename,'r')
    for f in file.readlines():
        dims.append(int(f.strip()))
    return dims

def read_list_from_file(f_):
    with open(f_, 'r') as h:
        filelist = h.read().splitlines()
        filelist = [x.split()[0] for x in filelist]
    return filelist

def read_MinMax_from_file(f_):
    with open(f_, 'r') as h:
        filelist = h.read().splitlines()
        filelist = [[float(x.split()[1]),float(x.split()[2])] for x in filelist]
    return filelist

def LoadImages(folder):
    LeftTileNames=[]
    RightTileNames=[]
    DispTileNames=[]
    files = [] 
    if not isinstance(folder,list):
        folder=[folder]
    for (dirpath, dirnames, filenames) in os.walk(folder.pop()):
        folder.extend(dirnames)
        files.extend(map(lambda n: os.path.join(*n), zip([dirpath] * len(filenames), filenames)))
    # classify file names into 3 classes Left Rigth and disparities
    LeftTileNames=[el for el in files if re.search("colored_0",el)]
    RightTileNames=[el for el in files if re.search("colored_1",el)]
    DispTileNames=[el for el in files if re.search("disp_occ",el)]
    return LeftTileNames,RightTileNames,DispTileNames
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
class CubesDataset(Dataset):
    def __init__(self,patch_size,ImageDataset,SimilModel,BUFFER):
        self.PATCH_SIZE=patch_size
        self.ImageDataset=ImageDataset
        self.SimilarityModel=SimilModel
        self.BUFF=BUFFER
    def __len__(self):
        return len(self.ImageDataset.LeftTilN)
    def __getitem__(self, index):
        x0,x1,dispnoc,masqnoc=self.ImageDataset.__getitem__(index)
        # Use the Model to get features and compute similarity
        FeatsL=self.SimilarityModel.feature(x0)
        FeatsR=self.SimilarityModel.feature(x1)
        H,W=FeatsL.size()[-2],FeatsL.size()[-1]
        xx=secrets.randbelow(W-self.PATCH_SIZE)
        yy=secrets.randbelow(H-self.PATCH_SIZE)
        Disp=dispnoc[yy:yy+self.PATCH_SIZE, xx:xx+self.PATCH_SIZE]*(-1.0) # inverse because of MM convention
        Masq=masqnoc[yy:yy+self.PATCH_SIZE, xx:xx+self.PATCH_SIZE]
        NappeSupLoc=torch.round(Disp+self.BUFF).int()
        NappeInfLoc=torch.round(Disp-self.BUFF).int()
        # Initialize Cube
        lower_lim=torch.min(NappeInfLoc).item()
        upper_lim=torch.max(NappeSupLoc).item()
        HGT= upper_lim-lower_lim
        CUBE=torch.ones((HGT,self.PATCH_SIZE,self.PATCH_SIZE))
        # IT IS POSSIBLE TO DISCRETIZE THE COST VOLUME TO SUB PIXEL LEVES ==> ???
        X_field = torch.arange(xx,xx+self.PATCH_SIZE,dtype=torch.int64).expand(NappeSupLoc.size())
        #print("XFIELD  ",X_field.shape)
        # Compute the Disparity Field
        Masq_Field = torch.zeros(CUBE.size(),dtype=torch.int16)
        for yy in range(self.PATCH_SIZE):
            for xx in range(self.PATCH_SIZE):
                mm_=torch.zeros(HGT)
                mm_[NappeInfLoc[yy,xx]-lower_lim:NappeSupLoc[yy,xx]-lower_lim]=\
                        torch.ones((NappeSupLoc[yy,xx]-NappeInfLoc[yy,xx]))
                # Add and encoding of 2 
                mm_[round(Disp[yy,xx].item()-lower_lim)]=2
                Masq_Field[:,yy,xx]=mm_
                # add disparity encoding of disparity (Disparity= 2 , Nappes =1 , Other= 0 (obliged shape constraint))
        # Gather  Rights Features to compute the overal cost volume at once
        Masq_Field=Masq_Field*Masq.unsqueeze(0).repeat_interleave(HGT,0)
        for d in range(HGT):
            X_D=X_field-(d+lower_lim) # Shape (PATCHSIZE, PATCHSIZE)
            Masq_X_D=(X_D>=0) * (X_D<W)
            X_D=X_D*Masq_X_D.int()
            indexes=X_D.unsqueeze(0).repeat_interleave(FeatsR.size()[1],0)
            FeatsRLoc=torch.gather(FeatsR.squeeze(),-1,indexes)
            #print("Right slice shape ",FeatsRLoc.shape,FeatsL[0,:,yy-RADIUS:yy+RADIUS, xx-RADIUS:xx+RADIUS].shape)
            FeatsLR=torch.cat((FeatsL[0,:,yy:yy+self.PATCH_SIZE, xx:xx+self.PATCH_SIZE],FeatsRLoc),0).unsqueeze(0)
            #print(" SHAPE OF CONCATENATION ",FeatsLR.shape)
            CUBE[d,:,:]=self.SimilarityModel.decisionNet(FeatsLR).sigmoid()*Masq_X_D.float()
        NewCube=CUBE.detach().clone()
        NewCube.requires_grad_(True)
        return NewCube, Masq_Field, (lower_lim,upper_lim)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
class DFCAerialDataset(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=896
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        self.LeftTilN=read_list_from_file(self.images+"/l_1_left_train.txt")
        self.RightTilN=read_list_from_file(self.images+"/l_1_right_train.txt")
        self.DispTilN=read_list_from_file(self.images+"/l_1_disp_train.txt")
        self.MasqTilN=read_list_from_file(self.images+"/l_1_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_2_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_2_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_2_disp_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_2_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_31_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_31_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_31_disp_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_31_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_32_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_32_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_32_disp_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_32_masq_train.txt")
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        #idx=idx%len(self.LeftTilN) 
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        imDisp=tff.imread(dispGT)*(-1.0)
        imMasq=tff.imread(masqGT)/255.0
        NotOCC=imDisp*imMasq
        NotOCC=NotOCC[NotOCC!=0.0]
        D_mu=int(np.mean(NotOCC))
        #print("Moyenne de disparite ++++>>>>>>>>>>>>>>>>>",D_mu)
        # Randomly pick a region from the left image and compute the corresponding offset in the right image 
        x_upl=D_mu+secrets.randbelow(self.init_size-self.patch_size-D_mu)
        y_upl=secrets.randbelow(self.init_size-self.patch_size)
        imMasq=imMasq[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images
        grayimL=tff.imread(leftimgname)
        grayimR=tff.imread(rightimgname)
        # Color images to Gray 
        # Save into tensors
        x0=torch.from_numpy(grayimL[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]).unsqueeze(0).float()
        x1=torch.from_numpy(grayimR[y_upl:y_upl+self.patch_size,:]).unsqueeze(0).float() # take whole Right image
        # NORMALIZE TILES LOCALLY 
        x0=(x0-x0.mean()).div(x0.std()+1e-12)
        x1=(x1-x1.mean()).div(x1.std()+1e-12)
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqnoc=torch.from_numpy(imMasq).to(torch.int16)
        return x0,x1,dispnoc,masqnoc,x_upl
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
class StereoValAerialDataset(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=768
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        self.LeftTilN=read_list_from_file(self.images+"/all_left_test.txt")
        self.RightTilN=read_list_from_file(self.images+"/all_right_test.txt")
        self.DispTilN=read_list_from_file(self.images+"/all_disp_test.txt")
        self.MasqTilN=read_list_from_file(self.images+"/all_masq_test.txt")
        """self.LeftTilN=[os.path.join(self.images,el) for el in self.LeftTilN]
        self.RightTilN=[os.path.join(self.images,el) for el in self.RightTilN]
        self.DispTilN=[os.path.join(self.images,el) for el in self.DispTilN]"""
        self.LeftTilN=self.LeftTilN[0:1000]
        self.RightTilN=self.RightTilN[0:1000]
        self.DispTilN=self.DispTilN[0:1000]
        self.MasqTilN=self.MasqTilN[0:1000]
        #print(self.LeftTilN)
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        imDisp=tff.imread(dispGT)*(-1.0)
        imMasq=tff.imread(masqGT)/255.0
        NotOCC=imDisp*imMasq
        NotOCC=NotOCC[NotOCC!=0.0]
        D_mu=int(np.mean(NotOCC))
        #print("Moyenne de disparite ",D_mu)
        # Randomly pick a region from the left image and compute the corresponding offset in the right image 
        x_upl=D_mu+secrets.randbelow(self.init_size-self.patch_size-D_mu)
        y_upl=secrets.randbelow(self.init_size-self.patch_size)
        imMasq=imMasq[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images
        grayimL=tff.imread(leftimgname)
        grayimR=tff.imread(rightimgname)
        # Color images to Gray 
        # Save into tensors
        x0=torch.from_numpy(grayimL[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]).unsqueeze(0).float()
        x1=torch.from_numpy(grayimR[y_upl:y_upl+self.patch_size,:]).unsqueeze(0).float() # take whole Right image
        # NORMALIZE TILES LOCALLY 
        x0=(x0-x0.mean()).div(x0.std()+1e-12)
        x1=(x1-x1.mean()).div(x1.std()+1e-12)
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqnoc=torch.from_numpy(imMasq).to(torch.int16)
        return x0,x1,dispnoc,masqnoc,x_upl
    
############################################################################################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
class StereoTrAerialDatasetSparse(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=768
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        self.LeftTilN=read_list_from_file(self.images+"/l_1_left_train.txt")
        self.RightTilN=read_list_from_file(self.images+"/l_1_right_train.txt")
        self.DispTilN=read_list_from_file(self.images+"/l_1_disp_sparse_train.txt")
        self.MasqTilN=read_list_from_file(self.images+"/l_1_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_2_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_2_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_2_disp_sparse_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_2_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_31_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_31_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_31_disp_sparse_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_31_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_32_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_32_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_32_disp_sparse_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_32_masq_train.txt")
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        #idx=idx%len(self.LeftTilN) 
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        imDisp=tff.imread(dispGT)*(-1.0)
        imMasq=tff.imread(masqGT)/255.0
        NotOCC=imDisp*imMasq
        NotOCC=NotOCC[NotOCC!=0.0]
        D_mu=int(np.mean(NotOCC))
        #print("Moyenne de disparite ++++>>>>>>>>>>>>>>>>>",D_mu)
        # Randomly pick a region from the left image and compute the corresponding offset in the right image 
        x_upl=D_mu+secrets.randbelow(self.init_size-self.patch_size-D_mu)
        y_upl=secrets.randbelow(self.init_size-self.patch_size)
        imMasq=imMasq[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images
        grayimL=tff.imread(leftimgname)
        grayimR=tff.imread(rightimgname)
        # Color images to Gray 
        # Save into tensors
        x0=torch.from_numpy(grayimL[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]).unsqueeze(0).float()
        x1=torch.from_numpy(grayimR[y_upl:y_upl+self.patch_size,:]).unsqueeze(0).float() # take whole Right image
        # NORMALIZE TILES LOCALLY 
        x0=(x0-x0.mean()).div(x0.std()+1e-12)
        x1=(x1-x1.mean()).div(x1.std()+1e-12)
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqnoc=torch.from_numpy(imMasq).to(torch.int16)
        return x0,x1,dispnoc,masqnoc,x_upl
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
class StereoValAerialDatasetSparse(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=768
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        self.LeftTilN=read_list_from_file(self.images+"/all_left_test.txt")
        self.RightTilN=read_list_from_file(self.images+"/all_right_test.txt")
        self.DispTilN=read_list_from_file(self.images+"/all_disp_sparse_test.txt")
        self.MasqTilN=read_list_from_file(self.images+"/all_masq_test.txt")
        """self.LeftTilN=[os.path.join(self.images,el) for el in self.LeftTilN]
        self.RightTilN=[os.path.join(self.images,el) for el in self.RightTilN]
        self.DispTilN=[os.path.join(self.images,el) for el in self.DispTilN]"""
        self.LeftTilN=self.LeftTilN[0:1000]
        self.RightTilN=self.RightTilN[0:1000]
        self.DispTilN=self.DispTilN[0:1000]
        self.MasqTilN=self.MasqTilN[0:1000]
        #print(self.LeftTilN)
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        imDisp=tff.imread(dispGT)*(-1.0)
        imMasq=tff.imread(masqGT)/255.0
        NotOCC=imDisp*imMasq
        NotOCC=NotOCC[NotOCC!=0.0]
        D_mu=int(np.mean(NotOCC))
        #print("Moyenne de disparite ",D_mu)
        # Randomly pick a region from the left image and compute the corresponding offset in the right image 
        x_upl=D_mu+secrets.randbelow(self.init_size-self.patch_size-D_mu)
        y_upl=secrets.randbelow(self.init_size-self.patch_size)
        imMasq=imMasq[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images
        grayimL=tff.imread(leftimgname)
        grayimR=tff.imread(rightimgname)
        # Color images to Gray 
        # Save into tensors
        x0=torch.from_numpy(grayimL[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]).unsqueeze(0).float()
        x1=torch.from_numpy(grayimR[y_upl:y_upl+self.patch_size,:]).unsqueeze(0).float() # take whole Right image
        # NORMALIZE TILES LOCALLY 
        x0=(x0-x0.mean()).div(x0.std()+1e-12)
        x1=(x1-x1.mean()).div(x1.std()+1e-12)
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqnoc=torch.from_numpy(imMasq).to(torch.int16)
        return x0,x1,dispnoc,masqnoc,x_upl
    
############################################################################################


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
class StereoTrAerialDatasetSparseN(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=768
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        self.LeftTilN=read_list_from_file(self.images+"/l_1_left_train.txt")
        self.RightTilN=read_list_from_file(self.images+"/l_1_right_train.txt")
        self.DispTilN=read_list_from_file(self.images+"/l_1_disp_sparse_train.txt")
        self.MasqTilN=read_list_from_file(self.images+"/l_1_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_2_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_2_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_2_disp_sparse_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_2_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_31_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_31_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_31_disp_sparse_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_31_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_32_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_32_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_32_disp_sparse_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_32_masq_train.txt")
    def __len__(self):
        return len(self.LeftTilN)*2
    def __getitem__(self, idx):
        idx=idx%len(self.LeftTilN) 
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        imDisp=tff.imread(dispGT)*(-1.0)
        imMasq=tff.imread(masqGT)/255.0
        NotOCC=imDisp*imMasq
        NotOCC=NotOCC[NotOCC!=0.0]
        D_mu=int(np.mean(NotOCC))
        #print("Moyenne de disparite ++++>>>>>>>>>>>>>>>>>",D_mu)
        # Randomly pick a region from the left image and compute the corresponding offset in the right image 
        x_upl=D_mu+secrets.randbelow(self.init_size-self.patch_size-D_mu)
        y_upl=secrets.randbelow(self.init_size-self.patch_size)
        imMasq=imMasq[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images
        grayimL=tff.imread(leftimgname)/255.0
        grayimR=tff.imread(rightimgname)/255.0
        # Color images to Gray 
        # Save into tensors
        x0=torch.from_numpy(grayimL[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]).unsqueeze(0).float()
        x1=torch.from_numpy(grayimR[y_upl:y_upl+self.patch_size,:]).unsqueeze(0).float() # take whole Right image
        # NORMALIZE TILES LOCALLY 
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqnoc=torch.from_numpy(imMasq).to(torch.int16)
        return x0,x1,dispnoc,masqnoc,x_upl
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
class StereoValAerialDatasetSparseN(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=768
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        self.LeftTilN=read_list_from_file(self.images+"/all_left_test.txt")
        self.RightTilN=read_list_from_file(self.images+"/all_right_test.txt")
        self.DispTilN=read_list_from_file(self.images+"/all_disp_sparse_test.txt")
        self.MasqTilN=read_list_from_file(self.images+"/all_masq_test.txt")
        """self.LeftTilN=[os.path.join(self.images,el) for el in self.LeftTilN]
        self.RightTilN=[os.path.join(self.images,el) for el in self.RightTilN]
        self.DispTilN=[os.path.join(self.images,el) for el in self.DispTilN]"""
        self.LeftTilN=self.LeftTilN[0:1000]
        self.RightTilN=self.RightTilN[0:1000]
        self.DispTilN=self.DispTilN[0:1000]
        self.MasqTilN=self.MasqTilN[0:1000]
        #print(self.LeftTilN)
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        imDisp=tff.imread(dispGT)*(-1.0)
        imMasq=tff.imread(masqGT)/255.0
        NotOCC=imDisp*imMasq
        NotOCC=NotOCC[NotOCC!=0.0]
        D_mu=int(np.mean(NotOCC))
        #print("Moyenne de disparite ",D_mu)
        # Randomly pick a region from the left image and compute the corresponding offset in the right image 
        x_upl=D_mu+secrets.randbelow(self.init_size-self.patch_size-D_mu)
        y_upl=secrets.randbelow(self.init_size-self.patch_size)
        imMasq=imMasq[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images
        grayimL=tff.imread(leftimgname)/255.0
        grayimR=tff.imread(rightimgname)/255.0
        # Color images to Gray 
        # Save into tensors
        x0=torch.from_numpy(grayimL[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]).unsqueeze(0).float()
        x1=torch.from_numpy(grayimR[y_upl:y_upl+self.patch_size,:]).unsqueeze(0).float() # take whole Right image
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqnoc=torch.from_numpy(imMasq).to(torch.int16)
        return x0,x1,dispnoc,masqnoc,x_upl
    
############################################################################################
class StereoTrAerialDatasetDenseN(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=768
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        self.LeftTilN=read_list_from_file(self.images+"/l_1_left_train.txt")
        self.RightTilN=read_list_from_file(self.images+"/l_1_right_train.txt")
        self.DispTilN=read_list_from_file(self.images+"/l_1_disp_train.txt")
        self.MasqTilN=read_list_from_file(self.images+"/l_1_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_2_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_2_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_2_disp_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_2_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_31_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_31_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_31_disp_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_31_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_32_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_32_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_32_disp_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_32_masq_train.txt")
        
        #ADDED ENSHEDE DATASET 
        """self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/left_train_clean.txt")
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/right_train_clean.txt")
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/disp_train_clean.txt")
        self.MasqTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/masq_train_clean.txt")"""
 
        #ADDED EUROSRD DATASET DATASET 
        """self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_left_train.txt")[:500]
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_right_train.txt")[:500]
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_disp_train.txt")[:500]
        self.MasqTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_masq_train.txt")[:500]"""
        
    def __len__(self):
        return len(self.LeftTilN)*2
    def __getitem__(self, idx):
        idx=idx%len(self.LeftTilN) 
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        imDisp=tff.imread(dispGT)*(-1.0)
        imMasq=tff.imread(masqGT)/255.0
        NotOCC=imDisp*imMasq
        NotOCC=NotOCC[NotOCC!=0.0]
        D_mu=int(np.mean(NotOCC))
        #print("Moyenne de disparite ++++>>>>>>>>>>>>>>>>>",D_mu)
        # Randomly pick a region from the left image and compute the corresponding offset in the right image 
        x_upl=D_mu+secrets.randbelow(self.init_size-self.patch_size-D_mu)
        y_upl=secrets.randbelow(self.init_size-self.patch_size)
        imMasq=imMasq[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images
        grayimL=tff.imread(leftimgname)/255.0
        grayimR=tff.imread(rightimgname)/255.0
        # Color images to Gray 
        # Save into tensors
        x0=torch.from_numpy(grayimL[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]).unsqueeze(0).float()
        x1=torch.from_numpy(grayimR[y_upl:y_upl+self.patch_size,:]).unsqueeze(0).float() # take whole Right image
        # NORMALIZE TILES LOCALLY 
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqnoc=torch.from_numpy(imMasq).float()#.to(torch.int16)
        masqdef=dispnoc!=0.0
        masqdef=masqdef.float()
        #print("AERIAL SAMPLE  ==> ",x0.shape,x1.shape,dispnoc.shape,masqdef.shape,masqnoc.shape,x_upl)
        return x0,x1,dispnoc,masqdef,masqnoc,x_upl
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
############################################################################################
class StereoTrPatchAerialDatasetDenseN(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.offset_size=300
        self.false1=2
        self.false2=6
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        self.LeftTilN=read_list_from_file(self.images+"/all_clean_left_train.txt")
        self.RightTilN=read_list_from_file(self.images+"/all_clean_right_train.txt")
        self.DispTilN=read_list_from_file(self.images+"/all_clean_disp_train.txt")
        self.MasqTilN=read_list_from_file(self.images+"/all_clean_masq_train.txt")
        
        """self.LeftTilN+=read_list_from_file(self.images+"/l_2_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_2_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_2_disp_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_2_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_31_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_31_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_31_disp_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_31_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_32_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_32_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_32_disp_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_32_masq_train.txt")"""
        
        #ADDED ENSHEDE DATASET 
        """self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/left_train_clean.txt")
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/right_train_clean.txt")
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/disp_train_clean.txt")
        self.MasqTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/masq_train_clean.txt")"""
 
        #ADDED EUROSRD DATASET DATASET 
        """self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_left_train.txt")[:500]
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_right_train.txt")[:500]
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_disp_train.txt")[:500]
        self.MasqTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_masq_train.txt")[:500]"""
        
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        idx=idx%len(self.LeftTilN) 
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        imDisp=tff.imread(dispGT)*(-1.0)
        imMasq=tff.imread(masqGT)/255.0
        imMasq=imMasq[50:self.init_size-50,self.offset_size:self.offset_size+512]
        Defined=np.where(imMasq!=0.0)
        idx_=secrets.randbelow(len(Defined[0]))
        Y_start,X_start=Defined[0][idx_]+50,Defined[1][idx_]+self.offset_size
        dd_=imDisp[Y_start,X_start]
        Offset_neg=((self.false1 - self.false2) * torch.rand(1) + self.false2)
        Offset_neg=Offset_neg.item()
        RandSens=torch.rand(1).item()
        RandSens=(float(RandSens < 0.5)+float(RandSens >= 0.5)*(-1.0))
        Offset_neg=Offset_neg*RandSens
        dd_neg=dd_+Offset_neg
        
        grayimL=tff.imread(leftimgname)/255.0
        grayimR=tff.imread(rightimgname)/255.0
        
        grayimL=grayimL[Y_start-28:Y_start+28,X_start-28:X_start+28]
        grayimRP=grayimR[Y_start-28:Y_start+28,X_start-int(dd_)-28:X_start-int(dd_)+28]
        grayimRN=grayimR[Y_start-28:Y_start+28,X_start-int(dd_neg)-28:X_start-int(dd_neg)+28]
        
        x0=torch.from_numpy(grayimL).unsqueeze(0)
        x1=torch.from_numpy(grayimRP).unsqueeze(0)
        x1n=torch.from_numpy(grayimRN).unsqueeze(0)
        # subsample and create the pyramid 
        xmid,ymid=28,28
        xRes0=x0[:,ymid-3:ymid+4,xmid-3:xmid+4]
        xRes2=x0[:,ymid-7:ymid+7,xmid-7:xmid+7] 
        xRes4=x0[:,ymid-14:ymid+14,xmid-14:xmid+14]
        #print(x_by_Res0.shape,x_by_Res2.shape,x_by_Res4.shape)
        # x here a multi patch set x is 56 x56
        x0_by2 = F.interpolate(xRes2.unsqueeze(0), (7,7),mode='bilinear')
        x0_by4 = F.interpolate(xRes4.unsqueeze(0), (7,7),mode='bilinear')  
        x0_by8 = F.interpolate(x0.unsqueeze(0), (7,7),mode='bilinear')

        x0=torch.cat((xRes0,x0_by2.squeeze(0),x0_by4.squeeze(0),x0_by8.squeeze(0)),0)

        xRes0=x1[:,ymid-3:ymid+4,xmid-3:xmid+4]
        xRes2=x1[:,ymid-7:ymid+7,xmid-7:xmid+7]
        xRes4=x1[:,ymid-14:ymid+14,xmid-14:xmid+14]
        #print(x_by_Res0.shape,x_by_Res2.shape,x_by_Res4.shape)
        # x here a multi patch set x is 56 x56
        x1_by2 = F.interpolate(xRes2.unsqueeze(0), (7,7),mode='bilinear')
        x1_by4 = F.interpolate(xRes4.unsqueeze(0), (7,7),mode='bilinear')  
        x1_by8 = F.interpolate(x1.unsqueeze(0), (7,7),mode='bilinear')

        x1=torch.cat((xRes0,x1_by2.squeeze(0),x1_by4.squeeze(0),x1_by8.squeeze(0)),0)
        
        xRes0=x1n[:,ymid-3:ymid+4,xmid-3:xmid+4]
        xRes2=x1n[:,ymid-7:ymid+7,xmid-7:xmid+7] 
        xRes4=x1n[:,ymid-14:ymid+14,xmid-14:xmid+14]
        #print(x_by_Res0.shape,x_by_Res2.shape,x_by_Res4.shape)
        # x here a multi patch set x is 56 x56
        x1_by2 = F.interpolate(xRes2.unsqueeze(0), (7,7),mode='bilinear')
        x1_by4 = F.interpolate(xRes4.unsqueeze(0), (7,7),mode='bilinear')  
        x1_by8 = F.interpolate(x1n.unsqueeze(0), (7,7),mode='bilinear')

        x1n=torch.cat((xRes0,x1_by2.squeeze(0),x1_by4.squeeze(0),x1_by8.squeeze(0)),0)

        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        x1n=(x1n-self.meanG).div(self.stdG+1e-12)
        #print(x0.shape,x1.shape,x1n.shape)
        return x0.float(),x1.float(),x1n.float()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
class StereoValPatchAerialDatasetDenseN(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.offset_size=300
        self.false1=2
        self.false2=6
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        self.LeftTilN=read_list_from_file(self.images+"/all_clean_left_test.txt")
        self.RightTilN=read_list_from_file(self.images+"/all_clean_right_test.txt")
        self.DispTilN=read_list_from_file(self.images+"/all_clean_disp_test.txt")
        self.MasqTilN=read_list_from_file(self.images+"/all_clean_masq_test.txt")
        #ADDED ENSHEDE DATASET 
        """self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/left_test_clean.txt")
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/right_test_clean.txt")
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/disp_test_clean.txt")
        self.MasqTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/masq_test_clean.txt")
        
        self.LeftTilN=self.LeftTilN[0:1000]+self.LeftTilN[-200:]
        self.RightTilN=self.RightTilN[0:1000]+self.RightTilN[-200:]
        self.DispTilN=self.DispTilN[0:1000]+self.DispTilN[-200:]
        self.MasqTilN=self.MasqTilN[0:1000]+self.MasqTilN[-200:]"""
        
        #ADDED EUROSRD DATASET DATASET 
        """self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_left_train.txt")[500:]
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_right_train.txt")[500:]
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_disp_train.txt")[500:]
        self.MasqTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_masq_train.txt")[500:]"""
        
        #print(self.LeftTilN)
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        idx=idx%len(self.LeftTilN) 
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        imDisp=tff.imread(dispGT)*(-1.0)
        imMasq=tff.imread(masqGT)/255.0
        imMasq=imMasq[50:self.init_size-50,self.offset_size:self.offset_size+512]
        Defined=np.where(imMasq!=0.0)
        idx_=secrets.randbelow(len(Defined[0]))
        Y_start,X_start=Defined[0][idx_]+50,Defined[1][idx_]+self.offset_size
        dd_=imDisp[Y_start,X_start]
        Offset_neg=((self.false1 - self.false2) * torch.rand(1) + self.false2)
        Offset_neg=Offset_neg.item()
        RandSens=torch.rand(1).item()
        RandSens=(float(RandSens < 0.5)+float(RandSens >= 0.5)*(-1.0))
        Offset_neg=Offset_neg*RandSens
        dd_neg=dd_+Offset_neg
        
        grayimL=tff.imread(leftimgname)/255.0
        grayimR=tff.imread(rightimgname)/255.0
        
        grayimL=grayimL[Y_start-28:Y_start+28,X_start-28:X_start+28]
        grayimRP=grayimR[Y_start-28:Y_start+28,X_start-int(dd_)-28:X_start-int(dd_)+28]
        grayimRN=grayimR[Y_start-28:Y_start+28,X_start-int(dd_neg)-28:X_start-int(dd_neg)+28]
        
        x0=torch.from_numpy(grayimL).unsqueeze(0)# 1,56,56
        x1=torch.from_numpy(grayimRP).unsqueeze(0)
        x1n=torch.from_numpy(grayimRN).unsqueeze(0)

        # subsample and create the pyramid 
        xmid,ymid=28,28
        xRes0=x0[:,ymid-3:ymid+4,xmid-3:xmid+4]
        xRes2=x0[:,ymid-7:ymid+7,xmid-7:xmid+7] 
        xRes4=x0[:,ymid-14:ymid+14,xmid-14:xmid+14]
        #print(x_by_Res0.shape,x_by_Res2.shape,x_by_Res4.shape)
        # x here a multi patch set x is 56 x56
        x0_by2 = F.interpolate(xRes2.unsqueeze(0), (7,7),mode='bilinear')
        x0_by4 = F.interpolate(xRes4.unsqueeze(0), (7,7),mode='bilinear')  
        x0_by8 = F.interpolate(x0.unsqueeze(0), (7,7),mode='bilinear')

        x0=torch.cat((xRes0,x0_by2.squeeze(0),x0_by4.squeeze(0),x0_by8.squeeze(0)),0)

        xRes0=x1[:,ymid-3:ymid+4,xmid-3:xmid+4]
        xRes2=x1[:,ymid-7:ymid+7,xmid-7:xmid+7]
        xRes4=x1[:,ymid-14:ymid+14,xmid-14:xmid+14]
        #print(x_by_Res0.shape,x_by_Res2.shape,x_by_Res4.shape)
        # x here a multi patch set x is 56 x56
        x1_by2 = F.interpolate(xRes2.unsqueeze(0), (7,7),mode='bilinear')
        x1_by4 = F.interpolate(xRes4.unsqueeze(0), (7,7),mode='bilinear')  
        x1_by8 = F.interpolate(x1.unsqueeze(0), (7,7),mode='bilinear')

        x1=torch.cat((xRes0,x1_by2.squeeze(0),x1_by4.squeeze(0),x1_by8.squeeze(0)),0)
        
        xRes0=x1n[:,ymid-3:ymid+4,xmid-3:xmid+4]
        xRes2=x1n[:,ymid-7:ymid+7,xmid-7:xmid+7] 
        xRes4=x1n[:,ymid-14:ymid+14,xmid-14:xmid+14]
        #print(x_by_Res0.shape,x_by_Res2.shape,x_by_Res4.shape)
        # x here a multi patch set x is 56 x56
        x1_by2 = F.interpolate(xRes2.unsqueeze(0), (7,7),mode='bilinear')
        x1_by4 = F.interpolate(xRes4.unsqueeze(0), (7,7),mode='bilinear')  
        x1_by8 = F.interpolate(x1n.unsqueeze(0), (7,7),mode='bilinear')

        x1n=torch.cat((xRes0,x1_by2.squeeze(0),x1_by4.squeeze(0),x1_by8.squeeze(0)),0)

        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        x1n=(x1n-self.meanG).div(self.stdG+1e-12)
        #print(x0.shape,x1.shape,x1n.shape)
        return x0.float(),x1.float(),x1n.float()
############################################################################################
class StereoTrAerial4SatDatasetDenseN(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=768
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        self.LeftTilN=read_list_from_file(self.images+"/l_1_left_train.txt")
        self.RightTilN=read_list_from_file(self.images+"/l_1_right_train.txt")
        self.DispTilN=read_list_from_file(self.images+"/l_1_disp_train.txt")
        self.MasqTilN=read_list_from_file(self.images+"/l_1_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_2_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_2_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_2_disp_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_2_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_31_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_31_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_31_disp_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_31_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_32_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_32_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_32_disp_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_32_masq_train.txt")
        
        #ADDED ENSHEDE DATASET 
        self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/left_train_clean.txt")
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/right_train_clean.txt")
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/disp_train_clean.txt")
        self.MasqTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/masq_train_clean.txt")
 
        #ADDED EUROSRD DATASET DATASET 
        self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_left_train.txt")[:500]
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_right_train.txt")[:500]
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_disp_train.txt")[:500]
        self.MasqTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_masq_train.txt")[:500]
        
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        imDisp=tff.imread(dispGT)*(-1.0)
        imMasq=tff.imread(masqGT)/255.0
        NotOCC=imDisp*imMasq
        NotOCC=NotOCC[NotOCC!=0.0]
        D_mu=int(np.mean(NotOCC))
        #print("Moyenne de disparite ++++>>>>>>>>>>>>>>>>>",D_mu)
        # Randomly pick a region from the left image and compute the corresponding offset in the right image 
        x_upl=D_mu+secrets.randbelow(self.init_size-self.patch_size-D_mu)
        y_upl=secrets.randbelow(self.init_size-self.patch_size)
        imMasq=imMasq[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images
        grayimL=tff.imread(leftimgname)/255.0
        grayimR=tff.imread(rightimgname)/255.0
        x0=torch.from_numpy(grayimL)
        x1=torch.from_numpy(grayimR)
        #print(x0.shape,x1.shape)
        # Downsampling and Upsampling 
        if random.random()>0.5:
        # Downsampling and Upsampling 
            x00=F.interpolate(x0.unsqueeze(0).unsqueeze(0), scale_factor=0.25,mode='bilinear')
            x11=F.interpolate(x1.unsqueeze(0).unsqueeze(0), scale_factor=0.25,mode='bilinear')
            x0=F.interpolate(x00, scale_factor=4, mode='bilinear')
            x1=F.interpolate(x11, scale_factor=4, mode='bilinear')
            x0=x0.squeeze()
            x1=x1.squeeze()
        # Color images to Gray 
        # Save into tensors
        x0=x0[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size].unsqueeze(0).float()
        x1=x1[y_upl:y_upl+self.patch_size,:].unsqueeze(0).float() # take whole Right image
        # NORMALIZE TILES LOCALLY 
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqnocc=torch.from_numpy(imMasq).float()#.to(torch.int16)
        masqdef=dispnoc!=0.0
        masqdef=masqdef.float()
        if random.random() > 0.5:
            x0 = T.functional.vflip(x0)
            x1 = T.functional.vflip(x1)
            dispnoc = T.functional.vflip(dispnoc.unsqueeze(0))
            masqdef = T.functional.vflip(masqdef.unsqueeze(0))
            masqnocc = T.functional.vflip(masqnocc.unsqueeze(0))
            dispnoc=dispnoc.squeeze()
            masqdef=masqdef.squeeze()
            masqnocc=masqnocc.squeeze()
        #print("AERIAL SAMPLE  ==> ",x0.shape,x1.shape,dispnoc.shape,masqdef.shape,masqnoc.shape,x_upl)
        return x0,x1,dispnoc,masqdef,masqnocc,x_upl
############################################################################################
############################################################################################
class StereoTrMontPellSatDatasetDenseN(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=768
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        self.LeftTilN=read_list_from_file(self.images+"/Left_Train_List_MPL.txt")
        self.RightTilN=read_list_from_file(self.images+"/Right_Train_List_MPL.txt")
        self.DispTilN=read_list_from_file(self.images+"/Disp_Train_List_MPL.txt")
        self.MasqTilN=read_list_from_file(self.images+"/Masq_Train_List_MPL.txt")
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        imDisp=tff.imread(dispGT)*(-1.0)
        imMasq=tff.imread(masqGT)/255.0
        NotOCC=imDisp*imMasq
        NotOCC=NotOCC[NotOCC!=0.0]
        D_mu=int(np.mean(NotOCC))
        #print("Moyenne de disparite ++++>>>>>>>>>>>>>>>>>",D_mu)
        # Randomly pick a region from the left image and compute the corresponding offset in the right image 
        x_upl=D_mu+secrets.randbelow(self.init_size-self.patch_size-D_mu)
        y_upl=secrets.randbelow(self.init_size-self.patch_size)
        imMasq=imMasq[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images
        grayimL=tff.imread(leftimgname)/255.0
        grayimR=tff.imread(rightimgname)/255.0
        x0=torch.from_numpy(grayimL)
        x1=torch.from_numpy(grayimR)
        #print(x0.shape,x1.shape)
        # Upsamling
        """if random.random()>0.5:
            x00=F.interpolate(x0.unsqueeze(0).unsqueeze(0), scale_factor=0.5,mode='bilinear')
            x11=F.interpolate(x1.unsqueeze(0).unsqueeze(0), scale_factor=0.5,mode='bilinear')
            x0=F.interpolate(x00, scale_factor=2, mode='bilinear')
            x1=F.interpolate(x11, scale_factor=2, mode='bilinear')
            x0=x0.squeeze()
            x1=x1.squeeze()"""
        # Color images to Gray 
        # Save into tensors
        x0=x0[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size].unsqueeze(0).float()
        x1=x1[y_upl:y_upl+self.patch_size,:].unsqueeze(0).float() # take whole Right image
        # NORMALIZE TILES LOCALLY 
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqnocc=torch.from_numpy(imMasq).float()#.to(torch.int16)
        masqdef=dispnoc!=0.0
        masqdef=masqdef.float()
        if random.random() > 0.5:
            x0 = T.functional.vflip(x0)
            x1 = T.functional.vflip(x1)
            dispnoc = T.functional.vflip(dispnoc.unsqueeze(0))
            masqdef = T.functional.vflip(masqdef.unsqueeze(0))
            masqnocc = T.functional.vflip(masqnocc.unsqueeze(0))
            dispnoc=dispnoc.squeeze()
            masqdef=masqdef.squeeze()
            masqnocc=masqnocc.squeeze()
        #print("AERIAL SAMPLE  ==> ",x0.shape,x1.shape,dispnoc.shape,masqdef.shape,masqnoc.shape,x_upl)
        return x0,x1,dispnoc,masqdef,masqnocc,x_upl
############################################################################################
############################################################################################
class StereoValMontPellSatDatasetDenseN(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=768
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        self.LeftTilN=read_list_from_file(self.images+"/Left_Val_List_MPL.txt")
        self.RightTilN=read_list_from_file(self.images+"/Right_Val_List_MPL.txt")
        self.DispTilN=read_list_from_file(self.images+"/Disp_Val_List_MPL.txt")
        self.MasqTilN=read_list_from_file(self.images+"/Masq_Val_List_MPL.txt")
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        imDisp=tff.imread(dispGT)*(-1.0)
        imMasq=tff.imread(masqGT)/255.0
        NotOCC=imDisp*imMasq
        NotOCC=NotOCC[NotOCC!=0.0]
        D_mu=int(np.mean(NotOCC))
        #print("Moyenne de disparite ++++>>>>>>>>>>>>>>>>>",D_mu)
        # Randomly pick a region from the left image and compute the corresponding offset in the right image 
        #x_upl=D_mu+secrets.randbelow(self.init_size-self.patch_size-D_mu)
        #y_upl=secrets.randbelow(self.init_size-self.patch_size)
        x_upl,y_upl=128,128
        imMasq=imMasq[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images
        grayimL=tff.imread(leftimgname)/255.0
        grayimR=tff.imread(rightimgname)/255.0
        x0=torch.from_numpy(grayimL)
        x1=torch.from_numpy(grayimR)
        #print(x0.shape,x1.shape)
        # Save into tensors
        x0=x0[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size].unsqueeze(0).float()
        x1=x1[y_upl:y_upl+self.patch_size,:].unsqueeze(0).float() # take whole Right image
        # NORMALIZE TILES LOCALLY 
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqnocc=torch.from_numpy(imMasq).float()#.to(torch.int16)
        masqdef=dispnoc!=0.0
        masqdef=masqdef.float()
        #print("AERIAL SAMPLE  ==> ",x0.shape,x1.shape,dispnoc.shape,masqdef.shape,masqnoc.shape,x_upl)
        return x0,x1,dispnoc,masqdef,masqnocc,x_upl
############################################################################################
class StereoTrSatDatasetDenseN(Dataset):
    def __init__(self,images,mean,std,nameDataset,Mult,MIN_SAT=0.0,MAX_SAT=65535.0):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.Multiplier=Mult
        self.patch_size=768
        self.ND=nameDataset #read_MinMax_from_file
        self.LeftTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/RVL_SatStereo' +"/Left_Train_List_RVL.txt")
        self.MinMaxL =read_MinMax_from_file('/home/ad/alichem/scratch/stereo_test/Data/RVL_SatStereo' +"/Left_Train_List_RVL.txt")
        # RIGHT CHIPS 
        self.RightTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/RVL_SatStereo' +"/Right_Train_List_RVL.txt")
        self.MinMaxR =read_MinMax_from_file('/home/ad/alichem/scratch/stereo_test/Data/RVL_SatStereo' +"/Right_Train_List_RVL.txt")
        # DISPARITY CHIPS 
        self.DispTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/RVL_SatStereo' +"/Disp_Train_List_RVL.txt")
        self.MasqTilN=[el.replace('Disparity','DefMasks') for el in self.DispTilN]
        self.MIN_SAT=MIN_SAT
        self.MAX_SAT=MAX_SAT
        
    def __len__(self):
        return len(self.LeftTilN)*self.Multiplier
    def __getitem__(self, idx):
        idx=idx%len(self.LeftTilN) 
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        #[aMinL,aMaxL]=self.MinMaxL[idx]
        #[aMinR,aMaxR]=self.MinMaxR[idx]
        with rasterio.open(leftimgname) as grayimL:
            HGT,WIDTH=grayimL.height,grayimL.width
            with rasterio.open(rightimgname) as grayimR:
                with rasterio.open(dispGT) as imDisp:
                    imMasq=tff.imread(masqGT)
                    Zone2Sample=cv2.erode(imMasq, np.ones((57, 57), 'uint8'), iterations=5)
                    PS0=1024
                    PS0_2=int(PS0/2)
                    PS_2=int(self.patch_size/2)
                    # Random select elements inside the Defined Mask
                    Defined=np.where(Zone2Sample!=0)
                    idx_=secrets.randbelow(len(Defined[0]))
                    Y_start,X_start=Defined[0][idx_],Defined[1][idx_]
                    if X_start-PS_2<0:
                        X_inf=0
                        X_sup=self.patch_size
                    elif X_start+PS_2>WIDTH:
                        X_inf=WIDTH-self.patch_size
                        X_sup=WIDTH  
                    else:
                        X_inf=X_start-PS_2
                        X_sup=X_start+PS_2

                    if Y_start-PS_2<0:
                        Y_inf=0
                        Y_sup=self.patch_size
                    elif Y_start+PS_2>HGT:
                        Y_inf=HGT-self.patch_size
                        Y_sup=HGT  
                    else:
                        Y_inf=Y_start-PS_2
                        Y_sup=Y_start+PS_2
                    # WINDOW READING FROM RASTERIO OBJECTS 
                    DISP=imDisp.read(1, window=Window(X_inf, Y_inf, self.patch_size, self.patch_size))*(-1.0)
                    MASQ=imMasq[Y_inf:Y_sup,X_inf:X_sup]/255.0
                    # Get Min Offset
                    X_0=np.arange(0,self.patch_size)
                    X_0=np.tile(X_0,(self.patch_size,1))
                    X_0=X_0+X_inf
                    X_R=X_0-DISP.round()
                    X_DEF=X_R[~(X_R!=X_R)] 
                    aMin=int(np.min(X_DEF))
                    # Save into tensors
                    x0=torch.from_numpy(grayimL.read(1, window=Window(X_inf, Y_inf, self.patch_size, self.patch_size))).unsqueeze(0).float()
                    x0[x0<0.0]=0.0
                    x0=(x0-self.MIN_SAT)/(self.MAX_SAT-self.MIN_SAT)
                    x1t=torch.from_numpy(grayimR.read(1,window=Window(aMin,Y_inf,PS0,self.patch_size))).unsqueeze(0).float() # take whole Right image
                    x1t[x1t<0.0]=0.0
                    x1t=(x1t-self.MIN_SAT)/(self.MAX_SAT-self.MIN_SAT)
                    if x1t.shape[-1]<PS0:
                        x1=torch.zeros((1,self.patch_size,PS0)).float()
                        x1[:,:,0:x1t.shape[-1]].copy_(x1t)
                    else:
                        x1=x1t
                    # NORMALIZE TILES LOCALLY 
                    # NORMALIZE GLOBALLY
                    masqnocc=~(DISP!=DISP)
                    masqnocc=torch.from_numpy(masqnocc).float()#.to(torch.int16)
                    DISP[DISP!=DISP]=0.0
                    dispnoc=torch.from_numpy(DISP)
                    masqdef=torch.from_numpy(MASQ).float()#.to(torch.int16)
                    #print("SAT RVL SAMPLE  ==> ",x0.shape,x1.shape,dispnoc.shape,masqdef.shape,masqnocc.shape)
                    # Random vertical flipping
                    if random.random() > 0.5:
                        x0 = T.functional.vflip(x0)
                        x1 = T.functional.vflip(x1)
                        dispnoc = T.functional.vflip(dispnoc.unsqueeze(0))
                        masqdef = T.functional.vflip(masqdef.unsqueeze(0))
                        masqnocc = T.functional.vflip(masqnocc.unsqueeze(0))
                        dispnoc=dispnoc.squeeze()
                        masqdef=masqdef.squeeze()
                        masqnocc=masqnocc.squeeze()
                    # Apply Horizontal Shear on images , masks and ground truth 
                    """if random.random() > 0.5:
                        RandAff=T.RandomAffine(shear=5)
                        x0 = RandAff(x0)
                        x1 = RandAff(x1)
                        dispnoc = RandAff(dispnoc)
                        masqdef = RandAff(masqdef)
                        masqnocc = RandAff(masqnocc)"""
                    return x0,x1,dispnoc,masqdef,masqnocc,X_inf-aMin
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

############################################ WHU STEREO DATASET ##############################
class StereoTrWHUDatasetDenseN(Dataset):
    def __init__(self,images,mean,std,nameDataset,Mult,MIN_SAT=0.0,MAX_SAT=3041.0):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.Multiplier=Mult
        self.patch_size=768
        self.ND=nameDataset 
        # LEFT CHIPS
        self.LeftTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO' +"/Left_Train_List_WHU.txt")
        # RIGHT CHIPS 
        self.RightTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Right_Train_List_WHU.txt")
        # DISPARITY CHIPS 
        self.DispTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO' +"/Disp_Train_List_WHU.txt")
        self.MasqTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO' +"/Masq_Train_List_WHU.txt")
        self.MIN_SAT=MIN_SAT
        self.MAX_SAT=MAX_SAT
        
    def __len__(self):
        return len(self.LeftTilN)*self.Multiplier
    def __getitem__(self, idx):
        idx=idx%len(self.LeftTilN) 
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        imMasqNOCC=tff.imread(masqGT)/255.0
        grayimL=tff.imread(leftimgname).astype(np.int16)
        HGT,WIDTH=grayimL.shape[0],grayimL.shape[1]
        grayimR=tff.imread(rightimgname).astype(np.int16)
        imDisp=tff.imread(dispGT)
        imMasq=(-128.0<imDisp)*(imDisp<128.0)
        Zone2Sample=np.zeros((HGT,WIDTH))
        Zone2Sample[HGT//2-self.patch_size//2:HGT//2+self.patch_size//2,WIDTH//2-self.patch_size//2:WIDTH//2+self.patch_size//2]=1
        PS0=1024
        PS0_2=int(PS0/2)
        PS_2=int(self.patch_size/2)
        # Random select elements inside the Defined Mask
        Defined=np.where(Zone2Sample!=0)
        idx_=secrets.randbelow(len(Defined[0]))
        Y_start,X_start=Defined[0][idx_],Defined[1][idx_]
        if X_start-PS_2<0:
            X_inf=0
            X_sup=self.patch_size
        elif X_start+PS_2>WIDTH:
            X_inf=WIDTH-self.patch_size
            X_sup=WIDTH  
        else:
            X_inf=X_start-PS_2
            X_sup=X_start+PS_2

        if Y_start-PS_2<0:
            Y_inf=0
            Y_sup=self.patch_size
        elif Y_start+PS_2>HGT:
            Y_inf=HGT-self.patch_size
            Y_sup=HGT  
        else:
            Y_inf=Y_start-PS_2
            Y_sup=Y_start+PS_2
        # SHEAR TO APPLY HERE ON WHOLE IMAGES 
        masqdef =torch.from_numpy(imMasq).float()
        masqnocc=torch.from_numpy(imMasqNOCC).float()
        dispnoc  =torch.from_numpy(imDisp)
        x0=torch.from_numpy(grayimL).unsqueeze(0).float()
        x1=torch.from_numpy(grayimR).unsqueeze(0).float() 
        # Apply random shear on all data before vertical flipping 
        # Apply Horizontal Shear on images , masks and ground truth 
        if random.random() > 0.5:
            degs,trans,scale,shears=T.RandomAffine.get_params(degrees=[0.0,0.0],translate=None,scale_ranges=None, shears=[-10.0,10.0],img_size=[PS0,PS0])
            x0 = T.functional.affine(x0,0.0,[0,0],1.0, list(shears))
            x1 = T.functional.affine(x1,0.0,[0,0],1.0, list(shears))
            dispnoc = T.functional.affine(dispnoc.unsqueeze(0),0.0,[0,0],1.0, list(shears))
            masqdef = T.functional.affine(masqdef.unsqueeze(0),0.0,[0,0],1.0, list(shears))
            masqnocc = T.functional.affine(masqnocc.unsqueeze(0),0.0,[0,0],1.0, list(shears))
            dispnoc=dispnoc.squeeze()
            masqdef=masqdef.squeeze()
            masqnocc=masqnocc.squeeze()
        # WINDOW READING FROM RASTERIO OBJECTS 
        dispnoc=dispnoc[Y_inf:Y_sup,X_inf:X_sup]
        masqdef=masqdef[Y_inf:Y_sup,X_inf:X_sup]
        masqnocc=masqnocc[Y_inf:Y_sup,X_inf:X_sup]
        # Save into tensors
        x0=x0[:,Y_inf:Y_sup,X_inf:X_sup]
        x1=x1[:,Y_inf:Y_sup,:]# take whole Right image
        x0=(x0-self.MIN_SAT)/(self.MAX_SAT-self.MIN_SAT)
        x1=(x1-self.MIN_SAT)/(self.MAX_SAT-self.MIN_SAT)
        # NORMALIZE TILES LOCALLY 
        # NORMALIZE GLOBALLY
        x0=(x0-self.meanG).div(self.stdG)
        x1=(x1-self.meanG).div(self.stdG)
        #print("SAT RVL SAMPLE  ==> ",x0.shape,x1.shape,dispnoc.shape,masqdef.shape,masqnocc.shape)
        # Random vertical flipping
        if random.random() > 0.5:
            x0 = T.functional.vflip(x0)
            x1 = T.functional.vflip(x1)
            dispnoc = T.functional.vflip(dispnoc.unsqueeze(0))
            masqdef = T.functional.vflip(masqdef.unsqueeze(0))
            masqnocc = T.functional.vflip(masqnocc.unsqueeze(0))
            dispnoc=dispnoc.squeeze()
            masqdef=masqdef.squeeze()
            masqnocc=masqnocc.squeeze()
        return x0,x1,dispnoc,masqdef,masqnocc,X_inf
                         
##############################################################################################################################
class StereoTrDFCDatasetDenseN(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=768
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #ADDED ENSHEDE DATASET 
        self.LeftTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Left_train_Tr1.txt")[:-20]
        self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Left_train_Tr2.txt")[:-20]
        self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Left_train_Tr3.txt")[:-20]
        self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Left_train_Tr4.txt")[:-20]
        
        self.RightTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Right_train_Tr1.txt")[:-20]
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Right_train_Tr2.txt")[:-20]
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Right_train_Tr3.txt")[:-20]
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Right_train_Tr4.txt")[:-20]
        
        self.DispTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Disp_train_Tr1.txt")[:-20]
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Disp_train_Tr2.txt")[:-20]
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Disp_train_Tr3.txt")[:-20]
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Disp_train_Tr4.txt")[:-20]

        #print(self.LeftTilN)
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        #idx=idx%len(self.LeftTilN)
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        imDisp=tff.imread(dispGT)
        Masqdef=imDisp!=-999
        x_upl=secrets.randbelow(self.init_size-self.patch_size)
        y_upl=secrets.randbelow(self.init_size-self.patch_size)
        Masqdef=Masqdef[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images 
        grayimL=np.array(Image.fromarray(tff.imread(leftimgname)).convert('L'))/255.0
        grayimR=np.array(Image.fromarray(tff.imread(rightimgname)).convert('L'))/255.0
        # Color images to Gray 
        # Save into tensors
        x0=torch.from_numpy(grayimL[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]).unsqueeze(0).float()
        x1=torch.from_numpy(grayimR[y_upl:y_upl+self.patch_size,:]).unsqueeze(0).float() # take whole Right image
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqdef=torch.from_numpy(Masqdef).float()
        masqnoc=torch.ones(dispnoc.shape)#.to(torch.int16)
        return x0,x1,dispnoc,masqdef,masqnoc,x_upl
    
##############################################################################################################################
class StereoTrDFCDatasetDenseNocc(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=768
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #ADDED ENSHEDE DATASET 
        self.LeftTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Left_train_Tr1.txt")[:-20]
        self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Left_train_Tr2.txt")[:-20]
        self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Left_train_Tr3.txt")[:-20]
        self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Left_train_Tr4.txt")[:-20]
        
        self.RightTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Right_train_Tr1.txt")[:-20]
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Right_train_Tr2.txt")[:-20]
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Right_train_Tr3.txt")[:-20]
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Right_train_Tr4.txt")[:-20]
        
        self.DispTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Dense_Disp_train_Tr1.txt")[:-20]
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Dense_Disp_train_Tr2.txt")[:-20]
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Dense_Disp_train_Tr3.txt")[:-20]
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Dense_Disp_train_Tr4.txt")[:-20]
        
        # MASKS
        self.MasqTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Masq_train_Tr1.txt")[:-20]
        self.MasqTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Masq_train_Tr2.txt")[:-20]
        self.MasqTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Masq_train_Tr3.txt")[:-20]
        self.MasqTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Masq_train_Tr4.txt")[:-20]

        #print(self.LeftTilN)
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        #idx=idx%len(self.LeftTilN)
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        imMasqNOCC=tff.imread(masqGT)/255.0
        imDisp=tff.imread(dispGT)
        Masqdef=imDisp!=-999
        x_upl=secrets.randbelow(self.init_size-self.patch_size)
        y_upl=secrets.randbelow(self.init_size-self.patch_size)
        
        # Read images 
        grayimL=np.array(Image.fromarray(tff.imread(leftimgname)).convert('L'))/255.0
        grayimR=np.array(Image.fromarray(tff.imread(rightimgname)).convert('L'))/255.0
        
        # SHEAR TO APPLY HERE ON WHOLE IMAGES 
        masqdef =torch.from_numpy(Masqdef).float()
        masqnocc=torch.from_numpy(imMasqNOCC).float()
        dispnoc  =torch.from_numpy(imDisp)
        x0=torch.from_numpy(grayimL).unsqueeze(0).float()
        x1=torch.from_numpy(grayimR).unsqueeze(0).float() 
        # Apply random shear on all data before vertical flipping 
        # Apply Horizontal Shear on images , masks and ground truth 
        if random.random() > 0.5:
            degs,trans,scale,shears=T.RandomAffine.get_params(degrees=[0.0,0.0],translate=None,scale_ranges=None, shears=[-10.0,10.0],img_size=[self.init_size,self.init_size])
            x0 = T.functional.affine(x0,0.0,[0,0],1.0, list(shears))
            x1 = T.functional.affine(x1,0.0,[0,0],1.0, list(shears))
            dispnoc = T.functional.affine(dispnoc.unsqueeze(0),0.0,[0,0],1.0, list(shears))
            masqdef = T.functional.affine(masqdef.unsqueeze(0),0.0,[0,0],1.0, list(shears))
            masqnocc = T.functional.affine(masqnocc.unsqueeze(0),0.0,[0,0],1.0, list(shears))
            dispnoc=dispnoc.squeeze()
            masqdef=masqdef.squeeze()
            masqnocc=masqnocc.squeeze()
        x0=x0[:,y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        x1=x1[:,y_upl:y_upl+self.patch_size,:] # take whole Right image
        masqdef=masqdef[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        masqnocc=masqnocc[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        dispnoc=dispnoc[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        # Random vertical flipping
        if random.random() > 0.5:
            x0 = T.functional.vflip(x0)
            x1 = T.functional.vflip(x1)
            dispnoc  = T.functional.vflip(dispnoc.unsqueeze(0))
            masqdef  = T.functional.vflip(masqdef.unsqueeze(0))
            masqnocc = T.functional.vflip(masqnocc.unsqueeze(0))
            dispnoc=dispnoc.squeeze()
            masqdef=masqdef.squeeze()
            masqnocc=masqnocc.squeeze()
        return x0,x1,dispnoc,masqdef,masqnocc,x_upl
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
class StereoValAerialDatasetDenseN(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=768
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        self.LeftTilN=read_list_from_file(self.images+"/all_left_test.txt")
        self.RightTilN=read_list_from_file(self.images+"/all_right_test.txt")
        self.DispTilN=read_list_from_file(self.images+"/all_disp_test.txt")
        self.MasqTilN=read_list_from_file(self.images+"/all_masq_test.txt")
        #ADDED ENSHEDE DATASET 
        """self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/left_test_clean.txt")
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/right_test_clean.txt")
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/disp_test_clean.txt")
        self.MasqTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/ENSH_DENSE'+"/masq_test_clean.txt")"""
        
        self.LeftTilN=self.LeftTilN[0:1000]
        self.RightTilN=self.RightTilN[0:1000]
        self.DispTilN=self.DispTilN[0:1000]
        self.MasqTilN=self.MasqTilN[0:1000]
        
        #ADDED EUROSRD DATASET DATASET 
        """self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_left_train.txt")[500:]
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_right_train.txt")[500:]
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_disp_train.txt")[500:]
        self.MasqTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/EuroSDR_vaihingen'+"/eurosdr_vahingen_masq_train.txt")[500:]"""
        
        """self.LeftTilN=[os.path.join(self.images,el) for el in self.LeftTilN]
        self.RightTilN=[os.path.join(self.images,el) for el in self.RightTilN]
        self.DispTilN=[os.path.join(self.images,el) for el in self.DispTilN]"""

        #print(self.LeftTilN)
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        imDisp=tff.imread(dispGT)*(-1.0)
        imMasq=tff.imread(masqGT)/255.0
        NotOCC=imDisp*imMasq
        NotOCC=NotOCC[NotOCC!=0.0]
        D_mu=int(np.mean(NotOCC))
        #print("Moyenne de disparite ",D_mu)
        # Randomly pick a region from the left image and compute the corresponding offset in the right image 
        x_upl=D_mu+secrets.randbelow(self.init_size-self.patch_size-D_mu)
        y_upl=secrets.randbelow(self.init_size-self.patch_size)
        imMasq=imMasq[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images
        grayimL=tff.imread(leftimgname)/255.0
        grayimR=tff.imread(rightimgname)/255.0
        # Color images to Gray 
        # Save into tensors
        x0=torch.from_numpy(grayimL[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]).unsqueeze(0).float()
        x1=torch.from_numpy(grayimR[y_upl:y_upl+self.patch_size,:]).unsqueeze(0).float() # take whole Right image
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqnoc=torch.from_numpy(imMasq).float()#.to(torch.int16)
        masqdef=dispnoc!=0.0
        masqdef=masqdef.float()
        return x0,x1,dispnoc,masqdef,masqnoc,x_upl

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
class StereoCheckDistribAerialDatasetN(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=384
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        self.LeftTilN=read_list_from_file(self.images+"/all_left_test.txt")
        self.RightTilN=read_list_from_file(self.images+"/all_right_test.txt")
        self.DispTilN=read_list_from_file(self.images+"/all_disp_test.txt")
        self.MasqTilN=read_list_from_file(self.images+"/all_masq_test.txt")
        self.LeftTilN=self.LeftTilN[1400:2000]
        self.RightTilN=self.RightTilN[1400:2000]
        self.DispTilN=self.DispTilN[1400:2000]
        self.MasqTilN=self.MasqTilN[1400:2000]
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        imDisp=tff.imread(dispGT)*(-1.0)
        imMasq=tff.imread(masqGT)/255.0
        NotOCC=imDisp*imMasq
        NotOCC=NotOCC[NotOCC!=0.0]
        D_mu=int(np.mean(NotOCC))
        #print("Moyenne de disparite ",D_mu)
        # Randomly pick a region from the left image and compute the corresponding offset in the right image 
        x_upl=D_mu+secrets.randbelow(self.init_size-self.patch_size-D_mu)
        y_upl=secrets.randbelow(self.init_size-self.patch_size)
        imMasq=imMasq[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images
        grayimL=tff.imread(leftimgname)/255.0
        grayimR=tff.imread(rightimgname)/255.0
        # Color images to Gray 
        # Save into tensors
        x0=torch.from_numpy(grayimL[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]).unsqueeze(0).float()
        x1=torch.from_numpy(grayimR[y_upl:y_upl+self.patch_size,:]).unsqueeze(0).float() # take whole Right image
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqnoc=torch.from_numpy(imMasq).float()#.to(torch.int16)
        masqdef=dispnoc!=0.0
        masqdef=masqdef.float()
        return x0,x1,dispnoc,masqnoc,x_upl   
############################################################################################
class StereoValSatDatasetDenseN(Dataset):
    def __init__(self,images,mean,std,nameDataset,Mult,MIN_SAT=0.0,MAX_SAT=65535.0):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.Multiplier=Mult
        self.patch_size=768
        self.ND=nameDataset #read_MinMax_from_file
        self.LeftTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/RVL_SatStereo' +"/Left_Val_List_RVL.txt")
        self.MinMaxL =read_MinMax_from_file('/home/ad/alichem/scratch/stereo_test/Data/RVL_SatStereo' +"/Left_Val_List_RVL.txt")
        # RIGHT CHIPS 
        self.RightTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/RVL_SatStereo' +"/Right_Val_List_RVL.txt")
        self.MinMaxR =read_MinMax_from_file('/home/ad/alichem/scratch/stereo_test/Data/RVL_SatStereo' +"/Right_Val_List_RVL.txt")
        # DISPARITY CHIPS 
        self.DispTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/RVL_SatStereo' +"/Disp_Val_List_RVL.txt")
        self.MasqTilN=[el.replace('Disparity','DefMasks') for el in self.DispTilN]
        self.MIN_SAT=MIN_SAT
        self.MAX_SAT=MAX_SAT
    def __len__(self):
        return len(self.LeftTilN)*self.Multiplier
    def __getitem__(self, idx):
        idx=idx%len(self.LeftTilN)
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        #[aMinL,aMaxL]=self.MinMaxL[idx]
        #[aMinR,aMaxR]=self.MinMaxR[idx]
        with rasterio.open(leftimgname) as grayimL:
            HGT,WIDTH=grayimL.height,grayimL.width
            with rasterio.open(rightimgname) as grayimR:
                with rasterio.open(dispGT) as imDisp:
                    imMasq=tff.imread(masqGT)
                    Zone2Sample=cv2.erode(imMasq, np.ones((57, 57), 'uint8'), iterations=5)
                    PS0=1024
                    PS0_2=int(PS0/2)
                    PS_2=int(self.patch_size/2)
                    # Random select elements inside the Defined Mask
                    Defined=np.where(Zone2Sample!=0)
                    idx_=secrets.randbelow(len(Defined[0]))
                    Y_start,X_start=Defined[0][idx_],Defined[1][idx_]
                    print("y_start, x_start ",Y_start,X_start)
                    if X_start-PS_2<0:
                        X_inf=0
                        X_sup=self.patch_size
                    elif X_start+PS_2>WIDTH:
                        X_inf=WIDTH-self.patch_size
                        X_sup=WIDTH  
                    else:
                        X_inf=X_start-PS_2
                        X_sup=X_start+PS_2

                    if Y_start-PS_2<0:
                        Y_inf=0
                        Y_sup=self.patch_size
                    elif Y_start+PS_2>HGT:
                        Y_inf=HGT-self.patch_size
                        Y_sup=HGT  
                    else:
                        Y_inf=Y_start-PS_2
                        Y_sup=Y_start+PS_2
                    # WINDOW READING FROM RASTERIO OBJECTS 
                    DISP=imDisp.read(1, window=Window(X_inf, Y_inf, self.patch_size, self.patch_size))*(-1.0)
                    MASQ=imMasq[Y_inf:Y_sup,X_inf:X_sup]/255.0
                    # Get Min Offset
                    X_0=np.arange(0,self.patch_size)
                    X_0=np.tile(X_0,(self.patch_size,1))
                    X_0=X_0+X_inf
                    X_R=X_0-DISP.round()
                    X_DEF=X_R[~(X_R!=X_R)] 
                    """if len(X_DEF)==0:
                        print(leftimgname,Y_start,X_start)"""
                    aMin=int(np.min(X_DEF))
                    # Save into tensors
                    x0=torch.from_numpy(grayimL.read(1, window=Window(X_inf, Y_inf, self.patch_size, self.patch_size))).unsqueeze(0).float()
                    x0[x0<0.0]=0.0
                    x0=(x0-self.MIN_SAT)/(self.MAX_SAT-self.MIN_SAT)
                    x1t=torch.from_numpy(grayimR.read(1,window=Window(aMin,Y_inf,PS0,self.patch_size))).unsqueeze(0).float() # take whole Right image
                    x1t[x1t<0.0]=0.0
                    x1t=(x1t-self.MIN_SAT)/(self.MAX_SAT-self.MIN_SAT)
                    if x1t.shape[-1]<PS0:
                        x1=torch.zeros((1,self.patch_size,PS0)).float()
                        x1[:,:,0:x1t.shape[-1]].copy_(x1t)
                    else:
                        x1=x1t
                    # NORMALIZE TILES LOCALLY 
                    # NORMALIZE GLOBALLY
                    masqnocc=~(DISP!=DISP)
                    masqnocc=torch.from_numpy(masqnocc).float()#.to(torch.int16)
                    DISP[DISP!=DISP]=0.0
                    dispnoc=torch.from_numpy(DISP)
                    masqdef=torch.from_numpy(MASQ).float()#.to(torch.int16)
                    #print("SAT RVL SAMPLE  ==> ",x0.shape,x1.shape,dispnoc.shape,masqdef.shape,masqnocc.shape)
                    return x0,x1,dispnoc,masqdef,masqnocc,X_inf-aMin
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
############################################ WHU STEREO DATASET ##############################
class StereoValWHUDatasetDenseN(Dataset):
    def __init__(self,images,mean,std,nameDataset,Mult,MIN_SAT=0.0,MAX_SAT=3041.0):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.Multiplier=Mult
        self.patch_size=768
        self.ND=nameDataset 
        # LEFT CHIPS
        self.LeftTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO' +"/Left_Val_List_WHU.txt")
        # RIGHT CHIPS 
        self.RightTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Right_Val_List_WHU.txt")
        # DISPARITY CHIPS 
        self.DispTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO' +"/Disp_Val_List_WHU.txt")
        self.MasqTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO' +"/Masq_Val_List_WHU.txt")
        self.MIN_SAT=MIN_SAT
        self.MAX_SAT=MAX_SAT
        
    def __len__(self):
        return len(self.LeftTilN)*self.Multiplier
    def __getitem__(self, idx):
        idx=idx%len(self.LeftTilN) 
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        imMasqNOCC=tff.imread(masqGT)/255.0
        grayimL=tff.imread(leftimgname).astype(np.int16)
        HGT,WIDTH=grayimL.shape[0],grayimL.shape[1]
        grayimR=tff.imread(rightimgname).astype(np.int16)
        imDisp=tff.imread(dispGT)
        imMasq=(-128.0<imDisp)*(imDisp<128.0)
        Zone2Sample=np.zeros((HGT,WIDTH))
        Zone2Sample[HGT//2-self.patch_size//2:HGT//2+self.patch_size//2,WIDTH//2-self.patch_size//2:WIDTH//2+self.patch_size//2]=1
        PS0=1024
        PS0_2=int(PS0/2)
        PS_2=int(self.patch_size/2)
        # Random select elements inside the Defined Mask
        Defined=np.where(Zone2Sample!=0)
        idx_=secrets.randbelow(len(Defined[0]))
        #Y_start,X_start=Defined[0][idx_],Defined[1][idx_]
        Y_start,X_start=512,512
        if X_start-PS_2<0:
            X_inf=0
            X_sup=self.patch_size
        elif X_start+PS_2>WIDTH:
            X_inf=WIDTH-self.patch_size
            X_sup=WIDTH  
        else:
            X_inf=X_start-PS_2
            X_sup=X_start+PS_2

        if Y_start-PS_2<0:
            Y_inf=0
            Y_sup=self.patch_size
        elif Y_start+PS_2>HGT:
            Y_inf=HGT-self.patch_size
            Y_sup=HGT  
        else:
            Y_inf=Y_start-PS_2
            Y_sup=Y_start+PS_2
        # WINDOW READING FROM RASTERIO OBJECTS 
        DISP=imDisp[Y_inf:Y_sup,X_inf:X_sup]
        MASQ=imMasq[Y_inf:Y_sup,X_inf:X_sup]
        MASQNOCC=imMasqNOCC[Y_inf:Y_sup,X_inf:X_sup]
        # Save into tensors
        x0=torch.from_numpy(grayimL[Y_inf:Y_sup,X_inf:X_sup]).unsqueeze(0).float()
        x0=(x0-self.MIN_SAT)/(self.MAX_SAT-self.MIN_SAT)
        x1=torch.from_numpy(grayimR[Y_inf:Y_sup,:]).unsqueeze(0).float() # take whole Right image
        x1=(x1-self.MIN_SAT)/(self.MAX_SAT-self.MIN_SAT)
        # NORMALIZE TILES LOCALLY 
        # NORMALIZE GLOBALLY
        x0=(x0-self.meanG).div(self.stdG)
        x1=(x1-self.meanG).div(self.stdG)
        #DISP[~MASQ]=0.0
        dispnoc=torch.from_numpy(DISP)
        masqdef=torch.from_numpy(MASQ).float()#.to(torch.int16)
        #masqnocc=torch.ones(dispnoc.shape)#.to(torch.int16)
        masqnocc=torch.from_numpy(MASQNOCC).float()
        #print("SAT RVL SAMPLE  ==> ",x0.shape,x1.shape,dispnoc.shape,masqdef.shape,masqnocc.shape)
        return x0,x1,dispnoc,masqdef,masqnocc,X_inf
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
class StereoValDFCDatasetDenseN(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=768
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #ADDED ENSHEDE DATASET 
        self.LeftTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Left_train_Tr1.txt")[-20:]
        self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Left_train_Tr2.txt")[-20:]
        self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Left_train_Tr3.txt")[-20:]
        self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Left_train_Tr4.txt")[-20:]
        
        self.RightTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Right_train_Tr1.txt")[-20:]
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Right_train_Tr2.txt")[-20:]
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Right_train_Tr3.txt")[-20:]
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Right_train_Tr4.txt")[-20:]
        
        self.DispTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Disp_train_Tr1.txt")[-20:]
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Disp_train_Tr2.txt")[-20:]
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Disp_train_Tr3.txt")[-20:]
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Disp_train_Tr4.txt")[-20:]

        #print(self.LeftTilN)
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        #idx=idx%len(self.LeftTilN)
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        imDisp=tff.imread(dispGT)
        Masqdef=imDisp!=-999
        x_upl=secrets.randbelow(self.init_size-self.patch_size)
        y_upl=secrets.randbelow(self.init_size-self.patch_size)
        Masqdef=Masqdef[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images 
        grayimL=np.array(Image.fromarray(tff.imread(leftimgname)).convert('L'))/255.0
        grayimR=np.array(Image.fromarray(tff.imread(rightimgname)).convert('L'))/255.0
        # Color images to Gray 
        # Save into tensors
        x0=torch.from_numpy(grayimL[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]).unsqueeze(0).float()
        x1=torch.from_numpy(grayimR[y_upl:y_upl+self.patch_size,:]).unsqueeze(0).float() # take whole Right image
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqdef=torch.from_numpy(Masqdef).float()
        masqnoc=torch.ones(dispnoc.shape)#.to(torch.int16)
        return x0,x1,dispnoc,masqdef,masqnoc,x_upl
    
##############################################################################################################################
class StereoValDFCDatasetDenseNocc(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=768
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #ADDED ENSHEDE DATASET 
        self.LeftTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Left_train_Tr1.txt")[-20:]
        self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Left_train_Tr2.txt")[-20:]
        self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Left_train_Tr3.txt")[-20:]
        self.LeftTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Left_train_Tr4.txt")[-20:]
        
        self.RightTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Right_train_Tr1.txt")[-20:]
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Right_train_Tr2.txt")[-20:]
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Right_train_Tr3.txt")[-20:]
        self.RightTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Right_train_Tr4.txt")[-20:]
        
        self.DispTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Dense_Disp_train_Tr1.txt")[-20:]
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Dense_Disp_train_Tr2.txt")[-20:]
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Dense_Disp_train_Tr3.txt")[-20:]
        self.DispTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Dense_Disp_train_Tr4.txt")[-20:]
        
        # MASKS
        self.MasqTilN=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Masq_train_Tr1.txt")[-20:]
        self.MasqTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Masq_train_Tr2.txt")[-20:]
        self.MasqTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Masq_train_Tr3.txt")[-20:]
        self.MasqTilN+=read_list_from_file('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO'+"/Masq_train_Tr4.txt")[-20:]

        #print(self.LeftTilN)
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        #idx=idx%len(self.LeftTilN)
        leftimgname=self.LeftTilN[idx]
        rightimgname=self.RightTilN[idx]
        dispGT=self.DispTilN[idx]
        masqGT=self.MasqTilN[idx]
        imMasqNOCC=tff.imread(masqGT)/255.0
        imDisp=tff.imread(dispGT)
        Masqdef=imDisp!=-999
        #x_upl=secrets.randbelow(self.init_size-self.patch_size)
        #y_upl=secrets.randbelow(self.init_size-self.patch_size)
        x_upl,y_upl=128,128
        Masqdef=Masqdef[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        MASQNOCC=imMasqNOCC[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images 
        grayimL=np.array(Image.fromarray(tff.imread(leftimgname)).convert('L'))/255.0
        grayimR=np.array(Image.fromarray(tff.imread(rightimgname)).convert('L'))/255.0
        # Color images to Gray 
        # Save into tensors
        x0=torch.from_numpy(grayimL[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]).unsqueeze(0).float()
        x1=torch.from_numpy(grayimR[y_upl:y_upl+self.patch_size,:]).unsqueeze(0).float() # take whole Right image
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqdef=torch.from_numpy(Masqdef).float()
        masqnoc=torch.from_numpy(MASQNOCC).float()
        return x0,x1,dispnoc,masqdef,masqnoc,x_upl
#########################################################################################################################
class StereoTrAerialDatasetOnline(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=768
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        self.LeftTilN=read_list_from_file(self.images+"/left_train_clean.txt")
        self.RightTilN=read_list_from_file(self.images+"/right_train_clean.txt")
        self.DispTilN=read_list_from_file(self.images+"/disp_train_clean.txt")
        self.MasqTilN=read_list_from_file(self.images+"/masq_train_clean.txt")
        """self.LeftTilN=[os.path.join(self.images,el) for el in self.LeftTilN]
        self.RightTilN=[os.path.join(self.images,el) for el in self.RightTilN]
        self.DispTilN=[os.path.join(self.images,el) for el in self.DispTilN]"""
        self.LeftTilN=self.LeftTilN[0:2023]
        self.RightTilN=self.RightTilN[0:2023]
        self.DispTilN=self.DispTilN[0:2023]
        self.MasqTilN=self.MasqTilN[0:2023]
        # Load all data into memory to fit cluster requirements 
        self.IML=[]
        self.IMR=[]
        self.DISP=[]
        self.DISPMASQ=[]
        for id_ in range(len(self.LeftTilN)):
            self.IML.append(tff.imread(self.LeftTilN[id_]))
            self.IMR.append(tff.imread(self.RightTilN[id_]))
            self.DISP.append(tff.imread(self.DispTilN[id_])*(-1.0))
            self.DISPMASQ.append(tff.imread(self.MasqTilN[id_])/255.0)
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        imDisp=self.DISP[idx]
        imMasq=self.DISPMASQ[idx]
        NotOCC=imDisp*imMasq
        NotOCC=NotOCC[NotOCC!=0.0]
        D_mu=int(np.mean(NotOCC))
        #print("Moyenne de disparite ",D_mu)
        # Randomly pick a region from the left image and compute the corresponding offset in the right image 
        x_upl=D_mu+secrets.randbelow(self.init_size-self.patch_size-D_mu)
        y_upl=secrets.randbelow(self.init_size-self.patch_size)
        imMasq=imMasq[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images
        grayimL=self.IML[idx]
        grayimR=self.IMR[idx]
        # Color images to Gray 
        # Save into tensors
        x0=torch.from_numpy(grayimL[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]).unsqueeze(0).float()
        x1=torch.from_numpy(grayimR[y_upl:y_upl+self.patch_size,:]).unsqueeze(0).float() # take whole Right image
        # NORMALIZE TILES LOCALLY 
        x0=(x0-x0.mean()).div(x0.std()+1e-12)
        x1=(x1-x1.mean()).div(x1.std()+1e-12)
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqnoc=torch.from_numpy(imMasq).to(torch.int16)
        return x0,x1,dispnoc,masqnoc,x_upl
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
class StereoValAerialDatasetOnline(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=768
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        self.LeftTilN=read_list_from_file(self.images+"/left_test_clean.txt")
        self.RightTilN=read_list_from_file(self.images+"/right_test_clean.txt")
        self.DispTilN=read_list_from_file(self.images+"/disp_test_clean.txt")
        self.MasqTilN=read_list_from_file(self.images+"/masq_test_clean.txt")
        """self.LeftTilN=[os.path.join(self.images,el) for el in self.LeftTilN]
        self.RightTilN=[os.path.join(self.images,el) for el in self.RightTilN]
        self.DispTilN=[os.path.join(self.images,el) for el in self.DispTilN]"""
        self.LeftTilN=self.LeftTilN[0:600]
        self.RightTilN=self.RightTilN[0:600]
        self.DispTilN=self.DispTilN[0:600]
        self.MasqTilN=self.MasqTilN[0:600]
        #print(self.LeftTilN)
        # Load all data into memory to fit cluster requirements 
        self.IML=[]
        self.IMR=[]
        self.DISP=[]
        self.DISPMASQ=[]
        for id_ in range(len(self.LeftTilN)):
            self.IML.append(tff.imread(self.LeftTilN[id_]))
            self.IMR.append(tff.imread(self.RightTilN[id_]))
            self.DISP.append(tff.imread(self.DispTilN[id_])*(-1.0))
            self.DISPMASQ.append(tff.imread(self.MasqTilN[id_])/255.0)
        
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        imDisp=self.DISP[idx]
        imMasq=self.DISPMASQ[idx]
        NotOCC=imDisp*imMasq
        NotOCC=NotOCC[NotOCC!=0.0]
        D_mu=int(np.mean(NotOCC))
        #print("Moyenne de disparite ",D_mu)
        # Randomly pick a region from the left image and compute the corresponding offset in the right image 
        x_upl=D_mu+secrets.randbelow(self.init_size-self.patch_size-D_mu)
        y_upl=secrets.randbelow(self.init_size-self.patch_size)
        imMasq=imMasq[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images
        grayimL=self.IML[idx]
        grayimR=self.IMR[idx]
        # Color images to Gray 
        # Save into tensors
        x0=torch.from_numpy(grayimL[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]).unsqueeze(0).float()
        x1=torch.from_numpy(grayimR[y_upl:y_upl+self.patch_size,:]).unsqueeze(0).float() # take whole Right image
        # NORMALIZE TILES LOCALLY 
        x0=(x0-x0.mean()).div(x0.std()+1e-12)
        x1=(x1-x1.mean()).div(x1.std()+1e-12)
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqnoc=torch.from_numpy(imMasq).to(torch.int16)
        return x0,x1,dispnoc,masqnoc,x_upl

#####################DISRTRIBUTED DATA PARALLEL DATASET #####################



class StereoTrAerialDatasetOnlineDDP(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=768
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        self.LeftTilN=read_list_from_file(self.images+"/l_1_left_train.txt")
        self.RightTilN=read_list_from_file(self.images+"/l_1_right_train.txt")
        self.DispTilN=read_list_from_file(self.images+"/l_1_disp_train.txt")
        self.MasqTilN=read_list_from_file(self.images+"/l_1_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_2_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_2_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_2_disp_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_2_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_31_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_31_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_31_disp_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_31_masq_train.txt")
        
        self.LeftTilN+=read_list_from_file(self.images+"/l_32_left_train.txt")
        self.RightTilN+=read_list_from_file(self.images+"/l_32_right_train.txt")
        self.DispTilN+=read_list_from_file(self.images+"/l_32_disp_train.txt")
        self.MasqTilN+=read_list_from_file(self.images+"/l_32_masq_train.txt")
        # Load all data into memory to fit cluster requirements 
        self.IML=[]
        self.IMR=[]
        self.DISP=[]
        self.DISPMASQ=[]
        for id_ in range(len(self.LeftTilN)):
            self.IML.append(tff.imread(self.LeftTilN[id_]))
            self.IMR.append(tff.imread(self.RightTilN[id_]))
            self.DISP.append(tff.imread(self.DispTilN[id_])*(-1.0))
            self.DISPMASQ.append(tff.imread(self.MasqTilN[id_])/255.0)
    def __len__(self):
        return len(self.LeftTilN)*10
    def __getitem__(self, idx):
        idx=idx%len(self.LeftTilN) 
        imDisp=self.DISP[idx]
        imMasq=self.DISPMASQ[idx]
        NotOCC=imDisp*imMasq
        NotOCC=NotOCC[NotOCC!=0.0]
        D_mu=int(np.mean(NotOCC))
        #print("Moyenne de disparite ",D_mu)
        # Randomly pick a region from the left image and compute the corresponding offset in the right image 
        x_upl=D_mu+secrets.randbelow(self.init_size-self.patch_size-D_mu)
        y_upl=secrets.randbelow(self.init_size-self.patch_size)
        imMasq=imMasq[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images
        grayimL=self.IML[idx]
        grayimR=self.IMR[idx]
        # Color images to Gray 
        # Save into tensors
        x0=torch.from_numpy(grayimL[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]).unsqueeze(0).float()
        x1=torch.from_numpy(grayimR[y_upl:y_upl+self.patch_size,:]).unsqueeze(0).float() # take whole Right image
        # NORMALIZE TILES LOCALLY 
        x0=(x0-x0.mean()).div(x0.std()+1e-12)
        x1=(x1-x1.mean()).div(x1.std()+1e-12)
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqnoc=torch.from_numpy(imMasq).to(torch.int16)
        return x0,x1,dispnoc,masqnoc,x_upl
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
class StereoValAerialDatasetOnlineDDP(Dataset):
    def __init__(self,images,mean,std,nameDataset):
        self.images=images
        self.meanG=mean
        self.stdG=std
        self.init_size=1024
        self.patch_size=768
        self.ND=nameDataset # DATASET NAME in 'ENSHEDE', 'EUROSDR', 'VAHINGEN'
        #self.LeftTilN,self.RightTilN, self.DispTilN=LoadImages(self.images)
        self.LeftTilN=read_list_from_file(self.images+"/all_left_test.txt")
        self.RightTilN=read_list_from_file(self.images+"/all_right_test.txt")
        self.DispTilN=read_list_from_file(self.images+"/all_disp_test.txt")
        self.MasqTilN=read_list_from_file(self.images+"/all_masq_test.txt")
        """self.LeftTilN=[os.path.join(self.images,el) for el in self.LeftTilN]
        self.RightTilN=[os.path.join(self.images,el) for el in self.RightTilN]
        self.DispTilN=[os.path.join(self.images,el) for el in self.DispTilN]"""
        self.LeftTilN=self.LeftTilN[0:1000]
        self.RightTilN=self.RightTilN[0:1000]
        self.DispTilN=self.DispTilN[0:1000]
        self.MasqTilN=self.MasqTilN[0:1000]
        #print(self.LeftTilN)
        # Load all data into memory to fit cluster requirements 
        self.IML=[]
        self.IMR=[]
        self.DISP=[]
        self.DISPMASQ=[]
        for id_ in range(len(self.LeftTilN)):
            self.IML.append(tff.imread(self.LeftTilN[id_]))
            self.IMR.append(tff.imread(self.RightTilN[id_]))
            self.DISP.append(tff.imread(self.DispTilN[id_])*(-1.0))
            self.DISPMASQ.append(tff.imread(self.MasqTilN[id_])/255.0)
        
    def __len__(self):
        return len(self.LeftTilN)
    def __getitem__(self, idx):
        imDisp=self.DISP[idx]
        imMasq=self.DISPMASQ[idx]
        NotOCC=imDisp*imMasq
        NotOCC=NotOCC[NotOCC!=0.0]
        D_mu=int(np.mean(NotOCC))
        #print("Moyenne de disparite ",D_mu)
        # Randomly pick a region from the left image and compute the corresponding offset in the right image 
        x_upl=D_mu+secrets.randbelow(self.init_size-self.patch_size-D_mu)
        y_upl=secrets.randbelow(self.init_size-self.patch_size)
        imMasq=imMasq[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        imDisp=imDisp[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]
        # Compute Mean of Disparity
        # Read images
        grayimL=self.IML[idx]
        grayimR=self.IMR[idx]
        # Color images to Gray 
        # Save into tensors
        x0=torch.from_numpy(grayimL[y_upl:y_upl+self.patch_size,x_upl:x_upl+self.patch_size]).unsqueeze(0).float()
        x1=torch.from_numpy(grayimR[y_upl:y_upl+self.patch_size,:]).unsqueeze(0).float() # take whole Right image
        # NORMALIZE TILES LOCALLY 
        x0=(x0-x0.mean()).div(x0.std()+1e-12)
        x1=(x1-x1.mean()).div(x1.std()+1e-12)
        # NORMALIZE GLOBALLY 
        x0=(x0-self.meanG).div(self.stdG+1e-12)
        x1=(x1-self.meanG).div(self.stdG+1e-12)
        dispnoc=torch.from_numpy(imDisp)
        masqnoc=torch.from_numpy(imMasq).to(torch.int16)
        return x0,x1,dispnoc,masqnoc,x_upl



if __name__=="__main__":
    #Val_dataset_Sat=StereoValSatDatasetDenseN('/home/ad/alichem/scratch/stereo_test/Data/RVL_SatStereo',0.1684881506278208,0.10536729988392025,'RVL_SatStereo',1)
    Train_dataset=StereoTrPatchAerialDatasetDenseN('/tmp/DUBLIN_DENSE/docker_image_names',0.434583236,0.1948717255,'dublin')
    x0,x1,x1n=Train_dataset.__getitem__(0)
    print(x0.shape,x1.shape,x1n.shape)
    # store positions
    """Write2File(lefts,"./all_clean_left_test.txt")
    Write2File(rights,"./all_clean_right_test.txt")
    Write2File(disps,"./all_clean_disp_test.txt")
    Write2File(Masqs,"./all_clean_masq_test.txt")"""
    #Tr_RVL_Dataset=StereoTrSatDatasetDenseN('/home/ad/alichem/scratch/stereo_test/Data/RVL_SatStereo',0.0964368,0.0587658,'RVL_SatStereo',4,MIN_SAT=0.0,MAX_SAT=15368.874)
    #DFC_Tr=StereoTrDFCDatasetDenseNocc('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO',0.5178164,0.20149627,'DFC19')
    #Tr_dataset_Aerial4Sat=StereoTrAerial4SatDatasetDenseN('/home/ad/alichem/scratch/stereo_test/Data/DUBLIN_DENSE',0.4357159999,0.1951853861,'Aerial4Sat')
    #tr_whu=StereoTrWHUDatasetDenseN('/home/ad/alichem/scratch/stereo_test/Data/PairwiseSTEREO',0.21019486,0.0745348,'WHU',1)
    #for id in range(len(tr_whu.LeftTilN)):
    #x0,x1,x1n=Train_dataset.__getitem__(0) #x0,x1,dispnoc,masqdef,masqnocc,X_off
        #print(x0.shape, x1.shape,masqdef.shape,masqnocc.shape,X_off)
    """import tifffile as tff
    tff.imwrite('./LeftTrain.tif',x0.detach().squeeze().numpy())
    tff.imwrite('./RightPatch.tif',x1.detach().squeeze().numpy())  
    tff.imwrite('./RightPatchN.tif',x1n.detach().squeeze().numpy())    """
