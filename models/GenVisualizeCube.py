# This Python file uses the following encoding: utf-8
from functools import update_wrapper
import sys
import os
import torch
import numpy as np
from PIL import Image
import tifffile as tff
#from Matcher3D import Matcher3D
from joblib import Parallel, delayed
import rasterio
import argparse
from ROF import compute_grad_Im,Gradx, Grad


# Generer les nappes
def ExtractNappes_Combined_DeLaunay(Disparity_sparse, DisparityDelaunay,MasqDelaunay, BUFF= 3, raster_meta=None):
    Mask_Holes_Interpolated=DisparityDelaunay!=0.0
    print(~Mask_Holes_Interpolated)
    Interpolated_meaningful_px=DisparityDelaunay[Mask_Holes_Interpolated]
    ZInf=np.zeros(DisparityDelaunay.shape,dtype=np.float32)
    ZSup=np.zeros(DisparityDelaunay.shape,dtype=np.float32)
    # FOR DEFINED VALUES USE FIXED BUFFER
    ZInf[Mask_Holes_Interpolated]=Interpolated_meaningful_px - BUFF
    ZSup[Mask_Holes_Interpolated]=Interpolated_meaningful_px + BUFF
    # ANOTHER LAST ITERATION : ENFORCE DISPARITY IN NAPPES EQUAL TO GT
    Disparity_sparse_gt=Disparity_sparse[Disparity_sparse!=0.0]
    ZInf[Disparity_sparse!=0.0]=Disparity_sparse_gt - BUFF
    ZSup[Disparity_sparse!=0.0]=Disparity_sparse_gt + BUFF
    #MasqWhereNotDefined=Mask_Holes_Interpolated
    # use mask of delaunay to define the search space when IDW does not work
    ZInf[~Mask_Holes_Interpolated]=DisparityDelaunay[~Mask_Holes_Interpolated] - BUFF
    ZSup[~Mask_Holes_Interpolated]=DisparityDelaunay[~Mask_Holes_Interpolated] + BUFF
    # Get ZSup and ZInf
    return ZSup, ZInf


def GenPseudoCube(FeatsL,FeatsR, Network, Loc, NappeSup, NappeInf, RADIUS):
    '''
    Computes a pseudo cube of size 2xRADIUS, 2xRADIUS, HAUTEUR

    '''
    DeviceGpu=torch.device("cuda:0")
    Device=torch.device("cpu")

    H,W=NappeSup.shape

    yy,xx=Loc

    assert(yy-RADIUS >0 and yy+RADIUS<H and xx-RADIUS>0 and xx+RADIUS<W)

    # Get sub layers
    NappeSupLoc=NappeSup[yy-RADIUS:yy+RADIUS, xx-RADIUS:xx+RADIUS]
    NappeInfLoc=NappeInf[yy-RADIUS:yy+RADIUS, xx-RADIUS:xx+RADIUS]
    NappeInfLoc=np.around(NappeInfLoc).astype(int)
    NappeSupLoc=np.around(NappeSupLoc).astype(int)
    # Initialize Cube
    lower_lim=np.min(NappeInfLoc)
    upper_lim=np.max(NappeSupLoc)
    HGT= upper_lim-lower_lim
    CUBE=torch.ones((HGT,2*RADIUS,2*RADIUS),device=Device).mul(0.5)

    # IT IS POSSIBLE TO DISCRETIZE THE COST VOLUME TO SUB PIXEL LEVES ==> ???
    x_0=torch.arange(xx-RADIUS,xx+RADIUS,dtype=torch.int64, device=Device)
    X_field=x_0.expand(NappeSupLoc.shape).unsqueeze(0).repeat_interleave(HGT,0) # Shape (HGT,2xRADIUS,2xRADIUS)
    print("XFIELD  ",X_field.shape)

    # Compute the Disparity Field
    D_field=torch.zeros(CUBE.size(),dtype=torch.int64,device=Device)
    for yy in range(2*RADIUS):
        for xx in range(2*RADIUS):
            # Compute D_field
            DispDefSlice=torch.zeros(HGT)
            #print(len(DispDefSlice))
            #print(NappeInfLoc[yy,xx]-lower_lim,NappeSupLoc[yy,xx]-lower_lim, lower_lim, upper_lim)
            #print(NappeInfLoc[yy,xx],NappeSupLoc[yy,xx])
            DispDefSlice[NappeInfLoc[yy,xx]-lower_lim:NappeSupLoc[yy,xx]-lower_lim]=torch.arange(NappeInfLoc[yy,xx],NappeSupLoc[yy,xx])
            #print(DispDefSlice[NappeInfLoc[yy,xx]-lower_lim:NappeSupLoc[yy,xx]-lower_lim].size())
            DispDefSlice[0:NappeInfLoc[yy,xx]-lower_lim]=torch.arange(lower_lim,NappeInfLoc[yy,xx])
            #print(torch.arange(NappeSupLoc[yy,xx]+1,upper_lim).size(), DispDefSlice[NappeSupLoc[yy,xx]-lower_lim:-1].size())
            DispDefSlice[NappeSupLoc[yy,xx]-lower_lim:]=torch.arange(NappeSupLoc[yy,xx],upper_lim)
            #print("DISP SLICE SHAPE ", DispDefSlice.shape)
            D_field[:,yy,xx]=DispDefSlice
    # Gather  Rights Features to compute the overal cost volume at once
    X_D=X_field-D_field # Shape (HGT, 2xRADIUS, 2xRADIUS)
    Masq_X_D=(X_D>=0) * (X_D<W)
    X_D=X_D*Masq_X_D.int()
    #FeatsRLoc=torch.empty((CUBE.size()[0],FeatsR.size[1],CUBE.size()[-2],CUBE.size()[-1],device=Device)
    for d in range(HGT):
        indexes=X_D[d,:,:].unsqueeze(0).repeat_interleave(FeatsR.size()[1],0)
        FeatsRLoc=torch.gather(FeatsR.squeeze(),-1,indexes)
        #print("Right slice shape ",FeatsRLoc.shape,FeatsL[0,:,yy-RADIUS:yy+RADIUS, xx-RADIUS:xx+RADIUS].shape)
        FeatsLR=torch.cat((FeatsL[0,:,yy-RADIUS:yy+RADIUS, xx-RADIUS:xx+RADIUS],FeatsRLoc),0).unsqueeze(0)
        print(" SHAPE OF CONCATENATION ",FeatsLR.shape)
        CUBE[d,:,:]=Network(FeatsLR).sigmoid()
    return CUBE*Masq_X_D.float(),NappeSupLoc,NappeInfLoc


def GenPseudoFastBatch(FeatsL,FeatsR, Network, Loc, NappeSup, NappeInf, RADIUS,IsLocRandom=True):
    '''
    Computes a set of sample 3D cost volumes of different disparity search spaces given a set
    of random locations inside tiles and a planar size of the pseudo cubes
    RADIUS: planar size : we can make it random later : for now Fixed to 15 ==> cube size: height x 30 x 30
    Loc: A set of Random Locations for each tile in the batch
    FeatsL: BS, 64, H, W
    FeatsR: BS, 64, H, W

    # loaded in batch
    # <<   For now non need because the buffer is fixed so we can do it online  >>
    NappeSup:BS, H, W
    NappeInf:BS, H, W
    '''
    # Random Locations
    nbloc=FeatsL.size()[0] # nbloc is the batch size for shape learning <<  it can be greater than the IMAGES BATCH SIZE >>
    H,W=FeatsL.size()[-2],FeatsL.size()[-1]
    if IsLocRandom:
        cols=torch.randint(RADIUS,W-RADIUS,nbloc)
        rows=torch.randint(RADIUS, H-RADIUS,nbloc)
    else: #just one random location applied to the whole batch
        cols=torch.randint(RADIUS,W-RADIUS,1).repeat((nbloc))
        rows=torch.randint(RADIUS, H-RADIUS,1).repeat((nbloc))
    # generate cubes
    # 1. remplir nappes  << pas tres urgent et encombrant >>
    NappeSupLoc=torch.empty(nbloc,2*RADIUS,2*RADIUS, dtype=torch.int64, decice=Device)
    NappeInfLoc=torch.empty(nbloc,2*RADIUS,2*RADIUS, dtype=torch.int64, decice=Device)
    Upper_lims=torch.empty((nbloc),device=Device)
    Lower_lims=torch.empty((nbloc),device=Device)
    for nap in range(len(cols)):
        Id_xx=torch.arange(cols[nap]-RADIUS,cols[nap]+RADIUS)
        Id_yy=torch.arange(rows[nap]-RADIUS,rows[nap]+RADIUS)
        NappeSupLoc[nap,:,:]=torch.round(NappeSup[nap,:,:].index_select(0,Id_yy).index_select(1,Id_xx)).int()
        Upper_lims[nap]=torch.max(NappeSupLoc[nap,:,:])
        NappeInfLoc[nap,:,:]=torch.round(NappeInf[nap,:,:].index_select(0,Id_yy).index_select(1,Id_xx)).int()
        Lower_lims[nap]=torch.min(NappeInfLoc[nap,:,:])
    # 2. Initialize different size cubes
    Max_de_max=torch.max(Upper_lims).item()
    Min_de_min=torch.min(Lower_lims).item()
    CUBES=torch.empty((Max_de_max-Min_de_min,2*RADIUS,2*RADIUS), dtype=torch.float, device=Device)
    GT_Disp_Encodings=torch.zeros((Max_de_max-Min_de_min,2*RADIUS,2*RADIUS), dtype=torch.float, device=Device)


    # pad cubes to have the same Height and disparity ground truthe encodings accordingly
    # training is done by binary cross entropy between the disparity Masks of  locations and the similarity values
    #  << Inside the zone that is defined by the lower and upper bounds >>
    # << later explore weighted distance to the surface as a training objective and pente max a ne pas depasser sur la surface >>








def DisplayCostVolume(CUBE):
    import plotly.graph_objects as go
    X,Y,Z=np.mgrid[0:CUBE.shape[-1]:50j, 0:CUBE.shape[-2]:50j,0:CUBE.shape[-3]:38j]
    CUBE=CUBE.transpose(0,1).transpose(1,2)
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=CUBE.cpu().detach().numpy().flatten(),
        isomin=0.1,
        isomax=1.0,
        opacity=0.1, # needs to be small to see through all surfaces
        surface_count=25, # needs to be a large number for good volume rendering
        ))
    with open('./costvolume.html', 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs=True))
    f.close()
    fig.show()

def generate_pointcloud(CUBE,NappeSup, NappeInf, ply_file):
    """
    Generate a colored ply from  the dense cube
    """
    NappeInf=np.around(NappeInf).astype(int)
    NappeSup=np.around(NappeSup).astype(int)
    # Initialize Cube
    lower_lim=np.min(NappeInf)
    upper_lim=np.max(NappeSup)
    points = []
    for zz in range(CUBE.size()[0]):
        for yy in range(CUBE.size()[1]):
            for xx in range(CUBE.size()[2]):
                #if zz>=NappeInf[yy,xx]-lower_lim and zz<NappeSup[yy,xx]-lower_lim:
                val=CUBE[zz,yy,xx]
                points.append("%f %f %f %f %f %f 0\n"%(xx,yy,zz,val,val,val))
    file = open(ply_file,"w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property float red
property float green
property float blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points)))
    file.close()

def main():
     UNET_MLP=torch.jit.load("./Models_18072022/Without_ATTENTION_RETRAINED_DUBLIN/UNET_DECISION_DUB.pt", \
                    map_location=torch.device("cpu"))
     # Read images
     DirImages="./Datasets/ENSCH/Enschede-stereo_echo/ENSCHDE_training/0021764_1_0021765_1"
     print(DirImages)
     leftimgname=DirImages+"/"+"DMTrain_Test_Mohamed-0021764_1_0021765_1_0006_Im1.tif"
     rightimgname=DirImages+"/"+"DMTrain_Test_Mohamed-0021764_1_0021765_1_0006_Im2.tif"
     dispGT=DirImages+"/"+"DMTrain_Test_Mohamed-0021764_1_0021765_1_0006_Pax1.tif"
     dispDelaunay=DirImages+"/"+"DensifyPx_DMTrain_Test_Mohamed-0021764_1_0021765_1_0006_Im1.tif"
     MasqDelaunay=DirImages+"/"+"DensifyMasq_DMTrain_Test_Mohamed-0021764_1_0021765_1_0006_Im1.tif"
     imL=tff.imread(leftimgname)
     imR=tff.imread(rightimgname)
     D_Delaunay=tff.imread(dispDelaunay)*(-1.0) # inversed in MicMac
     D_GT=tff.imread(dispGT)*(-1.0) # inversed in MicMac
     MasqDELAUNAY=tff.imread(MasqDelaunay)
     # Save into tensors
     x0=torch.from_numpy(np.array(imL)).unsqueeze(0).float()
     x1=torch.from_numpy(np.array(imR)).unsqueeze(0).float()
     # NORMALIZE TILES LOCALLY
     x0=(x0-x0.mean()).div(x0.std()+1e-12).unsqueeze(0)
     # FETAURES
     print(x0.shape)
     Fl=UNET_MLP.feature(x0)
     x1=(x1-x1.mean()).div(x1.std()+1e-12).unsqueeze(0)
     Fr=UNET_MLP.feature(x1)

     print("Features Left  ==> : ",Fl.shape)
     print("Features Right ==> : ",Fr.shape)

     # CALCULER LES NAPPES ENNGLOBANTES

     ZMAX,ZMIN=ExtractNappes_Combined_DeLaunay(D_GT,D_Delaunay,MasqDELAUNAY)

     # Compute PSEUDO CUBE @ LOCATION INSIDE THE IMAGE
     RADIUS=15
     CUBE,NapS,NapI=GenPseudoCube(Fl,Fr, UNET_MLP.decisionNet, (558,520), ZMAX, ZMIN, RADIUS)
     # Visualize CUBE to see if correlations are where they are meant to be
     generate_pointcloud(CUBE,NapS,NapI,"./cube.ply")
     CUBE=CUBE.detach().numpy()
     """for yy in np.arange(0,CUBE.shape[1],10):
         # Display every image @ levels of disparity
         tff.imwrite("./Cube_level_{}.tif".format(yy),CUBE[:,yy,:])"""
     # GET THE SIMILARITY BASED DISPARITY MAP
     Disparity=np.argmax(CUBE,axis=0)
     print(Disparity)
     Disparity=Disparity.astype(np.float32)
     # Filter disparity
     import cv2
     Disparity=cv2.bilateralFilter(Disparity-6, 7, 15, 15)
     tff.imwrite("./DISP_BRUTE.tif",Disparity)


     tff.imwrite("./DemaunaySlice.tif",D_Delaunay[558-RADIUS:558+RADIUS,520-RADIUS:520+RADIUS])
     #DisplayCostVolume(CUBE)
     #print( "CUBE ", CUBE)



if __name__ == "__main__":
    main()
