from collections import OrderedDict
from random import randint
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
import numpy as np
from pytorch_lightning import LightningModule
import gc
from models.model import DecisionNetwork5D, DecisionNetwork


def balanced_binary_cross_entropy(pred, gt,nogt, pos_w=2.0, neg_w=1.0):
    masked_nogt=nogt.sub(gt)
    # flatten vectors
    pred = pred.view(-1)
    gt = gt.view(-1)
    masked_nogt=masked_nogt.view(-1)
    # select postive/nevative samples
    pos_ind = gt.nonzero().squeeze(-1)
    neg_ind = masked_nogt.nonzero().squeeze(-1)

    # compute weighted loss
    pos_loss = pos_w*F.binary_cross_entropy(pred[pos_ind], gt[pos_ind], reduction='none')
    neg_loss = neg_w*F.binary_cross_entropy(pred[neg_ind], masked_nogt[neg_ind], reduction='none')
    g_loss=pos_loss + neg_loss
    g_loss=g_loss.div(nogt.count_nonzero()+1e-12)
    return g_loss

def mse(coords, coords_gt, prob_gt):

    # flatten vectors
    coords = coords.view(-1, 2)
    coords_gt = coords_gt.view(-1, 2)
    prob_gt = prob_gt.view(-1)

    # select positive samples
    pos_ind = prob_gt.nonzero().squeeze(-1)
    pos_coords = coords[pos_ind, :]
    pos_coords_gt = coords_gt[pos_ind, :]

    return F.mse_loss(pos_coords, pos_coords_gt)


def generate_pointcloud(CUBE, ply_file):
    """
    Generate a colored ply from  the dense cube
    """
    points = []
    for zz in range(CUBE.size()[0]):
        for yy in range(CUBE.size()[1]):
            for xx in range(CUBE.size()[2]):
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

class UNet(nn.Module):
    def __init__(self, in_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = nn.Sequential(
                        nn.Conv2d(
                            features*2,
                            features*2,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(features*2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            features*2,
                            features*2,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        )
                     )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec_inter = torch.cat((dec1, enc1), dim=1)
        output_feature = self.decoder1(dec_inter)
        return output_feature
    def _block(self, in_channels, features):
        return nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(features),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            features,
                            features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(features),
                        nn.ReLU(inplace=True)
                       )

class UNetInference(nn.Module):
    def __init__(self, in_channels=1, init_features=32):
        super(UNetInference, self).__init__()
        features = init_features
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = nn.Sequential(
                        nn.Conv2d(
                            features*2,
                            features*2,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(features*2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            features*2,
                            features*2,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        )
                     )

    def forward(self, x):
        if x.size()[-2] % 16 != 0:
            times = x.size()[-2]//16   
            top_pad = (times+1)*16 - x.size()[-2]
        else:
            top_pad = 0
        if x.size()[-1] % 16 != 0:
            times = x.size()[-1]//16
            right_pad = (times+1)*16-x.size()[-1] 
        else:
            right_pad = 0    

        x = F.pad(x,(0,right_pad, top_pad,0))

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec_inter = torch.cat((dec1, enc1), dim=1)
        output_feature = self.decoder1(dec_inter)

        if top_pad !=0 and right_pad != 0:
            out = output_feature[:,:,top_pad:,:-right_pad]
        elif top_pad ==0 and right_pad != 0:
            out = output_feature[:,:,:,:-right_pad]
        elif top_pad !=0 and right_pad == 0:
            out = output_feature[:,:,top_pad:,:]
        else:
            out = output_feature
        return out

    def _block(self, in_channels, features):
        return nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(features),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(
                            features,
                            features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(features),
                        nn.ReLU(inplace=True)
                       )

class UNetWithDecisionNetwork(nn.Module):
     def __init__(self,infeats,outfeats):
         super(UNetWithDecisionNetwork,self).__init__()
         self.in_features=infeats
         self.out_features=outfeats
         self.feature=UNetInference(init_features=self.in_features)
         self.decisionNet=DecisionNetwork(self.out_features)
     def forward(self,x):
         f_all=self.feature(x)
         #print(f_all.shape)
         # shape 2,64,w,h
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return torch.sigmoid(out)

class UNETWithDecisionNetwork_LM5D(LightningModule):
    def __init__(self,Inplanes,true1=1,false1=2,false2=8,NANS=-999.0):
        super(UNETWithDecisionNetwork_LM5D, self).__init__()
        #self.device=torch.device("cuda:0")
        self.true1=true1
        self.false1=false1
        self.false2=false2
        self.EPSILON=0.08
        self.nans=NANS
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.residual_criterion=nn.MSELoss(reduction='none') # add a weighting parameter around 0.1 
        self.inplanes = Inplanes
        self.learning_rate=0.0005
        self.nbsteps=0
        self.save_hyperparameters()
        #self.example_input_array=torch.zeros((2,1,1024,1024),dtype=torch.float32)
        self.feature=UNet(init_features=self.inplanes)
        self.decisionNet=DecisionNetwork5D(2*64)
    def training_step(self,batch, batch_idx):
        x0,x1,dispnoc0=batch
        Mask0=(dispnoc0!=self.nans).float().to(self.device)  # NAN=-999.0 # shape (N,1,h,w)
        #print( "Mask shape and content ",Mask0.shape)
        dispnoc0[dispnoc0==self.nans]=0.0 # Set Nans to 0.0 # shape (N,1,h,w)
        # Forward 
        N=x0.size(0)
        Feats=self.feature(torch.cat((x0,x1),dim=0))
        FeatsL=Feats[0:N]    # shape (N,64,h,w)
        FeatsR=Feats[N:2*N]  # shape (N,64,h,w)
        # Construire les nappes englobantes
        Index_X=torch.arange(0,dispnoc0.size()[-1],device=self.device)
        #print("Index values  ",Index_X.shape)
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        # Index_X of shape (N,1,H,W) give x coordinate 
        #print("Index values  ",Index_X.shape)
        # Repete FeatsL and FeatsR 2*false2 times 
        FeatsL=FeatsL.unsqueeze(1).repeat_interleave(2*self.false2,1) # N, 2*false2, 64,h,w
        #print("Features LEFT Repearted ",FeatsL.shape)
        torch._assert(torch.equal(FeatsL[0,0,:,:,:],(FeatsL[0,15,:,:,:])),"issue repeate interleave")
        # Generate positive sample tensor of shape equal to reference anchor tensor
        FeatsQuery=torch.empty(FeatsL.shape).to(self.device)
        MaskQuery=torch.empty((N,2*self.false2,1,dispnoc0.size(-2),dispnoc0.size(-1))).to(self.device)
        for i in np.arange(-self.false2,self.false2):
            Offset=Index_X-dispnoc0+i
            MaskOffNegative=((Offset>=0)*Mask0*(Offset<dispnoc0.size()[-1])).float().to(self.device)
            Offset=(Offset*MaskOffNegative).to(torch.int64)
            Offset=Offset.repeat_interleave(FeatsR.size()[1],1)
            FeatRSample=torch.gather(FeatsR,-1,Offset)
            #print("RIGHT SAMPLE SHAPE ",FeatRSample.shape)
            # Fill 5 Dimensional Tensor
            MaskQuery[:,i+self.false2,:,:,:]=MaskOffNegative
            FeatsQuery[:,i+self.false2,:,:,:]=FeatRSample
        # You have 2 5 Dimensional tensors to pass into the decison DecisionNetwork
        OutSimil=self.decisionNet(torch.cat((FeatsL,FeatsQuery),dim=2)).squeeze()# Dimension (N, 2f, 1 ,H, W) 
        OutSimil=OutSimil*MaskQuery.float().squeeze()
        # MASK_DISPNOC DEFINTION 
        ref_pos=OutSimil[:,self.false2,:,:] # Centerd on Gt disparities
        # search for the most annpying elements in the structure of the cube 
        NORM_DIFF=torch.abs(OutSimil.sigmoid()-ref_pos.sigmoid().unsqueeze(1).repeat_interleave(2*self.false2,1))
        ref_neg=torch.amin(torch.where(NORM_DIFF-self.EPSILON>0,OutSimil,torch.ones(NORM_DIFF.shape,device=self.device)),1)
        ref_neg=ref_neg*Mask0.squeeze()
        # INDICE DES PLUS FAIBLES MAIS SUPERIEURS A EPSILON ==> CES CEUX QUI VONT DEFINIR LESEXEMPLES NEGATIFS 
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones((N,x0.size(-2),x0.size(-1))), torch.zeros((N,x0.size(-2),x0.size(-1)))), dim=0)
        training_loss=self.criterion(sample+1e-20, target.to(device=self.device, dtype=torch.float))*torch.cat((Mask0,Mask0),0).squeeze()
        training_loss=training_loss.sum().div(2*Mask0.count_nonzero()+1e-12)
        gc.collect()
        self.log("training_loss",training_loss, on_epoch=True)
        return training_loss

    def validation_step(self,batch, batch_idx):
        x0,x1,dispnoc0=batch
        Mask0=(dispnoc0!=self.nans).float().to(self.device)  # NAN=-999.0 # shape (N,1,h,w)
        dispnoc0[dispnoc0==self.nans]=0.0 # Set Nans to 0.0 # shape (N,1,h,w)
        # Forward 
        N=x0.size(0)
        Feats=self.feature(torch.cat((x0,x1),dim=0))
        FeatsL=Feats[0:N]    # shape (N,64,h,w)
        FeatsR=Feats[N:2*N]  # shape (N,64,h,w)
        # Construire les nappes englobantes
        Index_X=torch.arange(0,dispnoc0.size()[-1],device=self.device)
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        # Index_X of shape (N,1,H,W) give x coordinate 
        # Repete FeatsL and FeatsR 2*false2 times 
        FeatsL=FeatsL.unsqueeze(1).repeat_interleave(2*self.false2,1) # N, 2*false2, 64,h,w
        # Generate positive sample tensor of shape equal to reference anchor tensor
        FeatsQuery=torch.empty(FeatsL.shape).to(self.device)
        MaskQuery=torch.empty((N,2*self.false2,1,dispnoc0.size(-2),dispnoc0.size(-1))).to(self.device)
        for i in np.arange(-self.false2,self.false2):
            Offset=Index_X-dispnoc0+i
            MaskOffNegative=((Offset>=0)*Mask0*(Offset<dispnoc0.size()[-1])).float().to(self.device)
            Offset=(Offset*MaskOffNegative).to(torch.int64)
            Offset=Offset.repeat_interleave(FeatsR.size()[1],1)
            FeatRSample=torch.gather(FeatsR,-1,Offset)
            # Fill 5 Dimensional Tensor
            MaskQuery[:,i+self.false2,:,:,:]=MaskOffNegative
            FeatsQuery[:,i+self.false2,:,:,:]=FeatRSample
        # You have 2 5 Dimensional tensors to pass into the decison DecisionNetwork
        OutSimil=self.decisionNet(torch.cat((FeatsL,FeatsQuery),dim=2)).squeeze() # Dimension (N, 2f, 1 ,H, W) 
        OutSimil=OutSimil*MaskQuery.float().squeeze()
        # MASK_DISPNOC DEFINTION 
        ref_pos=OutSimil[:,self.false2,:,:] # Centerd on Gt disparities
        # search for the most annpying elements in the structure of the cude 
        NORM_DIFF=torch.abs(OutSimil.sigmoid()-ref_pos.sigmoid().unsqueeze(1).repeat_interleave(2*self.false2,1))
        ref_neg=torch.amin(torch.where(NORM_DIFF-self.EPSILON>0,OutSimil,torch.ones(NORM_DIFF.shape,device=self.device)),1)
        # INDICE DES PLUS FAIBLES MAIS SUPERIEURS A EPSILON ==> CES CEUX QUI VONT DEFINIR LESEXEMPLES NEGATIFS
        ref_neg=ref_neg*Mask0.squeeze()
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones((N,x0.size(-2),x0.size(-1))), torch.zeros((N,x0.size(-2),x0.size(-1)))), dim=0)
        val_loss=self.criterion(sample+1e-20, target.to(device=self.device, dtype=torch.float))*torch.cat((Mask0,Mask0),0).squeeze()
        val_loss=val_loss.sum().div(2*Mask0.count_nonzero()+1e-12)
        gc.collect()
        self.log("val_loss",val_loss, on_epoch=True)
        """if (MaskGlob.count_nonzero().item()):
             Tplus=torch.masked_select(F.cosine_similarity(FeatsL, FeatsR_plus).unsqueeze(1),MaskGlob.bool())
             Tmoins=torch.masked_select(F.cosine_similarity(FeatsL, FeatsR_minus).unsqueeze(1),MaskGlob.bool())
             if len(torch.nonzero(Tplus.sub(Tplus.mean())))>1:# and torch.not_equal(Tmoins.sub(Tmoins.mean()),0)
                  #print(Tmoins.nelement(), Tplus.nelement())
                  self.logger.experiment.add_histogram('distribution positive',Tplus,global_step=self.nbsteps)
                  self.logger.experiment.add_histogram('distribution negative',Tmoins,global_step=self.nbsteps)
        self.nbsteps=self.nbsteps+1"""
        return val_loss
    def forward(self,x):
         f_all=self.feature(x)
         #print(f_all[0].shape)
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return out

    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15],gamma=0.9)
        # ReduceOnPlateau scheduler 
        """reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=10,
            min_lr=1e-6,
            verbose=True
        )
        sch_val = {
            'scheduler': reduce_lr_on_plateau,
            'monitor': "val_loss",
            'frequency': 1,
        }"""
        return [optimizer],[scheduler]


class UNETWithDecisionNetwork_Dense_LM(LightningModule):
    def __init__(self,Inplanes,true1=0,false1=2,false2=8,NANS=-999.0):
        super(UNETWithDecisionNetwork_Dense_LM, self).__init__()
        #self.device=torch.device("cuda:0")
        self.true1=true1
        self.false1=false1
        self.false2=false2
        self.nans=NANS
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.inplanes = Inplanes
        self.learning_rate=0.001
        self.nbsteps=0
        self.save_hyperparameters()
        #self.example_input_array=torch.zeros((2,1,1024,1024),dtype=torch.float32)
        self.feature=UNet(init_features=self.inplanes)
        self.decisionNet=DecisionNetwork(2*64)
    """def training_step(self,batch,batch_idx):
        x0,x1,dispnoc0,Mask0,x_offset=batch 
        dispnoc0=dispnoc0.unsqueeze(1)
        Mask0=Mask0.unsqueeze(1).mul(dispnoc0!=0.0)
        dispnoc0=dispnoc0*Mask0.float() # Set Nans to 0.0
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size()).cuda() + self.false2)
        RandSens=torch.rand(dispnoc0.size()).cuda()
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).cuda()
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1]).cuda()
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        # ADD OFFSET
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float().cuda()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float().cuda()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlob=Mask0*MaskOffPositive*MaskOffNegative
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size()), torch.zeros(x0.size())), dim=0)
        training_loss=self.criterion(sample+1e-20, target.float().cuda())*torch.cat((MaskGlob,MaskGlob),0)
        #if (torch.any(torch.isnan(training_loss))):
            #raise Exception("nan values encountered in training loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        training_loss=training_loss.sum().div(2*MaskGlob.count_nonzero()+1e-12)
        gc.collect()
        self.log("training_loss",training_loss, on_epoch=True)
        return training_loss"""

    def training_step(self,batch,batch_idx):
        x0,x1,dispnoc0,Mask0,x_offset=batch
        # ADD DIM 1
        dispnoc0=dispnoc0.unsqueeze(1)
        # DISPNOC0 ++> DENSE DISPARITY Map 
        MaskDef=dispnoc0!=0.0  # ==> defined (occluded + non occluded + non available) disparity values 
        # MASK 0 ==> non available + non occluded areas ==> we search for occluded areas  
        Mask0=Mask0.unsqueeze(1)
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size()).cuda() + self.false2)
        RandSens=torch.rand(dispnoc0.size()).cuda()
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).cuda()
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1]).cuda()
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float().cuda()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float().cuda()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlobP=MaskDef*MaskOffPositive
        MaskGlobN=MaskDef*MaskOffNegative
        # Get Mask Global
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size()).cuda()-OCCLUDED.float(), torch.zeros(x0.size()).cuda()), dim=0).float()
        training_loss=self.criterion(sample+1e-20, target)*torch.cat((MaskGlobP,MaskGlobN),0)
        #if (torch.any(torch.isnan(validation_loss))):
            #raise Exception("nan values encountered in validation loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        training_loss=training_loss.sum().div(MaskGlobP.count_nonzero()+MaskGlobN.count_nonzero()+1e-12)
        gc.collect()
        self.log("training_loss",training_loss, on_epoch=True)
        return training_loss

    """def validation_step(self,batch,batch_idx):
        x0,x1,dispnoc0,Mask0,x_offset=batch
        # ADD DIM 1
        dispnoc0=dispnoc0.unsqueeze(1)
        Mask0=Mask0.unsqueeze(1).mul(dispnoc0!=0.0)
        print("initial shapes ", x0.shape,x1.shape,dispnoc0.shape,Mask0.shape)
        dispnoc0=dispnoc0*Mask0.float() # Set Nans to 0.0
        print("Disparity shape ",dispnoc0.shape)
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size()).cuda() + self.false2)
        RandSens=torch.rand(dispnoc0.size()).cuda()
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).cuda()
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1]).cuda()
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        print("Index SHAPE ", Index_X.shape)
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        print("offsets shapes ", Offp.shape, Offn.shape)
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float().cuda()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float().cuda()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlob=Mask0*MaskOffPositive*MaskOffNegative
        MaskTest=MaskGlob.cpu().squeeze().detach().numpy()
        #MaskTest=MaskTest.astype(np.byte)
        import tifffile as tf
        tf.imwrite('./MASQTEST.tif',MaskTest)
        # Get Mask Global
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size()), torch.zeros(x0.size())), dim=0)
        validation_loss=self.criterion(sample+1e-20, target.float().cuda())*torch.cat((MaskGlob,MaskGlob),0)
        #if (torch.any(torch.isnan(validation_loss))):
            #raise Exception("nan values encountered in validation loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        validation_loss=validation_loss.sum().div(2*MaskGlob.count_nonzero()+1e-12)
        gc.collect()
        self.log("val_loss",validation_loss, on_epoch=True)
        if (MaskGlob.count_nonzero().item()):
             Tplus=torch.masked_select(F.cosine_similarity(FeatsL, FeatsR_plus).unsqueeze(1),MaskGlob.bool())
             Tmoins=torch.masked_select(F.cosine_similarity(FeatsL, FeatsR_minus).unsqueeze(1),MaskGlob.bool())
             if len(torch.nonzero(Tplus.sub(Tplus.mean())))>1:# and torch.not_equal(Tmoins.sub(Tmoins.mean()),0)
                  #print(Tmoins.nelement(), Tplus.nelement())
                  self.logger.experiment.add_histogram('distribution positive',Tplus,global_step=self.nbsteps)
                  self.logger.experiment.add_histogram('distribution negative',Tmoins,global_step=self.nbsteps)
        self.nbsteps=self.nbsteps+1
        return validation_loss"""

    def validation_step(self,batch,batch_idx):
        x0,x1,dispnoc0,Mask0,x_offset=batch
        # ADD DIM 1
        dispnoc0=dispnoc0.unsqueeze(1)
        # DISPNOC0 ++> DENSE DISPARITY Map 
        MaskDef=dispnoc0!=0.0  # ==> defined (occluded + non occluded + non available) disparity values 
        # MASK 0 ==> non available + non occluded areas ==> we search for occluded areas  
        Mask0=Mask0.unsqueeze(1)
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size()).cuda() + self.false2)
        RandSens=torch.rand(dispnoc0.size()).cuda()
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).cuda()
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1]).cuda()
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float().cuda()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float().cuda()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlobP=MaskDef*MaskOffPositive
        MaskGlobN=MaskDef*MaskOffNegative
        # Get Mask Global
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size()).cuda()-OCCLUDED.float(), torch.zeros(x0.size()).cuda()), dim=0).float()
        validation_loss=self.criterion(sample+1e-20, target)*torch.cat((MaskGlobP,MaskGlobN),0)
        #if (torch.any(torch.isnan(validation_loss))):
            #raise Exception("nan values encountered in validation loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        validation_loss=validation_loss.sum().div(MaskGlobP.count_nonzero()+MaskGlobN.count_nonzero()+1e-12)
        gc.collect()
        self.log("val_loss",validation_loss, on_epoch=True)
        """if (MaskGlob.count_nonzero().item()):
             Tplus=torch.masked_select(F.cosine_similarity(FeatsL, FeatsR_plus).unsqueeze(1),MaskGlob.bool())
             Tmoins=torch.masked_select(F.cosine_similarity(FeatsL, FeatsR_minus).unsqueeze(1),MaskGlob.bool())
             if len(torch.nonzero(Tplus.sub(Tplus.mean())))>1:# and torch.not_equal(Tmoins.sub(Tmoins.mean()),0)
                  #print(Tmoins.nelement(), Tplus.nelement())
                  self.logger.experiment.add_histogram('distribution positive',Tplus,global_step=self.nbsteps)
                  self.logger.experiment.add_histogram('distribution negative',Tmoins,global_step=self.nbsteps)
        self.nbsteps=self.nbsteps+1"""
        return validation_loss
    def forward(self,x):
         f_all=self.feature(x)
         #print(f_all[0].shape)
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return out
    def configure_optimizers(self):
        """EarlyFeaturesParams=[]
        EarlyFeaturesParams.extend(self.feature.encoder1.parameters())
        EarlyFeaturesParams.extend(self.feature.encoder2.parameters())

        MiddleFeauresParams=[]
        MiddleFeauresParams.extend(self.feature.encoder3.parameters())
        MiddleFeauresParams.extend(self.feature.encoder4.parameters())

        BottleneckAndGeneratorsParams=[p for p in self.feature.parameters() if p not in set(EarlyFeaturesParams) and p not in set(MiddleFeauresParams)]
        #print(BottleneckAndGeneratorsParams)
        #BottleneckAndGeneratorsParams=[p for p in BottleneckAndGeneratorsParams if p not in set(MiddleFeauresParams)]
        param_grps=[
            {'params':EarlyFeaturesParams,'lr':self.learning_rate**2},
            {'params':MiddleFeauresParams,'lr':self.learning_rate**2},
            {'params':BottleneckAndGeneratorsParams,'lr':self.learning_rate/2},
            {'params':self.decisionNet.parameters(),'lr':self.learning_rate},
        ]"""
        optimizer=torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40,50],gamma=0.7)
        return [optimizer],[scheduler]

class UNETWithDecisionNetwork_Dense_LM_N_2(LightningModule):
    def __init__(self,Inplanes,true1=0,false1=2,false2=8,NANS=-999.0):
        super(UNETWithDecisionNetwork_Dense_LM_N_2, self).__init__()
        #self.device=torch.device("cuda:0")
        self.true1=true1
        self.false1=false1
        self.false2=false2
        self.nans=NANS
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.inplanes = Inplanes
        self.learning_rate=0.001
        self.nbsteps=0
        self.save_hyperparameters()
        #self.example_input_array=torch.zeros((2,1,1024,1024),dtype=torch.float32)
        self.feature=UNet(init_features=self.inplanes)
        self.decisionNet=DecisionNetwork(2*64)
    def training_step(self,batch,batch_idx):
        x0,x1,dispnoc0,MaskDef,Mask0,x_offset=batch
        #print("SAMPLE  ==> ",x0.shape,x1.shape,dispnoc0.shape,MaskDef.shape,Mask0.shape,x_offset.shape)
        # ADD DIM 1
        dispnoc0=dispnoc0.unsqueeze(1)
        # DISPNOC0 ++> DENSE DISPARITY Map 
        MaskDef=MaskDef.unsqueeze(1)  # ==> defined (occluded + non occluded + non available) disparity values 
        # MASK 0 ==> non available + non occluded areas ==> we search for occluded areas  
        Mask0=Mask0.unsqueeze(1)
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        #print('OCCLUDEDE SHAPE ',torch.max(OCCLUDED))
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size()).cuda() + self.false2)
        RandSens=torch.rand(dispnoc0.size()).cuda()
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).cuda()
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1]).cuda()
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float().cuda()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float().cuda()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlobP=MaskDef*MaskOffPositive
        MaskGlobN=MaskDef*MaskOffNegative
        # Get Mask Global
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size()).cuda()-OCCLUDED.float(), torch.zeros(x0.size()).cuda()), dim=0)
        training_loss=self.criterion(sample+1e-20, target)*torch.cat((MaskGlobP,MaskGlobN),0)
        #if (torch.any(torch.isnan(validation_loss))):
            #raise Exception("nan values encountered in validation loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        training_loss=training_loss.sum().div(MaskGlobP.count_nonzero()+MaskGlobN.count_nonzero()+1e-12)
        gc.collect()
        self.log("training_loss",training_loss, on_epoch=True)
        return training_loss

    def validation_step(self,batch,batch_idx):
        x0,x1,dispnoc0,MaskDef,Mask0,x_offset=batch
        # ADD DIM 1
        dispnoc0=dispnoc0.unsqueeze(1)
        # DISPNOC0 ++> DENSE DISPARITY Map 
        MaskDef=MaskDef.unsqueeze(1)  # ==> defined (occluded + non occluded + non available) disparity values 
        # MASK 0 ==> non available + non occluded areas ==> we search for occluded areas  
        Mask0=Mask0.unsqueeze(1)
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        #print('OCCLUDEDE SHAPE ',OCCLUDED.shape)
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size()).cuda() + self.false2)
        RandSens=torch.rand(dispnoc0.size()).cuda()
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).cuda()
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1]).cuda()
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float().cuda()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float().cuda()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlobP=MaskDef*MaskOffPositive
        MaskGlobN=MaskDef*MaskOffNegative
        # Get Mask Global
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size()).cuda()-OCCLUDED.float(), torch.zeros(x0.size()).cuda()), dim=0)
        validation_loss=self.criterion(sample+1e-12, target)*torch.cat((MaskGlobP,MaskGlobN),0)
        #if (torch.any(torch.isnan(validation_loss))):
            #raise Exception("nan values encountered in validation loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        validation_loss=validation_loss.sum().div(MaskGlobP.count_nonzero()+MaskGlobN.count_nonzero()+1e-12)
        gc.collect()
        self.log("val_loss",validation_loss, on_epoch=True)
        return validation_loss
    def forward(self,x):
         f_all=self.feature(x)
         #print(f_all[0].shape)
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return out
    def configure_optimizers(self):
        """EarlyFeaturesParams=[]
        EarlyFeaturesParams.extend(self.feature.encoder1.parameters())
        EarlyFeaturesParams.extend(self.feature.encoder2.parameters())

        MiddleFeauresParams=[]
        MiddleFeauresParams.extend(self.feature.encoder3.parameters())
        MiddleFeauresParams.extend(self.feature.encoder4.parameters())

        BottleneckAndGeneratorsParams=[p for p in self.feature.parameters() if p not in set(EarlyFeaturesParams) and p not in set(MiddleFeauresParams)]
        #print(BottleneckAndGeneratorsParams)
        #BottleneckAndGeneratorsParams=[p for p in BottleneckAndGeneratorsParams if p not in set(MiddleFeauresParams)]
        param_grps=[
            {'params':EarlyFeaturesParams,'lr':self.learning_rate*0.2},
            {'params':MiddleFeauresParams,'lr':self.learning_rate*0.4},
            {'params':BottleneckAndGeneratorsParams,'lr':self.learning_rate*0.6},
            {'params':self.decisionNet.parameters(),'lr':self.learning_rate},
        ]"""
        #optimizer=torch.optim.SGD(self.parameters(),lr=self.learning_rate,momentum=0.9)
        optimizer=torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        """scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=1910,
                                          cycle_mult=1.0,
                                          max_lr=0.01,
                                          min_lr=0.0001,
                                          warmup_steps=200,
                                          gamma=0.95)"""
        """scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                              T_max = 10, # Maximum number of iterations.
                             eta_min = 1e-4)""" # Minimum learning rate.
        #scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = 1, T_mult = 1, eta_min = 1e-4)
        """scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
            }"""
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40,50,60,70,80,90],gamma=0.7)
        return [optimizer],[scheduler]

import torch.nn as nn
class FastMcCnnInference(nn.Module):
    """
    Define the mc_cnn fast neural network
    """
    def __init__(self):
        super().__init__()
        self.in_channels = 1
        self.num_conv_feature_maps = 64
        self.conv_kernel_size = 3

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels, out_channels=self.num_conv_feature_maps, kernel_size=self.conv_kernel_size,padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_conv_feature_maps,
                out_channels=self.num_conv_feature_maps,
                kernel_size=self.conv_kernel_size,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_conv_feature_maps,
                out_channels=self.num_conv_feature_maps,
                kernel_size=self.conv_kernel_size,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_conv_feature_maps,
                out_channels=self.num_conv_feature_maps,
                kernel_size=self.conv_kernel_size,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.num_conv_feature_maps,
                out_channels=self.num_conv_feature_maps,
                kernel_size=self.conv_kernel_size,
                padding=1,

            ),
        )

    # pylint: disable=arguments-differ
    # pylint: disable=no-else-return
    def forward(self, sample):
        with torch.no_grad():
            # Because input shape of nn.Conv2d is (Batch_size, Channel, H, W), we add 2 dimensions
            features = self.conv_blocks(sample)
            return F.normalize(features, p=2.0, dim=1)

class MCCNNWithDecisionNetwork(nn.Module):
     def __init__(self,infeats,outfeats):
         super(MCCNNWithDecisionNetwork,self).__init__()
         self.in_features=infeats
         self.out_features=outfeats
         self.feature=FastMcCnnInference()
         self.decisionNet=DecisionNetwork(self.out_features)
     def forward(self,x):
         f_all=self.feature(x)
         #print(f_all.shape)
         # shape 2,64,w,h
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return torch.sigmoid(out)

class MCCNWithDecisionNetwork_Dense_LM_N_2(LightningModule):
    def __init__(self,Inplanes,true1=0,false1=2,false2=8,NANS=-999.0):
        super(MCCNWithDecisionNetwork_Dense_LM_N_2, self).__init__()
        #self.device=torch.device("cuda:0")
        self.true1=true1
        self.false1=false1
        self.false2=false2
        self.nans=NANS
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.inplanes = Inplanes
        self.learning_rate=0.001
        self.nbsteps=0
        self.save_hyperparameters()
        #self.example_input_array=torch.zeros((2,1,1024,1024),dtype=torch.float32)
        self.feature=FastMcCnnInference()
        self.decisionNet=DecisionNetwork(2*64)
    def training_step(self,batch,batch_idx):
        x0,x1,dispnoc0,MaskDef,Mask0,x_offset=batch
        #print("SAMPLE  ==> ",x0.shape,x1.shape,dispnoc0.shape,MaskDef.shape,Mask0.shape,x_offset.shape)
        # ADD DIM 1
        dispnoc0=dispnoc0.unsqueeze(1)
        # DISPNOC0 ++> DENSE DISPARITY Map 
        MaskDef=MaskDef.unsqueeze(1)  # ==> defined (occluded + non occluded + non available) disparity values 
        # MASK 0 ==> non available + non occluded areas ==> we search for occluded areas  
        Mask0=Mask0.unsqueeze(1)
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        #print('OCCLUDEDE SHAPE ',torch.max(OCCLUDED))
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size()).cuda() + self.false2)
        RandSens=torch.rand(dispnoc0.size()).cuda()
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).cuda()
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1]).cuda()
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float().cuda()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float().cuda()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlobP=MaskDef*MaskOffPositive
        MaskGlobN=MaskDef*MaskOffNegative
        # Get Mask Global
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size()).cuda()-OCCLUDED.float(), torch.zeros(x0.size()).cuda()), dim=0)
        training_loss=self.criterion(sample+1e-20, target)*torch.cat((MaskGlobP,MaskGlobN),0)
        #if (torch.any(torch.isnan(validation_loss))):
            #raise Exception("nan values encountered in validation loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        training_loss=training_loss.sum().div(MaskGlobP.count_nonzero()+MaskGlobN.count_nonzero()+1e-12)
        gc.collect()
        self.log("training_loss",training_loss, on_epoch=True)
        return training_loss

    def validation_step(self,batch,batch_idx):
        x0,x1,dispnoc0,MaskDef,Mask0,x_offset=batch
        # ADD DIM 1
        dispnoc0=dispnoc0.unsqueeze(1)
        # DISPNOC0 ++> DENSE DISPARITY Map 
        MaskDef=MaskDef.unsqueeze(1)  # ==> defined (occluded + non occluded + non available) disparity values 
        # MASK 0 ==> non available + non occluded areas ==> we search for occluded areas  
        Mask0=Mask0.unsqueeze(1)
        OCCLUDED=torch.logical_and(MaskDef,torch.logical_not(Mask0))
        #print('OCCLUDEDE SHAPE ',OCCLUDED.shape)
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size()).cuda() + self.false2)
        RandSens=torch.rand(dispnoc0.size()).cuda()
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).cuda()
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1]).cuda()
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        #print("INDEX SHAPE   ",Index_X.shape )
        # ADD OFFSET
        #print(x_offset.unsqueeze(0).T.shape)
        Index_X=Index_X.add(x_offset.unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
        Offp=Index_X-D_pos.round()  
        Offn=Index_X-D_neg.round() 
        # Clean Indexes so there is no overhead 
        MaskOffPositive=((Offp>=0)*(Offp<FeatsR.size()[-1])).float().cuda()
        MaskOffNegative=((Offn>=0)*(Offn<FeatsR.size()[-1])).float().cuda()
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave 
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative 
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        # Test gather operator 
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        # Mask Global = Mask des batiments + Mask des offsets bien definis 
        MaskGlobP=MaskDef*MaskOffPositive
        MaskGlobN=MaskDef*MaskOffNegative
        # Get Mask Global
        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        sample = torch.cat((ref_pos, ref_neg), dim=0)
        target = torch.cat((torch.ones(x0.size()).cuda()-OCCLUDED.float(), torch.zeros(x0.size()).cuda()), dim=0)
        validation_loss=self.criterion(sample+1e-12, target)*torch.cat((MaskGlobP,MaskGlobN),0)
        #if (torch.any(torch.isnan(validation_loss))):
            #raise Exception("nan values encountered in validation loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        validation_loss=validation_loss.sum().div(MaskGlobP.count_nonzero()+MaskGlobN.count_nonzero()+1e-12)
        gc.collect()
        self.log("val_loss",validation_loss, on_epoch=True)
        return validation_loss
    def forward(self,x):
         f_all=self.feature(x)
         #print(f_all[0].shape)
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return out
    def configure_optimizers(self):
        """EarlyFeaturesParams=[]
        EarlyFeaturesParams.extend(self.feature.encoder1.parameters())
        EarlyFeaturesParams.extend(self.feature.encoder2.parameters())

        MiddleFeauresParams=[]
        MiddleFeauresParams.extend(self.feature.encoder3.parameters())
        MiddleFeauresParams.extend(self.feature.encoder4.parameters())

        BottleneckAndGeneratorsParams=[p for p in self.feature.parameters() if p not in set(EarlyFeaturesParams) and p not in set(MiddleFeauresParams)]
        #print(BottleneckAndGeneratorsParams)
        #BottleneckAndGeneratorsParams=[p for p in BottleneckAndGeneratorsParams if p not in set(MiddleFeauresParams)]
        param_grps=[
            {'params':EarlyFeaturesParams,'lr':self.learning_rate*0.2},
            {'params':MiddleFeauresParams,'lr':self.learning_rate*0.4},
            {'params':BottleneckAndGeneratorsParams,'lr':self.learning_rate*0.6},
            {'params':self.decisionNet.parameters(),'lr':self.learning_rate},
        ]"""
        #optimizer=torch.optim.SGD(self.parameters(),lr=self.learning_rate,momentum=0.9)
        optimizer=torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        """scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=1910,
                                          cycle_mult=1.0,
                                          max_lr=0.01,
                                          min_lr=0.0001,
                                          warmup_steps=200,
                                          gamma=0.95)"""
        """scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                              T_max = 10, # Maximum number of iterations.
                             eta_min = 1e-4)""" # Minimum learning rate.
        #scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = 1, T_mult = 1, eta_min = 1e-4)
        """scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
            }"""
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40,50,60,70,80,90],gamma=0.7)
        return [optimizer],[scheduler]


    
     
