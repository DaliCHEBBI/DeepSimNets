from collections import OrderedDict
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from pytorch_lightning import LightningModule
import gc
from models.model import DecisionNetwork

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

class UNetMS(nn.Module):
    def __init__(self, in_channels=1, init_features=32):
        super(UNetMS, self).__init__()
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
        return output_feature,dec2,dec3,dec4

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

class UNETWithDecisionNetwork_LM_MulScaleBCE(LightningModule):
    def __init__(self,Inplanes,true1=1,false1=2,false2=8,NANS=-999.0):
        super(UNETWithDecisionNetwork_LM_MulScaleBCE, self).__init__()
        #self.device=torch.device("cuda:0")
        self.true1=true1
        self.false1=false1
        self.false2=false2
        self.nans=NANS
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.inplanes = Inplanes
        self.learning_rate=0.0005
        self.nbsteps=0
        self.save_hyperparameters()
        #self.example_input_array=torch.zeros((2,1,1024,1024),dtype=torch.float32)
        self.feature=UNetMS(init_features=self.inplanes)
        self.decisionNet=DecisionNetwork(2*64)
    def training_step(self,batch,batch_idx):
        x0,x1,dispnoc0=batch
        Mask0=(dispnoc0!=self.nans).float().to(self.device)  # NAN=-999.0
        dispnoc0[dispnoc0==self.nans]=0.0 # Set Nans to 0.0
        # Forward
        FeatsL,FeatsL2,FeatsL4,FeatsL8=self.feature(x0) # MulScale Features
        FeatsR,FeatsR2,FeatsR4,FeatsR8=self.feature(x1) # MulScale Features
        Offset_pos=(-2*self.true1) * torch.rand(dispnoc0.size(),device=self.device) + self.true1 #[-true1,true1]
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size(),device=self.device) + self.false2)
        RandSens=torch.rand(dispnoc0.size(),device=self.device)
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).to(self.device)
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0+Offset_pos
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1],device=self.device)
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        Offp=Index_X-D_pos.round()
        Offn=Index_X-D_neg.round()
        # Clean Indexes so there is no overhead
        MaskOffPositive=((Offp>=0)*(Offp<dispnoc0.size()[-1])).float().to(self.device)
        MaskOffNegative=((Offn>=0)*(Offn<dispnoc0.size()[-1])).float().to(self.device)
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave training_loss
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        Offp2=Offp.resize_(FeatsR2.size())
        Offp4=Offp.resize_(FeatsR4.size())
        Offp8=Offp.resize_(FeatsR8.size())
        FeatsR_plus2=torch.gather(FeatsR2,-1,Offp2)
        FeatsR_plus4=torch.gather(FeatsR4,-1,Offp4)
        FeatsR_plus8=torch.gather(FeatsR8,-1,Offp8)
        # Test gather operator
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        Offn2=Offn.resize_(FeatsR2.size())
        Offn4=Offn.resize_(FeatsR4.size())
        Offn8=Offn.resize_(FeatsR8.size())
        FeatsR_minus2=torch.gather(FeatsR2,-1,Offn2)
        FeatsR_minus4=torch.gather(FeatsR4,-1,Offn4)
        FeatsR_minus8=torch.gather(FeatsR8,-1,Offn8)
        # Mask Global = Mask des batiments + Mask des offsets bien definis
        MaskGlob=Mask0*MaskOffPositive*MaskOffNegative
        MaskGlob2=F.interpolate(MaskGlob,FeatsR2.size()[2:],mode='nearest')
        MaskGlob4=F.interpolate(MaskGlob,FeatsR4.size()[2:],mode='nearest')
        MaskGlob8=F.interpolate(MaskGlob,FeatsR8.size()[2:],mode='nearest')

        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        # Scale 2
        ref_pos2=self.decisionNet(torch.cat((FeatsL2,FeatsR_plus2),1))
        ref_neg2=self.decisionNet(torch.cat((FeatsL2,FeatsR_minus2),1))
        # Scale 4
        ref_pos4=self.decisionNet(torch.cat((FeatsL4,FeatsR_plus4),1))
        ref_neg4=self.decisionNet(torch.cat((FeatsL4,FeatsR_minus4),1))
        # Scale 8
        ref_pos8=self.decisionNet(torch.cat((FeatsL8,FeatsR_plus8),1))
        ref_neg8=self.decisionNet(torch.cat((FeatsL8,FeatsR_minus8),1))

        sample = torch.cat((ref_pos, ref_neg), dim=0)
        sample2 = torch.cat((ref_pos2, ref_neg2), dim=0)
        sample4 = torch.cat((ref_pos4, ref_neg4), dim=0)
        sample8 = torch.cat((ref_pos8, ref_neg8), dim=0)

        target = torch.cat((torch.ones(x0.size()), torch.zeros(x0.size())), dim=0)
        target2 = torch.cat((torch.ones(x0.size(0),x0.size(1),x0.size(2)//2,x0.size(3)//2), torch.zeros(x0.size(0),x0.size(1),x0.size(2)//2,x0.size(3)//2)), dim=0)
        target4 = torch.cat((torch.ones(x0.size(0),x0.size(1),x0.size(2)//4,x0.size(3)//4), torch.zeros(x0.size(0),x0.size(1),x0.size(2)//4,x0.size(3)//4)), dim=0)
        target8 = torch.cat((torch.ones(x0.size(0),x0.size(1),x0.size(2)//8,x0.size(3)//8), torch.zeros(x0.size(0),x0.size(1),x0.size(2)//8,x0.size(3)//8)), dim=0)

        training_loss0=self.criterion(sample, target.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob,MaskGlob),0)
        training_loss2=self.criterion(sample2, target2.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob2,MaskGlob2),0)
        training_loss4=self.criterion(sample4, target4.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob4,MaskGlob4),0)
        training_loss8=self.criterion(sample8, target8.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob8,MaskGlob8),0)
        """if (torch.any(torch.isnan(training_loss))):
            raise Exception("nan values encountered in training loss ")"""
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        training_loss0=training_loss0.sum().div(2*MaskGlob.count_nonzero()+1e-20)
        training_loss2=training_loss2.sum().div(2*MaskGlob2.count_nonzero()+1e-20)
        training_loss4=training_loss4.sum().div(2*MaskGlob4.count_nonzero()+1e-20)
        training_loss8=training_loss8.sum().div(2*MaskGlob8.count_nonzero()+1e-20)
        gc.collect()
        self.log("training_loss",training_loss0+0.5*training_loss2+0.25*training_loss4+0.125*training_loss8, on_epoch=True)
        return training_loss0+0.5*training_loss2+0.25*training_loss4+0.125*training_loss8

    def validation_loss(self,batch,batch_idx):
        x0,x1,dispnoc0=batch
        Mask0=(dispnoc0!=self.nans).float().to(self.device)  # NAN=-999.0
        dispnoc0[dispnoc0==self.nans]=0.0 # Set Nans to 0.0
        # Forward
        FeatsL,FeatsL2,FeatsL4,FeatsL8=self.feature(x0) # MulScale Features
        FeatsR,FeatsR2,FeatsR4,FeatsR8=self.feature(x1) # MulScale Features
        Offset_pos=(-2*self.true1) * torch.rand(dispnoc0.size(),device=self.device) + self.true1 #[-true1,true1]
        Offset_neg=((self.false1 - self.false2) * torch.rand(dispnoc0.size(),device=self.device) + self.false2)
        RandSens=torch.rand(dispnoc0.size(),device=self.device)
        RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).to(self.device)
        Offset_neg=Offset_neg*RandSens
        #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
        D_pos=dispnoc0+Offset_pos
        D_neg=dispnoc0+Offset_neg
        Index_X=torch.arange(0,dispnoc0.size()[-1],device=self.device)
        Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
        Offp=Index_X-D_pos.round()
        Offn=Index_X-D_neg.round()
        # Clean Indexes so there is no overhead
        MaskOffPositive=((Offp>=0)*(Offp<dispnoc0.size()[-1])).float().to(self.device)
        MaskOffNegative=((Offn>=0)*(Offn<dispnoc0.size()[-1])).float().to(self.device)
        # Cleaned Offp and Offn
        Offp=(Offp*MaskOffPositive).to(torch.int64)
        Offn=(Offn*MaskOffNegative).to(torch.int64)
        # Need to repeat interleave training_loss
        Offp=Offp.repeat_interleave(FeatsR.size()[1],1)
        Offn=Offn.repeat_interleave(FeatsR.size()[1],1)
        # Get Examples positive and negative
        FeatsR_plus=torch.gather(FeatsR,-1,Offp)
        Offp2=Offp.resize_(FeatsR2.size())
        Offp4=Offp.resize_(FeatsR4.size())
        Offp8=Offp.resize_(FeatsR8.size())
        FeatsR_plus2=torch.gather(FeatsR2,-1,Offp2)
        FeatsR_plus4=torch.gather(FeatsR4,-1,Offp4)
        FeatsR_plus8=torch.gather(FeatsR8,-1,Offp8)
        # Test gather operator
        FeatsR_minus=torch.gather(FeatsR,-1,Offn)
        Offn2=Offn.resize_(FeatsR2.size())
        Offn4=Offn.resize_(FeatsR4.size())
        Offn8=Offn.resize_(FeatsR8.size())
        FeatsR_minus2=torch.gather(FeatsR2,-1,Offn2)
        FeatsR_minus4=torch.gather(FeatsR4,-1,Offn4)
        FeatsR_minus8=torch.gather(FeatsR8,-1,Offn8)
        # Mask Global = Mask des batiments + Mask des offsets bien definis
        MaskGlob=Mask0*MaskOffPositive*MaskOffNegative
        MaskGlob2=F.interpolate(MaskGlob,FeatsR2.size()[2:],mode='nearest')
        MaskGlob4=F.interpolate(MaskGlob,FeatsR4.size()[2:],mode='nearest')
        MaskGlob8=F.interpolate(MaskGlob,FeatsR8.size()[2:],mode='nearest')

        ref_pos=self.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
        ref_neg=self.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
        # Scale 2
        ref_pos2=self.decisionNet(torch.cat((FeatsL2,FeatsR_plus2),1))
        ref_neg2=self.decisionNet(torch.cat((FeatsL2,FeatsR_minus2),1))
        # Scale 4
        ref_pos4=self.decisionNet(torch.cat((FeatsL4,FeatsR_plus4),1))
        ref_neg4=self.decisionNet(torch.cat((FeatsL4,FeatsR_minus4),1))
        # Scale 8
        ref_pos8=self.decisionNet(torch.cat((FeatsL8,FeatsR_plus8),1))
        ref_neg8=self.decisionNet(torch.cat((FeatsL8,FeatsR_minus8),1))

        sample = torch.cat((ref_pos, ref_neg), dim=0)
        sample2 = torch.cat((ref_pos2, ref_neg2), dim=0)
        sample4 = torch.cat((ref_pos4, ref_neg4), dim=0)
        sample8 = torch.cat((ref_pos8, ref_neg8), dim=0)

        target = torch.cat((torch.ones(x0.size()), torch.zeros(x0.size())), dim=0)
        target2 = torch.cat((torch.ones(x0.size(0),x0.size(1),x0.size(2)//2,x0.size(3)//2), torch.zeros(x0.size(0),x0.size(1),x0.size(2)//2,x0.size(3)//2)), dim=0)
        target4 = torch.cat((torch.ones(x0.size(0),x0.size(1),x0.size(2)//4,x0.size(3)//4), torch.zeros(x0.size(0),x0.size(1),x0.size(2)//4,x0.size(3)//4)), dim=0)
        target8 = torch.cat((torch.ones(x0.size(0),x0.size(1),x0.size(2)//8,x0.size(3)//8), torch.zeros(x0.size(0),x0.size(1),x0.size(2)//8,x0.size(3)//8)), dim=0)

        validation_loss0=self.criterion(sample, target.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob,MaskGlob),0)
        validation_loss2=self.criterion(sample2, target2.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob2,MaskGlob2),0)
        validation_loss4=self.criterion(sample4, target4.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob4,MaskGlob4),0)
        validation_loss8=self.criterion(sample8, target8.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob8,MaskGlob8),0)
        validation_loss0=validation_loss0.sum().div(2*MaskGlob.count_nonzero()+1e-20)
        validation_loss2=validation_loss2.sum().div(2*MaskGlob2.count_nonzero()+1e-20)
        validation_loss4=validation_loss4.sum().div(2*MaskGlob4.count_nonzero()+1e-20)
        validation_loss8=validation_loss8.sum().div(2*MaskGlob8.count_nonzero()+1e-20)
        gc.collect()
        self.log("val_loss",validation_loss0+0.5*validation_loss2+0.25*validation_loss4+0.125*validation_loss8, on_epoch=True)
        return validation_loss0+0.5*validation_loss2+0.25*validation_loss4+0.125*validation_loss8

    def forward(self,x):
         f_all=self.feature(x)
         #print(f_all[0].shape)
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return out
    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15],gamma=0.9)
        return [optimizer],[scheduler]
        """return {
           "optimizer": optimizer,
           "lr_scheduler": {
              "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.9,patience=10,min_lr=1e-8),
              "interval": "step",
              "monitor": "training_loss",
              "frequency": 1
         },
        }"""

if __name__=="__main__":
     model=UNetWithDecisionNetwork(32,128)
     x=torch.rand((2,1,11,140))
     out=model(x)
     print(out.shape)
     
