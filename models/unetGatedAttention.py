from collections import OrderedDict
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
import numpy as np
from pytorch_lightning import LightningModule
import gc
from models.model import DecisionNetwork
from models.networks_other import init_weights
from models.grid_attention_layer import GridAttentionBlock2D

class UnetDsv(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetDsv, self).__init__()
        self.dsv = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0)
    def forward(self, input):
        #return F.interpolate(self.dsv(input), size=outSz, mode='bilinear')
        return self.dsv(input)

class UnetGridGatingSignal2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1), is_batchnorm=True):
        super(UnetGridGatingSignal2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, (1,1), (0,0)),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, (1,1), (0,0)),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs
    
class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock2D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.gate_block_2 = GridAttentionBlock2D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv2d(in_size*2, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm2d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock2D') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)
        gate_2, attention_2 = self.gate_block_2(input, gating_signal)

        return self.combine_gates(torch.cat([gate_1, gate_2], 1)), torch.cat([attention_1, attention_2], 1)

class UNetGatedAttention(nn.Module):
    def __init__(self, in_channels=1, init_features=32):
        super(UNetGatedAttention, self).__init__()
        features = init_features
        self.nonlocal_mode='concatenation'
        self.attention_dsample=(2,2)
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16)
        self.gating = UnetGridGatingSignal2(features * 16, features * 16, kernel_size=(1, 1), is_batchnorm=True)
        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(in_size=features * 2, gate_size=features * 4, inter_size=features * 2,
                                                   nonlocal_mode=self.nonlocal_mode, sub_sample_factor= self.attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=features * 4, gate_size=features * 8, inter_size=features * 4,
                                                   nonlocal_mode=self.nonlocal_mode, sub_sample_factor= self.attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=features * 8, gate_size=features * 16, inter_size=features * 8,
                                                   nonlocal_mode=self.nonlocal_mode, sub_sample_factor= self.attention_dsample)
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
        # deep supervision
        self.dsv4 = UnetDsv(in_size=features*8, out_size=features//2)
        self.dsv3 = UnetDsv(in_size=features*4, out_size=features//2)
        self.dsv2 = UnetDsv(in_size=features*2, out_size=features//2)
        self.dsv1 = nn.Conv2d(in_channels=features, out_channels=features//2, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2)) #128xh/4
        enc4 = self.encoder4(self.pool3(enc3)) #256xh/8

        bottleneck = self.bottleneck(self.pool4(enc4)) #512xh/16
        
        # Added Gating and Attention heads 
        gating = self.gating(bottleneck)  #512xh/16
        g_enc4, att4 = self.attentionblock4(enc4, gating) # 256 x h/8
        dec4 = self.upconv4(bottleneck) # 256 xh/8
        dec4 = torch.cat((g_enc4, dec4), dim=1) # 512xh/8
        dec4 = self.decoder4(dec4) # 256xh/8
        
        g_enc3, att3 = self.attentionblock3(enc3, dec4) # 128 x h/4
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((g_enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3) # 128xh/4

        g_enc2, att2 = self.attentionblock2(enc2, dec3) # 64 x h/2
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((g_enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)# 64 x h/2
        
        dec1 = self.upconv1(dec2) # 32xh
        # Deep Supervision
        dsv4 = F.interpolate(self.dsv4(dec4), size=x.size()[2:], mode='bilinear')
        dsv3 = F.interpolate(self.dsv3(dec3), size=x.size()[2:], mode='bilinear')
        dsv2 = F.interpolate(self.dsv2(dec2), size=x.size()[2:], mode='bilinear')
        dsv1 = self.dsv1(dec1)
        final = torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1)
        output_feature = self.decoder1(final)  #64xh
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


class UNetInferenceGatedAttention(nn.Module):
    def __init__(self, in_channels=1, init_features=32):
        super(UNetInferenceGatedAttention, self).__init__()
        features = init_features
        self.nonlocal_mode='concatenation'
        self.attention_dsample=(2,2)
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16)
        self.gating = UnetGridGatingSignal2(features * 16, features * 16, kernel_size=(1, 1), is_batchnorm=True)
        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(in_size=features * 2, gate_size=features * 4, inter_size=features * 2,
                                                   nonlocal_mode=self.nonlocal_mode, sub_sample_factor= self.attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=features * 4, gate_size=features * 8, inter_size=features * 4,
                                                   nonlocal_mode=self.nonlocal_mode, sub_sample_factor= self.attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=features * 8, gate_size=features * 16, inter_size=features * 8,
                                                   nonlocal_mode=self.nonlocal_mode, sub_sample_factor= self.attention_dsample)
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
        # deep supervision
        self.dsv4 = UnetDsv(in_size=features*8, out_size=features//2)
        self.dsv3 = UnetDsv(in_size=features*4, out_size=features//2)
        self.dsv2 = UnetDsv(in_size=features*2, out_size=features//2)
        self.dsv1 = nn.Conv2d(in_channels=features, out_channels=features//2, kernel_size=1)

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
        enc3 = self.encoder3(self.pool2(enc2)) #128xh/4
        enc4 = self.encoder4(self.pool3(enc3)) #256xh/8

        bottleneck = self.bottleneck(self.pool4(enc4)) #512xh/16
        
        # Added Gating and Attention heads 
        gating = self.gating(bottleneck)  #512xh/16
        g_enc4, att4 = self.attentionblock4(enc4, gating) # 256 x h/8
        dec4 = self.upconv4(bottleneck) # 256 xh/8
        dec4 = torch.cat((g_enc4, dec4), dim=1) # 512xh/8
        dec4 = self.decoder4(dec4) # 256xh/8
        
        g_enc3, att3 = self.attentionblock3(enc3, dec4) # 128 x h/4
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((g_enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3) # 128xh/4

        g_enc2, att2 = self.attentionblock2(enc2, dec3) # 64 x h/2
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((g_enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)# 64 x h/2
        
        dec1 = self.upconv1(dec2) # 32xh
        # Deep Supervision
        dsv4 = F.interpolate(self.dsv4(dec4), size=x.size()[2:], mode='bilinear')
        dsv3 = F.interpolate(self.dsv3(dec3), size=x.size()[2:], mode='bilinear')
        dsv2 = F.interpolate(self.dsv2(dec2), size=x.size()[2:], mode='bilinear')
        dsv1 = self.dsv1(dec1)
        final = torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1)
        output_feature = self.decoder1(final)  #64xh
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

class UNetGatedAttentionWithDecisionNetwork(nn.Module):
     def __init__(self,infeats,outfeats):
         super(UNetGatedAttentionWithDecisionNetwork,self).__init__()
         self.in_features=infeats
         self.out_features=outfeats
         self.feature=UNetInferenceGatedAttention(init_features=self.in_features)
         self.decisionNet=DecisionNetwork(self.out_features)
     def forward(self,x):
         f_all=self.feature(x)
         #print(f_all.shape)
         # shape 2,64,w,h
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return torch.sigmoid(out)

class UNETGatedAttentionWithDecisionNetwork_LM(LightningModule):
    def __init__(self,Inplanes,true1=1,false1=2,false2=8,NANS=-999.0):
        super(UNETGatedAttentionWithDecisionNetwork_LM, self).__init__()
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
        self.feature=UNetGatedAttention(init_features=self.inplanes)
        self.decisionNet=DecisionNetwork(2*64)
    def training_step(self,batch,batch_idx):
        x0,x1,dispnoc0=batch
        Mask0=(dispnoc0!=self.nans).float().to(self.device)  # NAN=-999.0
        dispnoc0[dispnoc0==self.nans]=0.0 # Set Nans to 0.0
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
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
        training_loss=self.criterion(sample+1e-20, target.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob,MaskGlob),0)
        #if (torch.any(torch.isnan(training_loss))):
            #raise Exception("nan values encountered in training loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        training_loss=training_loss.sum().div(2*MaskGlob.count_nonzero()+1e-12)
        gc.collect()
        self.log("training_loss",training_loss, on_epoch=True)
        return training_loss

    def validation_step(self,batch,batch_idx):
        x0,x1,dispnoc0=batch
        Mask0=(dispnoc0!=self.nans).float().to(self.device)  # NAN=-999.0
        dispnoc0[dispnoc0==self.nans]=0.0 # Set Nans to 0.0
        # Forward
        FeatsL=self.feature(x0) 
        FeatsR=self.feature(x1)
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
        validation_loss=self.criterion(sample+1e-20, target.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob,MaskGlob),0)
        #if (torch.any(torch.isnan(validation_loss))):
            #raise Exception("nan values encountered in validation loss ")
        #training_loss=_add_Intra_Class_Variance(FeatsL,FeatsR_plus,FeatsR_minus,MaskGlob,margin=self.triplet_loss.margin)
        validation_loss=validation_loss.sum().div(2*MaskGlob.count_nonzero()+1e-12)
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
        optimizer=torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,15],gamma=0.9)
        return [optimizer],[scheduler]

class UNETGATTWithDecisionNetwork_Dense_LM_N_2(LightningModule):
    def __init__(self,Inplanes,true1=0,false1=2,false2=8,NANS=-999.0):
        super(UNETGATTWithDecisionNetwork_Dense_LM_N_2, self).__init__()
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
        self.feature=UNetGatedAttention(init_features=self.inplanes)
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

if __name__=="__main__":
    MSFeatureExtractor=UNETWithDecisionNetwork_LM.load_from_checkpoint("/media/mohamedali/Thales/PythonProject/trained_models/unetfeaturesenshedebce/version_5/logs/epoch=15-step=12372.ckpt").to(torch.device("cuda:0"))
    model=UNETGatedAttentionWithDecisionNetwork_LM(32,128).to(torch.device("cuda:0"))
    # copy weights from model unet brut to unet with attention
    model.feature.load_state_dict(MSFeatureExtractor.feature.state_dict(),strict=False)
    
    model.decisionNet.load_state_dict(MSFeatureExtractor.decisionNet.state_dict())
    for p1,p2 in zip (model.feature.named_parameters(),MSFeatureExtractor.feature.named_parameters()):
        print(p1[0],p2[0])
        assert(torch.equal(p1[1],p2[1]))
    for p1,p2 in zip (model.decisionNet.parameters(),MSFeatureExtractor.decisionNet.parameters()):
        assert(torch.equal(p1,p2))
    x=torch.rand((2,1,256,256))
    out=model(x)
    print(out.shape)
     
