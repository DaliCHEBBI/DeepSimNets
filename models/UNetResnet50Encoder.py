import torch
import torch.nn as nn
import torchvision
from pytorch_lightning import LightningModule
import gc
from models.model import DecisionNetwork
import torch.nn.functional as F
resnet = torchvision.models.resnet50(pretrained=True)
mm=nn.Conv2d(1, 64, kernel_size=7, stride=2,padding=3,bias=False)
kern=list(resnet.children())[0].weight[:,0,:,:].unsqueeze(1)

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6
    def __init__(self, n_classes=2):
        super().__init__()
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        with torch.no_grad():
            self.input_block[0]=mm
            self.input_block[0].weight.copy_(kern)
        # Reactivate gradient computation
        self.input_block[0].weight.requires_grad=True
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 1, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        #self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=True):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        del pre_pools
        return x

class UNetWithResnet50EncoderInf(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.DEPTH = 6
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        with torch.no_grad():
            self.input_block[0]=mm
            self.input_block[0].weight.copy_(kern)
        # Reactivate gradient computation
        self.input_block[0].weight.requires_grad=True
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 1, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        #self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        if x.size()[-2] % 32 != 0:
            times = x.size()[-2]//32   
            top_pad = (times+1)*32 - x.size()[-2]
        else:
            top_pad = 0
        if x.size()[-1] % 32 != 0:
            times = x.size()[-1]//32
            right_pad = (times+1)*32-x.size()[-1] 
        else:
            right_pad = 0    

        x = F.pad(x,(0,right_pad, top_pad,0))
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i != (self.DEPTH - 1):
                pre_pools[f"layer_{i}"] = x
            #pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{self.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        del pre_pools
        if top_pad !=0 and right_pad != 0:
            out = x[:,:,top_pad:,:-right_pad]
        elif top_pad ==0 and right_pad != 0:
            out = x[:,:,:,:-right_pad]
        elif top_pad !=0 and right_pad == 0:
            out = x[:,:,top_pad:,:]
        else:
            out = x
        return out

class UNETRESNET50WithDecisionNetwork(nn.Module):
     def __init__(self,outfeats):
         super(UNETRESNET50WithDecisionNetwork,self).__init__()
         self.out_features=outfeats
         self.feature=UNetWithResnet50EncoderInf()
         self.decisionNet=DecisionNetwork(self.out_features)
     def forward(self,x):
         f_all=self.feature(x)
         #print(f_all.shape)
         # shape 2,64,w,h
         ref_other=torch.cat((f_all[0].unsqueeze(0),f_all[1].unsqueeze(0)),1)
         out=self.decisionNet(ref_other)
         return torch.sigmoid(out)

class UNETRESNET50WithDecisionNetwork_LM(LightningModule):
    def __init__(self,true1=1,false1=2,false2=8,NANS=-999.0):
        super(UNETRESNET50WithDecisionNetwork_LM, self).__init__()
        #self.device=torch.device("cuda:0")
        self.true1=true1
        self.false1=false1
        self.false2=false2
        self.nans=NANS
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.learning_rate=0.0005
        self.nbsteps=0
        self.save_hyperparameters()
        #self.example_input_array=torch.zeros((2,1,1024,1024),dtype=torch.float32)
        self.feature=UNetWithResnet50Encoder()
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
        training_loss=self.criterion(sample, target.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob,MaskGlob),0)
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
        validation_loss=self.criterion(sample, target.to(device=self.device, dtype=torch.float))*torch.cat((MaskGlob,MaskGlob),0)
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

if __name__=="__main__":
     model=UNETRESNET50WithDecisionNetwork(128)
     x=torch.rand((5,1,35,37))
     out=model(x)
     print(out.shape)