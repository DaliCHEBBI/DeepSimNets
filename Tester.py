import os, sys
import utils.utils as utils
import argparse
from pathlib import Path
import typing
from models.unetGatedAttention import UNetGatedAttentionWithDecisionNetwork,UNETGATTWithDecisionNetwork_Dense_LM_N_2
from models.MSNETPl import MSNETWithDecisionNetwork_Dense_LM_N_2,MSNETGatedAttentionWithDecisionNetwork
from datasets.CubeDataset import StereoCheckDistribAerialDatasetN
from models.UNetDecisionEBM import UNetWithDecisionNetwork,UNETWithDecisionNetwork_Dense_LM_N_2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import gc
#from matplotlib.ticker import FormatStrFormatter,ScalarFormatter,MaxNLocator,FuncFormatter


def PlotJointDistribution(Simsplus,Simsmoins,which):
    import pandas as pd
    import matplotlib.patches as  mpatches
    import matplotlib.cm as cm
    import seaborn as sns
    """
    fig,ax = plt.subplots(figsize=(6,6),gridspec_kw={'wspace':0.1,'hspace':0.1})
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
    Jp,yedges,xedges=np.histogram2d(Simsplus,Simsmoins,bins=200,weights=np.ones(len(Simsplus))/len(Simsplus),density=True)
    print(Jp)
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.imshow(Jp)
    plt.show()"""
    Simsplus=np.expand_dims(Simsplus, axis=1)
    Simsmoins=np.expand_dims(Simsmoins, axis=1)
    Simall=np.concatenate((Simsplus,Simsmoins),axis=1)
    Simall=Simall[0:100000000,:]
    print(Simall.shape)
    JDistrib=pd.DataFrame(Simall,columns=['+','-'])
    sns.color_palette("RdBu", 20)
    g = sns.jointplot(data = JDistrib,
                    x = "-",
                    y = "+",
                    xlim=(0,1.0),
                    ylim=(0.0,1.0),
                    cmap=cm.jet, 
                    kind="hist", 
                    marginal_kws={"color":"r", "alpha":.4, "bins":200, "stat":'percent'}, 
                    joint_kws={"bins":(200,200)},
                    stat='percent',
                    #shade=True, 
                    #thresh=0.05, 
                    #alpha=.9,
                    #fill=True,
                    marginal_ticks=True,
                    #n_levels=50,
                    cbar=True,
                    #cbar_kws={"use_gridspec":False, "location":"top"},
                    label='{} Module KDE'.format(which)
                    )
    plt.subplots_adjust(left=0.1, right=0.83, top=0.9, bottom=0.1)
    pos_joint_ax = g.ax_joint.get_position()
    pos_marg_x_ax = g.ax_marg_x.get_position()
    g.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
    g.fig.axes[-1].set_position([.85, pos_joint_ax.y0, .07, pos_joint_ax.height])
    #facecolor = g.ax_marg_x.collections[0].get_facecolor()
    #g.ax_marg_x.lines[0].set_color('orange')
    g.ax_marg_y.spines['right'].set_visible(False)
    g.ax_marg_y.spines['left'].set_visible(True)
    g.ax_marg_y.spines['bottom'].set_visible(True)
    g.ax_marg_y.spines['top'].set_visible(False)
    
    #g.ax_marg_y.spines['right'].set_linewidth(0.5)
    g.ax_marg_y.spines['left'].set_linewidth(0.5)
    g.ax_marg_y.spines['bottom'].set_linewidth(0.5)
    #g.ax_marg_y.spines['top'].set_linewidth(0.5)
    g.ax_marg_y.tick_params(axis='both', which='major', labelsize=10)
    g.ax_marg_y.tick_params(axis='both', which='minor', labelsize=10)
    #g.ax_marg_y.tick_params(bottom=True, top=True, left=True, right=True, direction='in')
    g.set_axis_labels(xlabel='Non-matching similarity', ylabel='Matching Similarity', size=12)
    g.ax_joint.grid(color = 'k', linestyle = ':', linewidth = 0.25, dashes=(1, 5))
    g.ax_marg_y.grid(color = 'k', linestyle = ':', linewidth = 0.25, dashes=(1, 5))
    g.ax_marg_x.grid(color = 'k', linestyle = ':', linewidth = 0.25, dashes=(1, 5))
    g.ax_marg_y.invert_yaxis()
    #g.ax_joint.xaxis.tick_top()
    values= g.ax_joint.collections[0].get_array()
    values=np.reshape(values, (200,200))
    #xy_ = g.ax_joint.collections[0]
    #print(xy_)
    SUM_GOOD=0.0
    for j in range(200):
        for i in range(j+1):
            if (values[j,i]!='--'):
                SUM_GOOD+=values[j,i]   
    pourcent=str("%.2f" % SUM_GOOD)+" %"            
    handles = [mpatches.Patch(facecolor=plt.cm.jet(255), label='{} : {}'.format(which,pourcent))]
    g.ax_joint.legend(handles=handles,loc=4)
    plt.savefig('jp_{}.svg'.format(which),format="svg")


def ComputeAreabetweencurves(CurveSet1,CurveSet2):
    SurfaceIntersection=[]
    SurfaceUnion=[]
    for i in range(len(CurveSet1)-1):
         # Get surface between both bins 
         x1,y1,x2,y2=CurveSet1[i][0],CurveSet1[i][1],CurveSet1[i+1][0],CurveSet1[i+1][1]
         xx1,yy1,xx2,yy2=CurveSet2[i][0],CurveSet2[i][1],CurveSet2[i+1][0],CurveSet2[i+1][1]
         Surface1=(x2-x1)*(y1+y2)*0.5
         Surface2=(xx2-xx1)*(yy1+yy2)*0.5
         SurfaceIntersection.append(np.maximum(Surface2,Surface1)-np.abs(Surface2-Surface1))
         SurfaceUnion.append(np.maximum(Surface2,Surface1))
    # likelihood
    #IOU=[a/b for a,b in zip(SurfaceIntersection,SurfaceUnion)]
    return np.sum(SurfaceIntersection)/np.sum(SurfaceUnion)

def ComputeAreaRationPositiveNegative(CurveSet1,CurveSet2,BIN):
    SurfacePositive=[]
    SurfaceNegative=[]
    for i in range(len(CurveSet1)-1):
         # Get surface between both bins 
         x1,y1,x2,y2=CurveSet1[i][0],CurveSet1[i][1],CurveSet1[i+1][0],CurveSet1[i+1][1]
         xx1,yy1,xx2,yy2=CurveSet2[i][0],CurveSet2[i][1],CurveSet2[i+1][0],CurveSet2[i+1][1]
         #print('offsets between both surfaces ',x1,xx1,y1,yy1)
         #print('offsets between both surfaces ',x2,xx2,y2,yy2)
         Surface1=(x2-x1)*(y1+y2)*0.5
         Surface2=(xx2-xx1)*(yy1+yy2)*0.5
         SurfacePositive.append(Surface1)
         SurfaceNegative.append(Surface2)
    # likelihood
    # Binning Factor every n values compute ratio
    likelihoodRatio=[]
    steps=[]
    for i in range(0,len(SurfacePositive),BIN):
        # Compute ratio 
        ratio=sum(SurfacePositive[i:i+BIN])/float(sum(SurfaceNegative[i:i+BIN]))
        likelihoodRatio.append(ratio)
        # Save steps 
        steps.append(CurveSet1[i][0])
    steps.append(CurveSet1[-1][0])
    # Get steps at midle locations 
    return steps,likelihoodRatio

def testing_step_decision_DFC(batch,modulems,device,NANS=-999.0):
    true1=1
    false1=1
    false2=8
    x0,x1,dispnoc0=batch
    dispnoc0=dispnoc0.to(device)
    Mask0=(dispnoc0!=NANS).float().to(device)  # NAN=-999.0
    dispnoc0[dispnoc0==NANS]=0.0 # Set Nans to 0.0
    # Forward
    FeatsL=modulems.feature(x0.to(device)) 
    FeatsR=modulems.feature(x1.to(device))
    Offset_pos=(-2*true1) * torch.rand(dispnoc0.size(),device=device) + true1 #[-true1,true1]
    Offset_neg=((false1 - false2) * torch.rand(dispnoc0.size(),device=device) + false2)
    RandSens=torch.rand(dispnoc0.size(),device=device)
    RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).to(device)
    Offset_neg=Offset_neg*RandSens
    #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
    D_pos=dispnoc0+Offset_pos
    D_neg=dispnoc0+Offset_neg
    Index_X=torch.arange(0,dispnoc0.size()[-1],device=device)
    Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
    Offp=Index_X-D_pos.round()  
    Offn=Index_X-D_neg.round() 
    # Clean Indexes so there is no overhead 
    MaskOffPositive=((Offp>=0)*(Offp<dispnoc0.size()[-1])).float().to(device) 
    MaskOffNegative=((Offn>=0)*(Offn<dispnoc0.size()[-1])).float().to(device)
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
    ref_pos=modulems.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
    ref_neg=modulems.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
    ref_pos=F.sigmoid(ref_pos)
    ref_neg=F.sigmoid(ref_neg)
    simplus=torch.masked_select(ref_pos, MaskGlob.bool())
    simmins=torch.masked_select(ref_neg, MaskGlob.bool())
    return simplus.squeeze().cpu().detach(),simmins.squeeze().cpu().detach()


def ComputeSurfacesIntersection(CurveSet1,CurveSet2):
    SurfaceIntersection=[]
    SurfaceUnion=[]
    for i in range(len(CurveSet1)-1):
         # Get surface between both bins 
         x1,y1,x2,y2=CurveSet1[i][0],CurveSet1[i][1],CurveSet1[i+1][0],CurveSet1[i+1][1]
         xx1,yy1,xx2,yy2=CurveSet2[i][0],CurveSet2[i][1],CurveSet2[i+1][0],CurveSet2[i+1][1]
         Surface1=(x2-x1)*(y1+y2)*0.5
         Surface2=(xx2-xx1)*(yy1+yy2)*0.5
         SurfaceIntersection.append(np.maximum(Surface2,Surface1)-np.abs(Surface2-Surface1))
         SurfaceUnion.append(np.maximum(Surface2,Surface1))
    # likelihood
    """IOU=[a/b for a,b in zip(SurfaceIntersection,SurfaceUnion)]
    print("IOU OF SURFACES   ==>  ",IOU)
    return IOU"""
    IOU=np.asarray(SurfaceIntersection)/np.sum(SurfaceUnion)
    IOU.tofile("./IOU_UNETATT.bin")
    print("IOU   =====>>>>>   ",IOU)
    #return np.sum(SurfaceIntersection)/np.sum(SurfaceUnion)

def ROCCurveAuc(Simplus,Simminus):
    from sklearn import metrics
    from scipy.special import kl_div
    print("KL DIVERGENCE UNETATT ==> ",np.sum(kl_div(Simplus,Simminus))+np.sum(kl_div(Simminus,Simplus)))
    AllSims=np.concatenate((Simplus,Simminus),axis=0)
    Labels=np.concatenate((np.ones(len(Simplus)),np.zeros(len(Simminus))))
    fpr,tpr,_=metrics.roc_curve(Labels,AllSims)
    AUC=metrics.roc_auc_score(Labels,AllSims)
    # RETURN ALL RESULTS 
    """print("AREA UNDER CURVE  ==>  ",AUC)
    print("FALSE POSITVE RATE ==> ",fpr)
    print("TRUE POSITVE RATE  ==> ",tpr)"""
    fpr.tofile('./FPR_UNETATT.bin')
    print("SHAPE OF TPR UNETATT  ====>  ",tpr.shape)
    tpr.tofile('./TPR_UNETATT.bin')
    print("SHAPE OF FPR UNETATT  ====>  ",fpr.shape)
    print("AUC UNETATT     =====>  ",AUC)

def testing_step_dense(batch,modulems,device,NANS=-999.0):  
    false1=2
    false2=40
    x0,x1,dispnoc0,Mask0,x_offset=batch
    # ADD DIM 1
    dispnoc0=dispnoc0.unsqueeze(1).cuda()
    Mask0=Mask0.unsqueeze(1).cuda()

    #print("initial shapes ", x0.shape,x1.shape,dispnoc0.shape,Mask0.shape)
    dispnoc0=dispnoc0*Mask0.float() # Set Nans to 0.0
    #print("Disparity shaope ",dispnoc0.shape)
    # Forward
    FeatsL=modulems.feature(x0.cuda()) 
    FeatsR=modulems.feature(x1.cuda())
    Offset_neg=((false1 - false2) * torch.rand(dispnoc0.size()).cuda() + false2)
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
    Index_X=Index_X.add(x_offset.cuda().unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
    #print("Index SHAPE ", Index_X.shape)
    Offp=Index_X-D_pos.round()  
    Offn=Index_X-D_neg.round() 
    #print("offsets shapes ", Offp.shape, Offn.shape)
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
    ref_pos=modulems.decisionNet(torch.cat((FeatsL,FeatsR_plus),1))
    ref_neg=modulems.decisionNet(torch.cat((FeatsL,FeatsR_minus),1))
    ref_pos=F.sigmoid(ref_pos)
    ref_neg=F.sigmoid(ref_neg)
    simplus=torch.masked_select(ref_pos, MaskGlob.bool())
    simmins=torch.masked_select(ref_neg, MaskGlob.bool())
    return simplus.squeeze().cpu().detach(),simmins.squeeze().cpu().detach()

def testing_step_DFC(batch,modulems,device,NANS=-999.0):
    true1=1
    false1=2
    false2=8
    x0,x1,dispnoc0=batch
    dispnoc0=dispnoc0.to(device)
    Mask0=(dispnoc0!=NANS).float().to(device)  # NAN=-999.0
    dispnoc0[dispnoc0==NANS]=0.0 # Set Nans to 0.0
    # Forward
    FeatsL=modulems(x0.to(device)) 
    FeatsR=modulems(x1.to(device))
    Offset_pos=(-2*true1) * torch.rand(dispnoc0.size(),device=device) + true1 #[-true1,true1]
    Offset_neg=((false1 - false2) * torch.rand(dispnoc0.size(),device=device) + false2)
    RandSens=torch.rand(dispnoc0.size(),device=device)
    RandSens=((RandSens < 0.5).float()+(RandSens >= 0.5).float()*(-1.0)).to(device)
    Offset_neg=Offset_neg*RandSens
    #dispnoc0=torch.nan_to_num(dispnoc0, nan=0.0)
    D_pos=dispnoc0+Offset_pos
    D_neg=dispnoc0+Offset_neg
    Index_X=torch.arange(0,dispnoc0.size()[-1],device=device)
    Index_X=Index_X.expand(dispnoc0.size()[-2],dispnoc0.size()[-1]).unsqueeze(0).unsqueeze(0).repeat_interleave(x0.size()[0],0)
    Offp=Index_X-D_pos.round()  
    Offn=Index_X-D_neg.round() 
    # Clean Indexes so there is no overhead 
    MaskOffPositive=((Offp>=0)*(Offp<dispnoc0.size()[-1])).float().to(device) 
    MaskOffNegative=((Offn>=0)*(Offn<dispnoc0.size()[-1])).float().to(device) 
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
    # Save maks global 
    simplus=F.cosine_similarity(FeatsL, FeatsR_plus).unsqueeze(1)
    simmins=F.cosine_similarity(FeatsL, FeatsR_minus).unsqueeze(1)
    simplus=torch.masked_select(simplus, MaskGlob.bool())
    simmins=torch.masked_select(simmins, MaskGlob.bool())
    return simplus.squeeze().cpu().detach(),simmins.squeeze().cpu().detach()

def testing_step_dense_COSINE(batch,modulems,device,NANS=-999.0):
    false1=2
    false2=6
    x0,x1,dispnoc0,Mask0,x_offset=batch
    # ADD DIM 1
    dispnoc0=dispnoc0.unsqueeze(1).cuda()
    Mask0=Mask0.unsqueeze(1).cuda()

    #print("initial shapes ", x0.shape,x1.shape,dispnoc0.shape,Mask0.shape)
    dispnoc0=dispnoc0*Mask0.float() # Set Nans to 0.0
    #print("Disparity shaope ",dispnoc0.shape)
    # Forward
    FeatsL=modulems.feature(x0.cuda()) 
    FeatsR=modulems.feature(x1.cuda())
    Offset_neg=((false1 - false2) * torch.rand(dispnoc0.size()).cuda() + false2)
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
    Index_X=Index_X.add(x_offset.cuda().unsqueeze(0).T.unsqueeze(2).unsqueeze(3))
    #print("Index SHAPE ", Index_X.shape)
    Offp=Index_X-D_pos.round()  
    Offn=Index_X-D_neg.round() 
    #print("offsets shapes ", Offp.shape, Offn.shape)
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
    simplus=F.cosine_similarity(FeatsL, FeatsR_plus).unsqueeze(1)
    simmins=F.cosine_similarity(FeatsL, FeatsR_minus).unsqueeze(1)
    simplus=torch.masked_select(simplus, MaskGlob.bool())
    simmins=torch.masked_select(simmins, MaskGlob.bool())
    return simplus.squeeze().cpu().detach(),simmins.squeeze().cpu().detach()

def TestMSNet(net, test_loader, device,nans=-999.0):
    torch.no_grad()
    net.eval()
    Simsplus=[]
    Simsmoins=[]
    for _, batch in enumerate(test_loader, 0):
        simP,simN=testing_step_dense(batch,net,device,nans)
        gc.collect()
        print("Sizes of Simplus ",simP.shape)
        # compute loss
        Simsplus.append(simP.numpy())
        Simsmoins.append(simN.numpy())
    return Simsplus,Simsmoins
        
def MemStatus(loc):
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t -m').readlines()[-1].split()[1:])
    # Memory usage
    print("RAM memory % used before @ "+ loc, round((used_memory/total_memory) * 100, 2))

def collate_func(batch):
    return torch.cat(batch,0)

def make_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-mdl",
        type=str,
        help="Model name to train, possible names are: 'MS-AFF', 'U-Net32', 'U-Net_Attention'",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        "-ckpt",
        type=str,
        help="Model checkpoint to load",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        help="output folder to store results",
        required=True,
    )
    return parser

def main(arg_strings: typing.Sequence[str]):
    arg_parser=make_arg_parser()
    args = arg_parser.parse_args(arg_strings)
    torch.backends.cudnn.benchmark=True
    _Model2test=args.model
    _Model_ckpt=args.checkpoint
    _output_folder=args.output_folder
    os.makedirs(_output_folder,exist_ok=True)
    if (_Model2test=="MS-AFF"):
        # feature extractor = MS-AFF
        # decision network  = MLP 
        ALL_CHECKPOINTS=[
            _Model_ckpt,
        ]
        for aCKPT in ALL_CHECKPOINTS:
            NAME_EPOCH='MS_AFF_'+os.path.basename(aCKPT)[:-5]
            MSFeatureExtractor=MSNETWithDecisionNetwork_Dense_LM_N_2.load_from_checkpoint(aCKPT)
            ModelInf=MSNETGatedAttentionWithDecisionNetwork(32,128)
            ModelInf.feature.load_state_dict(MSFeatureExtractor.feature.state_dict())
            ModelInf.decisionNet.load_state_dict(MSFeatureExtractor.decisionNet.state_dict())
            for p1,p2 in zip (ModelInf.feature.parameters(),MSFeatureExtractor.feature.parameters()):
                assert(torch.equal(p1,p2))
            for p1,p2 in zip (ModelInf.decisionNet.parameters(),MSFeatureExtractor.decisionNet.parameters()):
                assert(torch.equal(p1,p2))
            ModelInf=ModelInf.cuda()
            Test_dataset=StereoCheckDistribAerialDatasetN('/tmp/DUBLIN_DENSE/docker_image_names',0.434583236,0.194871725,'DUBLIN')
            # Data loader for testing 
            val_loader=torch.utils.data.DataLoader(Test_dataset, batch_size=1, shuffle=False,drop_last=True,pin_memory=True, num_workers=12)
            Simsplus, Simsmoins=TestMSNet(ModelInf,val_loader,torch.device("cuda:0"),0.0)
            Simsplus=np.concatenate(Simsplus, axis=0 )
            Simsmoins=np.concatenate(Simsmoins, axis=0 )
            fig1=plt.figure()
            # Compute histogram of positive similarities and negatives 
            Histplus,Histplusbins=np.histogram(Simsplus,bins=200,density=True,normed=True)
            #Histplusbins = Histplusbins[:-1] + (Histplusbins[1] - Histplusbins[0])/2
            Histmoins,Histmoinsbins=np.histogram(Simsmoins,bins=200,density=True,normed=True)
            #Histmoinsbins = Histmoinsbins[:-1] + (Histmoinsbins[1] - Histmoinsbins[0])/2
            #print(np.trapz(Histplus,Histplusbins))
            plt.rcParams.update({'font.size': 10})
            plt.plot(Histplusbins[1:],Histplus, label='Prob. mass of positive samples')
            plt.plot(Histmoinsbins[1:],Histmoins, label='Prob. mass of negative samples')
            PourcentageIntersection=ComputeAreabetweencurves(list(tuple(zip(Histplusbins, Histplus))),list(tuple(zip(Histmoinsbins, Histmoins))))
            ComputeSurfacesIntersection(list(tuple(zip(Histplusbins, Histplus))),list(tuple(zip(Histmoinsbins, Histmoins))))
            ROCCurveAuc(Simsplus,Simsmoins)

            plt.fill_between(Histplusbins[1:], Histplus, step="pre", alpha=0.2)
            plt.fill_between(Histmoinsbins[1:], Histmoins, step="pre", alpha=0.2)
            Histplus=np.cumsum(Histplus*np.diff(Histplusbins))
            Histmoins=np.cumsum(Histmoins*np.diff(Histmoinsbins))
            STEPS,RATIO=ComputeAreaRationPositiveNegative(list(tuple(zip(Histplusbins, Histplus))),list(tuple(zip(Histmoinsbins, Histmoins))),1)
            plt.vlines(x=np.mean(Simsplus), ymin=0, ymax=3.0, colors='blue', ls=':', lw=2, label='mean+ ='+"{:.2f}".format(np.mean(Simsplus)))
            plt.vlines(x=np.mean(Simsmoins), ymin=0, ymax=3.0, colors='orange', ls=':', lw=2, label='mean- ='+"{:.2f}".format(np.mean(Simsmoins)))
            plt.xlabel("Similarity values")
            plt.ylabel("Count Number (%)")
            plt.title("MS_AFF-DECISION:Tr-DUB_RETR:Te-DUB_RETR_{:.2f}%".format(PourcentageIntersection*100))
            #legend_properties = {'weight':'bold'}
            plt.legend(fontsize=10)
            plt.savefig("{}/OCC_AWARE_NORMED_False2PX40_DENSE_DATA_MS_AFF+Decision:Tr-AERIAL:Te-DUB_{}_Surf_{:.2f}%".format(_output_folder,NAME_EPOCH,PourcentageIntersection*100)+".png")
            # plot likelihood ration function
            fig2, ax = plt.subplots()
            RATIO=np.asarray(RATIO)
            STEPS=np.asarray(STEPS)
            RATIO=np.interp(RATIO,(RATIO.min(),RATIO.max()),(0,1.0))
            plt.plot(STEPS[:-1], RATIO)
            print(STEPS,RATIO)
            plt.xlabel("Similarity values")
            plt.ylabel("Ratio")
            plt.title("likelihood Ratio MS_AFF-DECISION Network")
            plt.savefig("{}/OCC_AWARE_NORMED_False2PX40_DENSE_Likelihood_Model_MS_AFF_Tr-AERIAL:Te-DUB_{}_Surf_{:.2f}%".format(_output_folder,NAME_EPOCH,PourcentageIntersection*100)+".png")
    elif(_Model2test=="U-Net32"):
        ALL_CHECKPOINTS=[
            _Model_ckpt,
        ]
        for aCKPT in ALL_CHECKPOINTS:
            NAME_EPOCH='MS_AFF_'+os.path.basename(aCKPT)[:-5]
            MSFeatureExtractor=UNETWithDecisionNetwork_Dense_LM_N_2.load_from_checkpoint(aCKPT)
            ModelInf=UNetWithDecisionNetwork(32,128)
            ModelInf.feature.load_state_dict(MSFeatureExtractor.feature.state_dict())
            ModelInf.decisionNet.load_state_dict(MSFeatureExtractor.decisionNet.state_dict())
            for p1,p2 in zip (ModelInf.feature.parameters(),MSFeatureExtractor.feature.parameters()):
                assert(torch.equal(p1,p2))
            for p1,p2 in zip (ModelInf.decisionNet.parameters(),MSFeatureExtractor.decisionNet.parameters()):
                assert(torch.equal(p1,p2))
            ModelInf=ModelInf.cuda()
            Test_dataset=StereoCheckDistribAerialDatasetN('/tmp/DUBLIN_DENSE/docker_image_names',0.434583236,0.194871725,'DUBLIN')
            # Data loader for testing 
            val_loader=torch.utils.data.DataLoader(Test_dataset, batch_size=1, shuffle=False,drop_last=True,pin_memory=True, num_workers=12)
            Simsplus, Simsmoins=TestMSNet(ModelInf,val_loader,torch.device("cuda:0"),0.0)
            Simsplus=np.concatenate(Simsplus, axis=0 )
            Simsmoins=np.concatenate(Simsmoins, axis=0 )
            fig1=plt.figure()
            # Compute histogram of positive similarities and negatives 
            Histplus,Histplusbins=np.histogram(Simsplus,bins=200,density=True,normed=True)
            #Histplusbins = Histplusbins[:-1] + (Histplusbins[1] - Histplusbins[0])/2
            Histmoins,Histmoinsbins=np.histogram(Simsmoins,bins=200,density=True,normed=True)
            #Histmoinsbins = Histmoinsbins[:-1] + (Histmoinsbins[1] - Histmoinsbins[0])/2
            #print(np.trapz(Histplus,Histplusbins))
            plt.rcParams.update({'font.size': 10})
            plt.plot(Histplusbins[1:],Histplus, label='Prob. mass of positive samples')
            plt.plot(Histmoinsbins[1:],Histmoins, label='Prob. mass of negative samples')
            PourcentageIntersection=ComputeAreabetweencurves(list(tuple(zip(Histplusbins, Histplus))),list(tuple(zip(Histmoinsbins, Histmoins))))
            ComputeSurfacesIntersection(list(tuple(zip(Histplusbins, Histplus))),list(tuple(zip(Histmoinsbins, Histmoins))))
            ROCCurveAuc(Simsplus,Simsmoins)

            plt.fill_between(Histplusbins[1:], Histplus, step="pre", alpha=0.2)
            plt.fill_between(Histmoinsbins[1:], Histmoins, step="pre", alpha=0.2)
            Histplus=np.cumsum(Histplus*np.diff(Histplusbins))
            Histmoins=np.cumsum(Histmoins*np.diff(Histmoinsbins))
            STEPS,RATIO=ComputeAreaRationPositiveNegative(list(tuple(zip(Histplusbins, Histplus))),list(tuple(zip(Histmoinsbins, Histmoins))),1)
            plt.vlines(x=np.mean(Simsplus), ymin=0, ymax=3.0, colors='blue', ls=':', lw=2, label='mean+ ='+"{:.2f}".format(np.mean(Simsplus)))
            plt.vlines(x=np.mean(Simsmoins), ymin=0, ymax=3.0, colors='orange', ls=':', lw=2, label='mean- ='+"{:.2f}".format(np.mean(Simsmoins)))
            plt.xlabel("Similarity values")
            plt.ylabel("Count Number (%)")
            plt.title("UNET32-DECISION:Tr-DUB_RETR:Te-DUB_RETR_{:.2f}%".format(PourcentageIntersection*100))
            #legend_properties = {'weight':'bold'}
            plt.legend(fontsize=10)
            plt.savefig("{}/OCC_AWARE_NORMED_False2PX40_DENSE_DATA_UNET32+Decision:Tr-AERIAL:Te-DUB_{}_Surf_{:.2f}%".format(_output_folder,NAME_EPOCH,PourcentageIntersection*100)+".png")
            # plot likelihood ration function
            fig2, ax = plt.subplots()
            RATIO=np.asarray(RATIO)
            STEPS=np.asarray(STEPS)
            RATIO=np.interp(RATIO,(RATIO.min(),RATIO.max()),(0,1.0))
            plt.plot(STEPS[:-1], RATIO)
            print(STEPS,RATIO)
            plt.xlabel("Similarity values")
            plt.ylabel("Ratio")
            plt.title("likelihood Ratio UNET32-DECISION Network")
            plt.savefig("{}/OCC_AWARE_NORMED_False2PX40_DENSE_Likelihood_Model_UNET32_Tr-AERIAL:Te-DUB_{}_Surf_{:.2f}%".format(_output_folder,NAME_EPOCH,PourcentageIntersection*100)+".png")
    elif(_Model2test=="U-Net_Attention"):
        ALL_CHECKPOINTS=[
            _Model_ckpt,
        ]
        for aCKPT in ALL_CHECKPOINTS:
            NAME_EPOCH='ATTENTION_'+os.path.basename(aCKPT)[:-5]
            MSFeatureExtractor=UNETGATTWithDecisionNetwork_Dense_LM_N_2.load_from_checkpoint(aCKPT)
            ModelInf=UNetGatedAttentionWithDecisionNetwork(32,128)
            ModelInf.feature.load_state_dict(MSFeatureExtractor.feature.state_dict())
            ModelInf.decisionNet.load_state_dict(MSFeatureExtractor.decisionNet.state_dict())
            for p1,p2 in zip (ModelInf.feature.parameters(),MSFeatureExtractor.feature.parameters()):
                assert(torch.equal(p1,p2))
            for p1,p2 in zip (ModelInf.decisionNet.parameters(),MSFeatureExtractor.decisionNet.parameters()):
                assert(torch.equal(p1,p2))
            ModelInf=ModelInf.to(torch.device("cuda:0"))
            Test_dataset=StereoCheckDistribAerialDatasetN('/tmp/DUBLIN_DENSE/docker_image_names',0.434583236,0.194871725,'DUBLIN')
            # Data loader for testing 
            val_loader=torch.utils.data.DataLoader(Test_dataset, batch_size=1, shuffle=False,drop_last=True,pin_memory=True, num_workers=12)
            Simsplus, Simsmoins=TestMSNet(ModelInf,val_loader,torch.device("cuda:0"),0.0)
            Simsplus=np.concatenate(Simsplus, axis=0 )
            Simsmoins=np.concatenate(Simsmoins, axis=0 )
            fig1=plt.figure()
            # Compute histogram of positive similarities and negatives 
            Histplus,Histplusbins=np.histogram(Simsplus,bins=200,density=True,normed=True)
            #Histplusbins = Histplusbins[:-1] + (Histplusbins[1] - Histplusbins[0])/2
            Histmoins,Histmoinsbins=np.histogram(Simsmoins,bins=200,density=True,normed=True)
            #Histmoinsbins = Histmoinsbins[:-1] + (Histmoinsbins[1] - Histmoinsbins[0])/2
            #print(np.trapz(Histplus,Histplusbins))
            plt.rcParams.update({'font.size': 10})
            plt.plot(Histplusbins[1:],Histplus, label='Prob. mass of positive samples')
            plt.plot(Histmoinsbins[1:],Histmoins, label='Prob. mass of negative samples')
            PourcentageIntersection=ComputeAreabetweencurves(list(tuple(zip(Histplusbins, Histplus))),list(tuple(zip(Histmoinsbins, Histmoins))))
            ComputeSurfacesIntersection(list(tuple(zip(Histplusbins, Histplus))),list(tuple(zip(Histmoinsbins, Histmoins))))
            ROCCurveAuc(Simsplus,Simsmoins)

            plt.fill_between(Histplusbins[1:], Histplus, step="pre", alpha=0.2)
            plt.fill_between(Histmoinsbins[1:], Histmoins, step="pre", alpha=0.2)
            Histplus=np.cumsum(Histplus*np.diff(Histplusbins))
            Histmoins=np.cumsum(Histmoins*np.diff(Histmoinsbins))
            STEPS,RATIO=ComputeAreaRationPositiveNegative(list(tuple(zip(Histplusbins, Histplus))),list(tuple(zip(Histmoinsbins, Histmoins))),1)
            plt.vlines(x=np.mean(Simsplus), ymin=0, ymax=3.0, colors='blue', ls=':', lw=2, label='mean+ ='+"{:.2f}".format(np.mean(Simsplus)))
            plt.vlines(x=np.mean(Simsmoins), ymin=0, ymax=3.0, colors='orange', ls=':', lw=2, label='mean- ='+"{:.2f}".format(np.mean(Simsmoins)))
            plt.xlabel("Similarity values")
            plt.ylabel("Count Number (%)")
            plt.title("UNET_ATTENTION-DECISION:Tr-DUB_RETR:Te-DUB_RETR_{:.2f}%".format(PourcentageIntersection*100))
            #legend_properties = {'weight':'bold'}
            plt.legend(fontsize=10)
            plt.savefig("{}/OCC_AWARE_NORMED_False2PX40_DENSE_DATA_UNET_ATTENTION+Decision:Tr-AERIAL:Te-DUB_{}_Surf_{:.2f}%".format(_output_folder,NAME_EPOCH,PourcentageIntersection*100)+".png")
            # plot likelihood ration function
            fig2, ax = plt.subplots()
            RATIO=np.asarray(RATIO)
            STEPS=np.asarray(STEPS)
            RATIO=np.interp(RATIO,(RATIO.min(),RATIO.max()),(0,1.0))
            plt.plot(STEPS[:-1], RATIO)
            print(STEPS,RATIO)
            plt.xlabel("Similarity values")
            plt.ylabel("Ratio")
            plt.title("likelihood Ratio UNET_ATTENTION-DECISION Network")
            plt.savefig("{}/OCC_AWARE_NORMED_False2PX40_DENSE_Likelihood_Model_UNET_ATTENTION_Tr-AERIAL:Te-DUB_{}_Surf_{:.2f}%".format(_output_folder,NAME_EPOCH,PourcentageIntersection*100)+".png")
    else:
        raise RuntimeError("Model name should be one of  these models : 'MS-AFF', 'U-Net32', 'U-Net_Attention'")

if __name__ == '__main__':
    main(sys.argv[1:])