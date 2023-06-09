from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes,track_running_stats=True))


def conv1x1(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
			nn.BatchNorm2d(out_planes,track_running_stats=True))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class MSCAM(nn.Module):
    def __init__(self, in_channels,r):
        super(MSCAM, self).__init__()
		inter_channels=in_channels/r;
		# LOCAL ATTENTION
		alocal_attention=aglobal_attention=[]
		alocal_attention.append(conv1x1(in_channels,inter_channels,1,0))
		alocal_attention.append(nn.BatchNorm2d(inter_channels,track_running_stats=True))
		alocal_attention.append(nn.ReLU(inplace=True))
		alocal_attention.append(conv1x1(inter_channels,in_channels,1,0))
		alocal_attention.append(nn.BatchNorm2d(in_channels,track_running_stats=True))
		self.local_attention=nn.Sequential(*alocal_attention)

		# GLOBAL ATTENTION Sequential
		aglobal_attention.append(nn.AdaptiveAvgPool2d(1))
		aglobal_attention.append(conv1x1(in_channels,inter_channels,1,0))
		aglobal_attention.append(nn.BatchNorm2d(inter_channels,track_running_stats=True))
		aglobal_attention.append(nn.ReLU(inplace=True))
		aglobal_attention.append(conv1x1(inter_channels,in_channels,1,0))
		aglobal_attention.append(nn.BatchNorm2d(in_channels,track_running_stats=True))
		self.global_attention=nn.Sequential(*aglobal_attention)
	
	def forward(self,X,Y):
		X_all=X+Y
		xl=self.local_attention(X_all)
		xg=self.global_attention(X_all)
		xlg=xl+xg
		weight=torch.sigmoid(xlg)
		return X.mul(weight)+ Y.mul(weight.mul(-1.0).add(1.0))

class MSNET(nn.Module):
    def __init__(self):
        super(MSNET, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(1, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 64, 3, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, 3, 1,1,1) 
        self.layer3 = self._make_layer(BasicBlock, 64, 3, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 64, 3, 1,1,1)

		# Multi-Scale fusion attention modules 
		MultiScaleFeatureFuser3_4=nn.Sequential(MSCAM(64,4))
		MultiScaleFeatureFuser2_3_4=nn.Sequential(MSCAM(64,4)) 
		MultiScaleFeatureFuser1_2_3_4=nn.Sequential(MSCAM(64,4))
		
		# common network to compute the last features 
		self.common=nn.Sequential(convbn(64,64, 1, 0, 1),
								  nn.ReLU(inplace=True),
								  convbn(64,64, 1, 0, 1),
								  nn.ReLU(inplace=True),
								  nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, dilation = 1, bias=False))
	
    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)
	
	# msnet attention foward 
	def foward(self,x):
		grps=x.chunk(4,1)
		x1=self.firstconv(grps[0])
		x2=self.firstconv(grps[1])
		x3=self.firstconv(grps[2])
		x4=self.firstconv(grps[3])
		x1=self.layer1(x1)
		x2=self.layer2(x2)
		x3=self.layer3(x3)
		x4=self.layer4(x4)
		# FUSE FEATURES FROM DIFFERENT SCALES 
		x_3_4=self.MultiScaleFeatureFuser3_4(x3,x4)
		x_2_3_4=self.MultiScaleFeatureFuser2_3_4(x2,x_3_4)
		x_all=self.MultiScaleFeatureFuser1_2_3_4(x1,x_2_3_4)
		# Pass to last conv block
		x_all=self.common(x_all)
		return x_all
