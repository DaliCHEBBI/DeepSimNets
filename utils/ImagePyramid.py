import numpy as np
import torch 
import torch.nn.functional as F
#import kornia.geometry as KGeom 
#from kornia.filters import gaussian_blur2d

#USED GAUSSIAN KERNEL FOR BLURRING RECURSIVELY THE IMAGES 
def _get_pyramid_gaussian_kernel():
 #Utility function that return a pre-computed gaussian kernel.
    return (
        torch.tensor(
            [
                [
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [6.0, 24.0, 36.0, 24.0, 6.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                ]
            ]
        )
        / 256.0
    )



"""
	def GaussKern(self,LOWPASS_R, scale):
		ivar2 = float(1.0 / (2.0*scale*scale))
		kernel=torch.empty(( 2 * LOWPASS_R + 1,2 * LOWPASS_R + 1 ), dtype=torch.float32,device=torch.device('cpu'))
		for i in np.arange(-LOWPASS_R,LOWPASS_R+1):	
			for j in np.arange(-LOWPASS_R,LOWPASS_R+1):
				val = float(np.exp(-(j*j + i*i)/ivar2))
				print("kernel value ",val)
				kernel[i+LOWPASS_R,j+LOWPASS_R]=val	
		return kernel
"""
"""
		IMAGE PYRAMID GENERATION UTILITY  """

class ImagePyramid:
	def __init__(self,nLevels, scaleFactor=1.6):
		self.mnLevels=nLevels
		self.mscaleFactor=scaleFactor

	def GaussKern(self,kernel_size, sigma):
		sample_points=torch.arange(-kernel_size, kernel_size + 1,1,dtype=torch.float32).unsqueeze(1).expand(-1,2*kernel_size+1)
		kernel = torch.exp(-(sample_points.square() + sample_points.transpose(0, 1).square())/ (2 * sigma * sigma)) / np.sqrt((2 * np.pi * sigma * sigma))
		kernel /= kernel.sum()
		return kernel

	def resizeT(self,im, Newsizex, Newsizey ):
		Out=F.interpolate(im,size=(Newsizey,Newsizex),mode='bilinear')
		return Out
	def TensorPyramidCustom(self,imT):
		m_imPyrT=[]
		#m_imPyrT.append(torch.empty(( 1, 1 , imT.shape[2],imT.shape[3] ), dtype=torch.float32,device=torch.device('cpu')))
		kern = self.GaussKern(2, 1.0)
		print("difference of kernels ",kern -_get_pyramid_gaussian_kernel())
		#print( " KERNEL  GAUSS at Level  0 ",kern )
		imTcnv = F.conv2d(imT, kern.unsqueeze(0).unsqueeze(0),padding=2)
		m_imPyrT.append(imT)
		if (self.mnLevels>1):
			for lvl in np.arange(1,self.mnLevels):
				scale = 1.0 / pow(self.mscaleFactor,lvl)
				Newsizex = int(np.floor(imT.shape[3]*scale))
				Newsizey = int(np.floor(imT.shape[2]*scale))
				NewIm=gaussian_blur2d(m_imPyrT[lvl-1])
				aCropT = self.resizeT(NewIm, Newsizex, Newsizey)
				m_imPyrT.append(aCropT)
		return m_imPyrT
	def TensorPyramid(self, imT):
		#kern = self.GaussKern(2, 1.0)
		#print("difference of kernels ",(kern -_get_pyramid_gaussian_kernel()).sum())
		m_imPyrT=KGeom.transform.pyramid.build_pyramid(imT, 4)
		return m_imPyrT
