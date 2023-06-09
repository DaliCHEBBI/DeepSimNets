import torch
import torch.nn as nn
import torch.nn.functional as F

class DecisionNetwork(nn.Module):
    def __init__(self,features_in):
        super(DecisionNetwork,self).__init__()
        self.f_in=features_in
        # Construct Linear Modules 
        self.fl_blocks = nn.Sequential(
            nn.Linear(in_features=self.f_in, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=1),
            #nn.Sigmoid(),
        )
    def forward(self,x):
        x=x.permute(0, 2, 3, 1)
        x=self.fl_blocks(x)
        return x.permute(0, 3, 1, 2)

class DecisionNetworkOnCube(nn.Module):
    def __init__(self,features_in):
        super(DecisionNetworkOnCube,self).__init__()
        self.f_in=features_in
        # Construct Linear Modules 
        self.fl_blocks = nn.Sequential(
            nn.Linear(in_features=self.f_in, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=1),
            #nn.Sigmoid(),
        )
    def forward(self,x):
        # x is a cube of shape FEATURES,Disparity, H,W
        x=x.permute(1, 2, 3, 0)
        x=self.fl_blocks(x)
        return x.squeeze()

class DecisionNetworkOnNCubes(nn.Module):
    def __init__(self,features_in):
        super(DecisionNetworkOnNCubes,self).__init__()
        self.f_in=features_in
        # Construct Linear Modules 
        self.fl_blocks = nn.Sequential(
            nn.Linear(in_features=self.f_in, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=1),
            #nn.Sigmoid(),
        )
    def forward(self,x):
        # x is a cube of shape FEATURES,Disparity, H,W
        x=x.permute(0,2, 3, 4, 1)
        x=self.fl_blocks(x)
        x=x.permute(0,4, 1, 2, 3)
        return x

class DecisionNetwork5D(nn.Module):
    def __init__(self,features_in):
        super(DecisionNetwork5D,self).__init__()
        self.f_in=features_in
        # Construct Linear Modules 
        self.fl_blocks = nn.Sequential(
            nn.Linear(in_features=self.f_in, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=1),
            #nn.Sigmoid(),
        )
    def forward(self,x):
        x=x.permute(0, 1, 3, 4, 2) # N ,2f, 64 ,h, w ==> N, 2f, h, w ,64 
        x=self.fl_blocks(x)                            # 0,  1, 2, 3 ,4
        return x.permute(0, 1, 4, 2, 3) # ==> dim:  N ,2f, 1 ,h, w 

if __name__=="__main__":
     model=DecisionNetwork(128)
     x=torch.rand((2,128,100,100))
     x=x.permute(0, 2, 3, 1)
     out=model(x)
     out=out.permute(0, 3, 1, 2)
     print(out.shape)
