# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 19:58:39 2025

@author: Emadeldeeen
"""


import math
import torch
import torch.nn as nn


class DownsampleLY(nn.Module):
    def __init__(self, learnable=False):
        super(DownsampleLY, self).__init__()
     
        # 1d CONVOLUTION LAYER
        self.filter1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        
        # fIR filter weights
        #fir_weights1 = torch.tensor([[ [0.25, -0.5, 0.25] ]], dtype=torch.float32)
        fir_weights1 = torch.randn(1, 1, 3, dtype=torch.float32)
        self.filter1.weight = nn.Parameter(fir_weights1, requires_grad=learnable)
        
        # # 1d CONVOLUTION LAYER
        self.filter2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        
        # # fIR filter weights
        #fir_weights2 = torch.tensor([[ [0.25, 0.5, 0.25] ]], dtype=torch.float32)
        fir_weights2 = torch.randn(1, 1, 3, dtype=torch.float32)
        self.filter2.weight = nn.Parameter(fir_weights2, requires_grad=learnable)
        
    	
        
    def forward(self, x):
        x = x.unsqueeze(1) 
        
        x = self.filter1(x)   
        x =  x[:, :, ::2]    
        x = self.filter2(x)   
        x =  x[:, :, ::2]    
        
        x = x.squeeze(1)    
        
        return x

        
def hardthreshold(x,t):
    x = torch.mul(torch.sign(x), torch.nn.functional.relu(torch.abs(x)-torch.abs(t)))
    y=x+torch.sign(x)*torch.abs(t)
    
    return y
        
        
def create_dct_ii_matrix(N):
    """ Create a DCT-II (Type II Discrete Cosine Transform) matrix of size NxN using math."""
    dct_mat = [[0]*N for _ in range(N)]
    for k in range(N):
        for n in range(N):
            if k == 0:
                alpha = math.sqrt(1/N)
            else:
                alpha = math.sqrt(2/N)
            dct_mat[k][n] = alpha * math.cos(math.pi * k * (2 * n + 1) / (2 * N))
    return dct_mat


def discrete_cosine_transform(u, axis=-1):
    if axis != -1:
        u = u.transpose(-1, axis)

    n = u.shape[-1]
    D = create_dct_ii_matrix(n)  # Ensure this function returns a torch.Tensor if mixing with PyTorch, or keep as list and convert
    D = torch.tensor(D, dtype=torch.float32, device=u.device)  # Convert list to tensor properly
    y = torch.matmul(u, D)  # Use matmul for clarity and compatibility
    
    if axis != -1:
        y = y.transpose(-1, axis)
        
    return y


class DCT1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.v = torch.nn.Parameter(torch.rand(16))
        self.T = torch.nn.Parameter(torch.ones(16)*0.1)
    def forward(self, x):
        # Apply the discrete cosine transform directly
        dct_coeffs = discrete_cosine_transform(x)
        dct_coeffs = self.v*hardthreshold(dct_coeffs, self.T)
        
        return dct_coeffs


   
class DCTNN(nn.Module):
    def __init__(self):
        super(DCTNN, self).__init__()
        # Integrate the custom Downsample layer
        self.DownsampleLayer1 = DownsampleLY(learnable=True)
        self.DownsampleLayer2 = DownsampleLY(learnable=True)
        self.encoder = nn.Sequential(
            nn.Linear(256, 16),#52
            #nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        self.dct_layer = DCT1D()#dct layer
        self.decoder = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    
    def forward(self, x):
        R = self.DownsampleLayer1(x)
        Z = self.DownsampleLayer2(x)
        x = torch.cat((R,Z), dim=1)
        x = self.encoder(x)
        x = self.dct_layer(x)
        x = self.decoder(x)
        return x
    
    
