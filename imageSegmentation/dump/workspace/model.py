#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import pandas
import tqdm as tqdm
import numpy as np
import torchvision
import math


# In[2]:


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


#based on arXiv:2006.15055

_e=1e-20
class SlotAttention(nn.Module):
    def __init__(self,feature_size,iterations=5,nslots=None,resolution=None,evolver=None,residual_hidden=None):
        super().__init__()
        self.nslots=nslots
        self.iterations=iterations
        self.attn_norm_div=feature_size**-.5

        self.nu=nn.Parameter(torch.rand(1,1,feature_size))
        self.sigma=nn.Parameter(torch.rand(1,1,feature_size))

        self.emb=PosEmbedder(feature_size,2*feature_size,resolution)
        self.to_q=nn.Linear(feature_size,feature_size)
        self.to_k=nn.Linear(feature_size,feature_size)
        self.to_v=nn.Linear(feature_size,feature_size)

        self.evolver=nn.GRUCell(feature_size,feature_size) if evolver is None else evolver

        self.residual=None if residual_hidden is None else nn.Sequential(
            nn.LayerNorm(feature_size),
            nn.Linear(feature_size,residual_hidden),
            nn.ReLU(),
            nn.Linear(residual_hidden,feature_size),
            nn.ReLU(),
        )

        self.pre_norm=nn.LayerNorm(feature_size)
        self.slot_norm=nn.LayerNorm(feature_size)



    def forward(self,inputs,slots=None,nslots=None):
        batch_size,n_tokens,feature_size=inputs.shape
        nslots=nslots if nslots is not None else self.nslots

        slots=slots if slots is not None else torch.normal(self.nu.expand(batch_size,nslots,-1),self.sigma.expand(batch_size,nslots,-1))

        inputs=self.pre_norm(inputs)
        k,v=self.to_k(inputs),self.to_v(inputs)
        for i in range(self.iterations):
            slots=self.slot_norm(slots)
            q=self.to_q(slots)
            attn=torch.softmax(self.attn_norm_div*torch.einsum("bsf,bif->bsi",q,k),dim=1)+_e
            attn=attn/attn.sum(dim=-1,keepdim=True)
            updates=torch.einsum("bnf,bsn->bsf",v,attn)
            slots=self.evolver(updates.reshape(-1,feature_size),slots.reshape(-1,feature_size)).reshape(batch_size,nslots,feature_size)
            if self.residual is not None:
                slots=slots+slots.residual(slots)
        return slots
    def get_attn(self,inputs,slots): #dim(slots,inputs)
        inputs=self.pre_norm(inputs)
        k,v,q=self.to_k(inputs),self.to_v(inputs),self.to_q(slots)
        attn=torch.softmax(self.attn_norm_div*torch.einsum("bsf,bif->bsi",q,k),dim=1)+_e
        attn=attn/attn.sum(dim=-1,keepdim=True)
        return attn



class PosEmbedder(nn.Module):
    def __init__(self,feature_size,hidden_size,resolution=None):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(4,hidden_size),
            nn.Sigmoid(), #my personal experience tells me that sigmoid encodes the infromation better in smaller networks with io in the range [0,1]
            nn.Linear(hidden_size,feature_size),
        )        
    def forward(self,features):
        b,c,h,w = features.shape
        ly=torch.linspace(0,1,h,device=features.device)
        lx=torch.linspace(0,1,w,device=features.device)
        my,mx=torch.meshgrid(ly,lx,indexing='ij')
        grid=torch.stack([my,mx,1-my,1-mx],dim=-1).unsqueeze(0).expand(b,-1,-1,-1)
        emb=self.layers(grid).permute(0,3,1,2)
        return features+emb

class CnnEncoder(nn.Module): #multiple 8
    def __init__(self,in_channels=3,out_channels=64):
        super().__init__()
        self.seq=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,8,padding=7),
            nn.LeakyReLU(),
            nn.Dropout2d(.2),
            nn.Conv2d(out_channels,out_channels,8,padding=7),
            nn.LeakyReLU(),
            nn.Dropout2d(.3),
            nn.Conv2d(out_channels,out_channels,8,padding=7),
            nn.LeakyReLU(),
            nn.Dropout2d(.5),
            nn.Conv2d(out_channels,out_channels,8,stride=8),
            nn.LeakyReLU(),
        )
    def forward(self,image):
        return self.seq(image)

class CnnDecoder(nn.Module):
    def __init__(self,feature_size=64,base_resolution=(10,10),resolution=(320,480),stepsize=4,internal_channels=32,out_channels=4):
        super().__init__()
        bw,bh=base_resolution
        self.base=base_resolution
        self.target=resolution
        self.internal_channels=internal_channels
        w,h=self.w,self.h=resolution
        steps=math.ceil(math.log(max(w/bw,h/bh),stepsize))
        self.final=(base_resolution[0]*(stepsize**steps),base_resolution[1]*(stepsize**steps))
        self.projection=nn.Sequential(
            nn.Linear(feature_size,feature_size),
            nn.LeakyReLU(),
            nn.Linear(feature_size,internal_channels*bw*bh),
            nn.LeakyReLU(),
            nn.Linear(internal_channels*bw*bh,internal_channels*bw*bh),
            nn.LeakyReLU(),
        )
        self.seq=nn.Sequential()
        for i in range(0,steps):
            self.seq.append(nn.UpsamplingBilinear2d(scale_factor=stepsize))
            self.seq.append(nn.Dropout2d(0.2))
            self.seq.append(nn.Conv2d(internal_channels,internal_channels,stepsize,padding=stepsize-1))
            self.seq.append(nn.LeakyReLU())
            self.seq.append(nn.Conv2d(internal_channels,internal_channels,stepsize,padding=stepsize-1))
            self.seq.append(nn.LeakyReLU())
        self.seq.append(nn.Conv2d(internal_channels,out_channels,stepsize,padding=stepsize-1))
    def forward(self,token):
        b,s,_=token.shape
        next=self.seq(self.projection(token).reshape(-1,self.internal_channels,self.base[0],self.base[1]))[:,:,(self.final[0]-self.target[0])// 2:(self.final[0]-self.target[0])//2+self.target[0],(self.final[1]-self.target[1])//2:(self.final[1]-self.target[1])//2+self.target[1]]
        return next.reshape(b,s,*(next.shape[-3:]))

class ImageMerger(nn.Module):
    def __init__(self,has_img=True):
        super().__init__()
        self.has_img=has_img
    def forward(self,images,has_img=None):
        #batch slot channels x y
        _,slots,_,_,_=images.shape
        has_img = self.has_img if has_img is None else has_img
        alpha=images[:,:,0,:,:]
        alpha=torch.softmax(alpha,1)
        return (alpha[:,:,None,:,:]*(torch.sigmoid(images[:,:,1:,:,:]) if has_img else torch.arange(start=1,end=slots+1).to(alpha.device)[None,None,-1,None,None]*50)).sum(dim=1)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        feature_size=64,
        n_slots=10,
        slot_iterations=3,
        base_resolution=(5, 8),
        target_resolution=(320, 480),
        stepsize=4,
        internal_channels=8,
        residual_hidden=None,
        just_alpha=False,
        evolver=None,
        CustomEncoder=None
    ):
        super().__init__()
        self.encoder = CnnEncoder(in_channels=in_channels, out_channels=feature_size) if CustomEncoder is None else CustomEncoder
        self.posemb=PosEmbedder(feature_size=feature_size,resolution=target_resolution,hidden_size=feature_size)
        self.slot_attention = SlotAttention(
            feature_size=feature_size,
            iterations=slot_iterations,
            nslots=n_slots,
            resolution=base_resolution,
            evolver=evolver,
            residual_hidden=residual_hidden)
        self.decoder = CnnDecoder(
            feature_size=feature_size,
            base_resolution=base_resolution,
            resolution=target_resolution,
            stepsize=stepsize,
            internal_channels=internal_channels,
            out_channels=in_channels+1 if not just_alpha else 1)
        self.merger = ImageMerger(has_img=not just_alpha)
        self.feature_size = feature_size
        self.base_resolution = base_resolution
        self.target_resolution = target_resolution
        self.just_alpha = just_alpha
    def forward(self,batch,get_masks=False):
        batch_size=batch.shape[0]
        batch=self.encoder(batch)
        batch=self.posemb(batch)
        batch=self.slot_attention(batch.reshape(batch_size,-1,self.feature_size))
        batch=self.decoder(batch)
        if get_masks:
            return self.merger(batch),torch.softmax(batch[:,:,0,:,:],1)
        return self.merger(batch)




# In[4]:


# import tqdm
# auenc=AutoEncoder().to(device)
# for i in tqdm.trange(100):
#     imgs=torch.rand(4,3,320,480).to(device)
#     o=auenc(imgs)
#     #print(o.shape)


# In[5]:


#https://gist.github.com/ivanstepanovftw/e079280976f2466e58f3200e14d9e3a1
#Ivan Stepanov
#Lincesed under MIT No Attribution
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
def match_targets(outputs, targets):
    cost_matrix = torch.cdist(outputs, targets, p=1)
    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())
    return row_ind, col_ind


def hungarian_loss(outputs, targets):
    row_ind, col_ind = match_targets(outputs, targets)
    matched_outputs = outputs[row_ind]
    matched_targets = targets[col_ind]
    loss = F.l1_loss(matched_outputs, matched_targets)
    return loss


# In[ ]:




def batch_matrix_hungarian_loss(outputs,target):
    batch,slots,data=outputs.shape[0],outputs.shape[1],outputs.shape[2:]
    tbatch,tslots,tdata=outputs.shape[0],outputs.shape[1],outputs.shape[2:]
    assert batch==tbatch      #"batchsize needs to be equal"
    assert data==tdata        #"the shape of the data needs to be the same"
    loss=0
    for i in range(batch):
        loss=loss+hungarian_loss(torch.flatten(outputs[0],1),torch.flatten(target[0],1))
    return loss