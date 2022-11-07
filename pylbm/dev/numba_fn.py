# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 21:55:22 2022

@author: Ted
"""
import numpy as np
from numba import njit,float64,int8

@njit()
def calcFeq(ueq,c,w,rho):  # calc the equilibrium distribution
    nz,ny,nx,nphase,dum=ueq.shape
    dum,ndir=c.shape
    Feq=np.array((nz,ny,nx,nphase,ndir),dtype=float64)
    for pp in range(nphase):
        u2c=1.5*(ueq[...,pp,:]**2).sum(axis=-1)
        #k0,k1,k2=self.flowIdx[pp]
        for ii in range(ndir):
            cuns=3*(c[0,ii]*ueq[...,pp,0]+c[1,ii]*ueq[...,pp,1]+c[2,ii]*ueq[...,pp,2])
            #if(k0[0].size): self.fields.Feq[k0+(pp,ii,)]=self.w[ii]*self.fields.rho[k0+(pp,)]
            #if(k1[0].size): self.fields.Feq[k1+(pp,ii,)]=self.w[ii]*self.fields.rho[k1+(pp,)]*(1+cuns[k1])
            #if(k2[0].size): self.fields.Feq[k2+(pp,ii,)]=self.w[ii]*self.fields.rho[k2+(pp,)]*(1+cuns[k2]+0.5*cuns[k2]**2-u2c[k2])
            #self.fields.Feq[...,pp,ii]=self.w[ii]*self.fields.rho[...,pp]*(1+cuns+0.5*np.power(cuns,2)-u2c)
            Feq[...,pp,ii]=w[ii]*rho[...,pp]*(1+cuns)