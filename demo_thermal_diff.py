# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 17:19:11 2022

@author: Ted
"""

import matplotlib.pyplot as plt
import importlib

import pylbm2 as pylbm
importlib.reload(pylbm)

ny=10
nx=30
S=pylbm.LBM((ny,nx),nphase=1)

S.fields['ns'][:,:,0]=0
S.fields['flowMode']*=0;

def cb_rho(self):
    self.fields['rho'][:,0,0]=1.1
    self.fields['rho'][:,-1,:]=.2
    
S.sim(steps=2000,callbacks={'postMacro':[cb_rho]})

# %%
plt.figure(figsize=(12,6))
plt.subplot(2,2,1)
plt.imshow(S.fields['v'][:,:,0]);plt.colorbar();plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(S.fields['v'][:,:,1]);plt.colorbar();plt.axis('off')
plt.subplot(2,2,3)
plt.imshow(S.fields['rho']);plt.colorbar()
plt.subplot(2,2,4)
plt.plot(S.fields['rho'][ny//2,:,0])
