# -*- coding: utf-8 -*-
"""
Demo of Shan-Chen fluid-fluid interaction

https://exolete.com/lbm/
simple 2D LBM
@author: Ted
"""
import numpy as np
import os
import sys
if(os.pardir not in sys.path): sys.path.append(os.pardir)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# custom 
from pylbm import LBM
import callbacks as CB
import importlib
import pylbm
importlib.reload(pylbm)
importlib.reload(CB)

import cProfile

def myplot(x,title):
    plt.imshow(x,origin='lower');plt.colorbar();plt.title(title)

# %% Instance and configure simulator
nx=30;ny=30;nz=1
scl=1
nx=nx//scl;ny=ny//scl;
nphase=2

dim=(nz,ny,nx)
S=LBM(dim,nphase=nphase)

# specify where fluid is
S.fields['rho'][0,10:20,10:20,0]=0
S.fields['rho'][0,:,:,1]=1-S.fields['rho'][0,:,:,0]
# reinit distribution after changing rho
S.initDistribution();

# shan-chen
S.shanChen={'pairs':[[0,1]]}
S.fields['G']=np.zeros((*dim,1))+2;

# %% Simulate
S.imgStack=[]
def cb_postMacro(self):
    if(self.step%10!=0): return
    plt.subplot(2,2,1);myplot(self.fields['rho'][0,:,:,0],'rho0')
    plt.subplot(2,2,2);myplot(self.fields['rho'][0,:,:,1],'rho1')
    plt.subplot(2,2,3);myplot(self.fields['u'][0,:,:,2],'ux')
    plt.subplot(2,2,4);myplot(self.fields['u'][0,:,:,1],'uy')
    plt.tight_layout()
    plt.show()
    
# specify callbacks
cb={'postMacro':[#cb_postMacro
                 ],
    'postUeq':[CB.fluidFluidInteractionMCMP]}

cProfile.run('S.sim(steps=1000,callbacks=cb)')
#S.sim(steps=100,callbacks=cb)
