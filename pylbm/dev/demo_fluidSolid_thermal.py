# -*- coding: utf-8 -*-
"""
Demonstration of single-phase fluid flow through channel, heating up
a solid part of the domain.
@author: Ted Tower
"""
import numpy as np
import matplotlib.pyplot as plt
import importlib

import pylbm2 as pylbm
import boundaryConditions as bc
importlib.reload(pylbm)

ny=30
nx=30
wthk=10

# walls, inlets, outlets
solid=np.zeros((ny,nx));
solid[:wthk,:]=1
k1=np.where(solid)
k0=np.where(1-solid)

inlet=np.zeros((ny,nx));inlet[wthk:-1,0]=1
kin=np.where(inlet)
outlet=np.zeros((ny,nx));outlet[wthk:-1,-1]=1
kout=np.where(outlet)

# instance simulation
S=pylbm.LBM((ny,nx),nphase=2)
S.fluidPhases=[0]

# top/bottom walls (all distributions)
S.fields['ns'][0,:,:]=1
S.fields['ns'][-1,:,:]=1  # comment out for periodic boundary

# fluid distribution
S.fields['flowMode'][...,0]=2; # default is full NS
S.fields['ns'][k1+(0,)]=1      # solid
S.fields['invtau'][...,0]=1

# thermal distribution
# -- solid
S.fields['flowMode'][k1+(1,)]=0;  # diffusion
S.fields['invtau'][k1+(1,)]=.1
S.fields['rho'][k1+(1,)]=10
# -- fluid
S.fields['flowMode'][k0+(1,)]=1;  # advection
S.fields['invtau'][k0+(1,)]=1
S.fields['rho'][k0+(1,)]=10

# BC
def cb_postStream(self):
    # pressure-driven fluid flow
    F=self.fields['Fin'][...,0,:]
    # west
    F=bc.zhouHePressure(F,fromdir='w',k=kin,rho0=1.01)
    # east
    F=bc.zhouHePressure(F,fromdir='e',k=kout,rho0=.99)
    self.fields['Fin'][...,0,:]=F
    
    # thermal -- incoming thermal; open exit
    F=self.fields['Fin'][...,1,:]
    # west
    F=bc.zhouHePressure(F,fromdir='w',k=kin,rho0=11)
    # east
    for ii in range(self.ndir):
        F[:,-1,ii]=F[:,-2,ii]
    self.fields['Fin'][...,1,:]=F
    
    # temp-dependent viscosity
    #self.fields['invtau'][k0+(0,)]=1+1./(self.fields['rho'][k0+(1,)]+1)
    
    # south -- # uncomment for periodic boundary
    #self.fields['Fin'][-1,...]=self.fields['Fin'][-2,...]

S.initDistribution()
S.sim(steps=300,callbacks={'postStream':[cb_postStream]})

# %%
nrow=3;ncol=3
plt.figure(figsize=(12,6))
plt.subplot(nrow,ncol,1)
plt.imshow(S.fields['rho'][:,:,0]);plt.colorbar();plt.axis('off');plt.title('rho')
plt.subplot(nrow,ncol,2)
plt.imshow(S.fields['v'][:,:,0]);plt.colorbar();plt.axis('off');plt.title('$v_x$')
plt.subplot(nrow,ncol,3)
plt.imshow(S.fields['v'][:,:,1]);plt.colorbar();plt.axis('off');plt.title('$v_y$')

plt.subplot(nrow,ncol,4)
plt.plot(S.fields['rho'][20,:,0]);plt.title('rho, center of channel')
plt.subplot(nrow,ncol,5)
plt.plot(S.fields['v'][20,:,0]);plt.title('$v_x$, center of channel')
plt.subplot(nrow,ncol,6)
plt.plot(S.fields['v'][:,nx//2,0]);plt.title('$v_x$, cross-section')

plt.subplot(nrow,ncol,7)
plt.imshow(S.fields['rho'][:,:,1]);plt.colorbar();plt.axis('off');plt.title('thermal')
plt.subplot(nrow,ncol,8)
plt.plot(S.fields['rho'][20,:,1]);plt.title('thermal, center of channel')
plt.subplot(nrow,ncol,9)
plt.plot(S.fields['rho'][:,nx//2,1]);plt.title('thermal, cross-section')
plt.tight_layout()
