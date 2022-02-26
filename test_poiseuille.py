# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 19:35:11 2021

https://exolete.com/lbm/
simple 2D LBM
@author: Ted
"""
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import unittest

# custom 
import importlib
import pylbm2
importlib.reload(pylbm2)
import boundaryConditions as BC
import callbacks
import plotlbm

def simSetup(dirx,dPdL,nu,driven='pressure'):
    
    if(dirx=='we'):
        nx=10;ny=31;
    elif(dirx=='sn'):
        nx=31;ny=10
    else:
        raise Exception('not supported (dirx=%s)'%dirx)
    scl=1
    nx=nx//scl;ny=ny//scl;

    # WALLS
    solid=np.zeros((ny,nx,1));
    if(dirx=='we'):
        solid[0,:]=1;solid[-1,:]=1
    elif(dirx=='sn'):
        solid[:,0]=1;solid[:,-1]=1
    
    # physics
    tau=3*nu+.5
    
    # instance
    S=pylbm2.LBM((ny,nx))
    S.tau=tau
    # assign solid
    S.fields['ns']=solid;
    
    
    if(driven=='pressure'):
        # INLET/OUTLET
        inlet=np.zeros((ny,nx,1))
        outlet=np.zeros((ny,nx,1))
        if(dirx=='we'):
            solid[0,:]=1;solid[-1,:]=1
            inlet[1:-1,0]=1
            outlet[1:-1,-1]=1
            dp=dPdL*nx
        elif(dirx=='sn'):
            solid[:,0]=1;solid[:,-1]=1
            inlet[-1,1:-1]=1
            outlet[0,1:-1]=1
            dp=dPdL*ny
    
        ki=np.where(inlet)
        ko=np.where(outlet)
        def cb_pres(self):
            if(dirx=='we'):
                self.fields['Fin']=BC.zhouHePressure(self.fields['Fin'], fromdir='w', k=ki, rho0=1+dp/2)
                self.fields['Fin']=BC.zhouHePressure(self.fields['Fin'], fromdir='e', k=ko, rho0=1-dp/2)
            elif(dirx=='sn'):
                self.fields['Fin']=BC.zhouHePressure(self.fields['Fin'], fromdir='s', k=ki, rho0=1+dp/2)
                self.fields['Fin']=BC.zhouHePressure(self.fields['Fin'], fromdir='n', k=ko, rho0=1-dp/2)
        cb={'postStream':[cb_pres],
            'postMacro':[]}
    elif(driven=='gravity'):
        S.fields['gravity']=np.zeros((*S.dim,2,3))
        S.fields['gravity'][...,1]=dPdL
        cb={'postStream':[],
            'postMacro':[],
            'postUeq':[callbacks.ueqForcingHSD()]}
    else:
        raise Exception('not supported')
        
    
    
    S.initDistribution();
    
    #S.sim(steps=200,callbacks=cb)
    return S,cb

def postprocessing(S,dirx):
    
    ny,nx=S.dim
    mx,my=np.meshgrid(range(nx),range(ny));
    
    plt.figure(figsize=(6,6))
    plt.subplot(2,1,1)
    if(dirx=='we'):
        plt.imshow(S.fields['v'][:,:,0]);
    elif(dirx=='sn'):
        plt.imshow(S.fields['v'][:,:,1]);

    plt.axis('off')
    vx=S.fields['v'][:,:,0]
    vy=S.fields['v'][:,:,1]
    plt.streamplot(mx,my,vx,vy,density=1,linewidth=1,color=(1,0,0,.5))
    plt.subplot(2,1,2)
    #if(dirx=='')
    if(dirx=='we'):
        plt.plot(S.fields['v'][:,nx//2,0])
    elif(dirx=='sn'):
        plt.plot(S.fields['v'][ny//2,:,1])
    plt.tight_layout()

    # %
    #plt.figure()
    #plotlbm.plotf(S.fields['Fin'][:,:,0,:],origin='upper')


class TestPoiseuille(unittest.TestCase):
    
    def test_pressureSN(self):
        # northbound pressure
        dPdL=1e-2/60
        S,cb=simSetup('sn',dPdL,driven='pressure')
        S.sim(steps=200,callbacks=cb)
        
    def test_pressureWE(self):
        # eastbound pressure
        dPdL=1e-2/60
        S,cb=simSetup('we',dPdL,driven='pressure')
        S.sim(steps=200,callbacks=cb)

def u(G,mu,a,x):
    # per Sukop, pg 8
    return G/(2*mu)*(a**2-x**2)

# %%
#dirx='we'
dirx='sn'
dPdL=1e-2/20
nu=1/6.

S,cb=simSetup(dirx,dPdL,nu,driven='gravity')
S.sim(steps=5000,callbacks=cb)


postprocessing(S,dirx=dirx)

# %%
Re=4.4
rho=1
#nu=1/6.
mu=nu/rho
umax=0.1
uavg=2/3*umax
# gravity
#G=3uavg*mu/a^2=rho.g --> g=3uavg*nu/a^2
# using definition of Re: 2a=Re*nu/u
two_a=Re*nu/uavg
a=(two_a/2)

g=3*uavg*nu/a**2

# %% pressure
G=dPdL
uavg=G*a**2/3/mu

# %%


G=dPdL
#mu=
uavg_theo=2/3*G/(2*mu)*a**2


# %% UNIT TEST
#unittest.main()