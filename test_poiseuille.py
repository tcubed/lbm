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
import pandas as pd

import unittest

# custom 
import importlib
import pylbm
importlib.reload(pylbm)
import boundaryConditions as BC
importlib.reload(BC)
import callbacks
importlib.reload(callbacks)
import plotlbm

np.set_printoptions(formatter=dict(float=lambda x: "%.5g"%x))


def simSetup(dirx,dPdL,nu,driven='pressure',channelWidth=10):
    
    if(dirx=='we'):
        ny=channelWidth;nx=ny*3
    elif(dirx=='sn'):
        nx=channelWidth;ny=nx*3
    else:
        raise Exception('not supported (dirx=%s)'%dirx)
    
    nz=1
    # WALLS
    solid=np.zeros((nz,ny,nx,1));
    if(dirx=='we'):
        solid[:,0,:]=1;solid[:,-1,:]=1
    elif(dirx=='sn'):
        solid[:,:,0]=1;solid[:,:,-1]=1
    
    # physics
    tau=3*nu+.5
    
    # instance
    S=pylbm.LBM((nz,ny,nx))
    S.fields['tau'][...]=tau
    # assign solid
    S.fields['ns']=solid;
    S.history={'file':'history.csv'}
    
    
    if(driven=='pressure'):
        # INLET/OUTLET
        inlet=np.zeros((nz,ny,nx,1))
        outlet=np.zeros((nz,ny,nx,1))
        if(dirx=='we'):
            inlet[:,1:-1,0]=1
            outlet[:,1:-1,-1]=1
            dp=dPdL*(nx-1)*3
        elif(dirx=='sn'):
            inlet[:,0,1:-1]=1
            outlet[:,-1,1:-1]=1
            dp=dPdL*(ny-1)*3
    
        ki=np.where(inlet)
        ko=np.where(outlet)
        def cb_pres(self):
            if(dirx=='we'):
                self.fields['Fin']=BC.zhouHePressure(self.fields['Fin'], fromdir='w', k=ki, rho0=1+dp/2)
                self.fields['Fin']=BC.zhouHePressure(self.fields['Fin'], fromdir='e', k=ko, rho0=1-dp/2)
            elif(dirx=='sn'):
                self.fields['Fin']=BC.zhouHePressure(self.fields['Fin'], fromdir='s', k=ki, rho0=1+dp/2)
                self.fields['Fin']=BC.zhouHePressure(self.fields['Fin'], fromdir='n', k=ko, rho0=1-dp/2)
        cb={#'init':[cb_pres],
            'postStream':[cb_pres],
            'postMacro':[],
            'final':[]}
    elif(driven=='gravity'):
        S.fields['gravity']=np.zeros((*S.dim,1,3))
        if(dirx=='sn'):
            S.fields['gravity'][...,1]=dPdL
        elif(dirx=='we'):
            S.fields['gravity'][...,2]=dPdL
        else:
            raise Exception('"%s" not supported'%dirx)
        cb={'postStream':[],
            'postMacro':[],
            'postUeq':[callbacks.ueqForcingHSD],
            'postFeq':[callbacks.feqForcingHSD],
            'final':[]}
        # cb={'postStream':[],
        #     'postMacro':[],
        #     'postUeq':[callbacks.ueqForcingSC],
        #     'postFeq':[],
        #     'final':[]}
    else:
        raise Exception('not supported')
    cb['postMacro'].append(callbacks.history)
    cb['final'].append(callbacks.history)
    
    
    S.initDistribution();
    
    #S.sim(steps=200,callbacks=cb)
    return S,cb

def postprocessing(S,dirx):
    
    nz,ny,nx=S.dim
    mx,my=np.meshgrid(range(nx),range(ny));
    
    plt.figure(figsize=(6,6))
    plt.subplot(1,2,1)
    if(dirx=='we'):
        plt.imshow(S.fields['u'][0,...,2],origin='lower');
    elif(dirx=='sn'):
        plt.imshow(S.fields['u'][0,...,1],origin='lower');

    plt.axis('off')
    vx=S.fields['u'][0,...,2]
    vy=S.fields['u'][0,...,1]
    plt.streamplot(mx,my,vx,vy,density=1,linewidth=1,color=(1,0,0,.5))
    
    plt.subplot(2,2,2)
    #if(dirx=='')
    if(dirx in ['we','ew']):
        plt.plot(S.fields['u'][0,:,nx//2,2])
    elif(dirx in ['sn','ns']):
        plt.plot(S.fields['u'][0,ny//2,:,1])
        
    plt.subplot(2,2,4)
    df=pd.read_csv('history.csv')
    plt.plot(df['step'],df['maxu']);plt.title('maxv')

    plt.tight_layout()

    # %
    #plt.figure()
    #plotlbm.plotf(S.fields['Fin'][0,:,:,0,:],origin='upper')


class TestPoiseuille(unittest.TestCase):
    
    def test_pressureSN(self):
        # northbound pressure
        # pick nx, Re
        Re=10
        nx=11
        
        dirx='sn'
        Ma=0.1
        rho=1
        umax=Ma
        # poiseuille
        uavg=2/3*umax
        nu=(nx-1)*uavg/Re
        mu=nu*rho
        # driving force
        dPdL=umax*2*mu/((nx-2)/2)**2
        
        S,cb=simSetup(dirx,dPdL,nu,channelWidth=nx,driven='pressure')
        S.sim(steps=1000,callbacks=cb)
        
        # post-processing
        umax1=S.fields['u'].max()
        postprocessing(S,dirx=dirx)
        
        rerr=(umax1-umax)/umax
        print('test_pressureSN: umax %.4g vs %.4g (analytical): %.2f%% error'%(umax1,umax,rerr*100))
        self.assertAlmostEqual(umax1,umax,delta=0.002)
        
    def test_pressureWE(self):
        # eastbound pressure
        # pick nx, Re
        Re=10
        nx=11
        
        dirx='we'
        Ma=0.1
        rho=1
        umax=Ma
        # poiseuille
        uavg=2/3*umax
        nu=(nx-1)*uavg/Re
        mu=nu*rho
        # driving force
        dPdL=umax*2*mu/((nx-2)/2)**2
        
        S,cb=simSetup(dirx,dPdL,nu,channelWidth=nx,driven='pressure')
        S.sim(steps=1000,callbacks=cb)
        
        # post-processing
        umax1=S.fields['u'].max()
        postprocessing(S,dirx=dirx)
        rerr=(umax1-umax)/umax
        print('test_pressureWE: umax %.4g vs %.4g (analytical): %.2f%% error'%(umax1,umax,rerr*100))
        self.assertAlmostEqual(umax1,umax,delta=0.002)

    def test_gravitySN(self):
        # northbound gravity
        # pick nx, Re
        Re=10
        nx=11
        
        dirx='sn'
        Ma=0.1
        rho=1
        umax=Ma
        # poiseuille
        uavg=2/3*umax
        nu=(nx-1)*uavg/Re
        mu=nu*rho
        # driving force
        dPdL=umax*2*mu/((nx-2)/2)**2
        
        S,cb=simSetup(dirx,dPdL,nu,channelWidth=nx,driven='gravity')
        S.sim(steps=1000,callbacks=cb)
        
        # post-processing
        umax1=S.fields['u'].max()
        postprocessing(S,dirx=dirx)
        
        rerr=(umax1-umax)/umax
        print('test_gravitySN: umax %.4g vs %.4g (analytical): %.2f%% error'%(umax1,umax,rerr*100))
        self.assertAlmostEqual(umax1,umax,delta=0.002)
    
    def test_gravityWE(self):
        # northbound gravity
        # pick nx, Re
        Re=10
        nx=11
        
        dirx='we'
        Ma=0.1
        rho=1
        umax=Ma
        # poiseuille
        uavg=2/3*umax
        nu=(nx-1)*uavg/Re
        mu=nu*rho
        # driving force
        dPdL=umax*2*mu/((nx-2)/2)**2
        
        S,cb=simSetup(dirx,dPdL,nu,channelWidth=nx,driven='gravity')
        S.sim(steps=1000,callbacks=cb)
        
        # post-processing
        umax1=S.fields['u'].max()
        postprocessing(S,dirx=dirx)
        
        rerr=(umax1-umax)/umax
        print('test_gravityWE: umax %.4g vs %.4g (analytical): %.2f%% error'%(umax1,umax,rerr*100))
        self.assertAlmostEqual(umax1,umax,delta=0.002)
        
def u(G,mu,a,x):
    # per Sukop, pg 8
    return G/(2*mu)*(a**2-x**2)

# %% UNIT TEST
if(__name__=='__main__'):
    unittest.main()

# %%
#dirx='we'

#dPdL=1e-2/40
#nu=1/6.



# %%
# dirx='sn'
# Ma=0.1
# rho=1
# umax=Ma
# # poiseuille
# uavg=2/3*umax


# # Re=10
# # 

# # nu=1/6.
# # nu=0.093333333333333



# # mu=nu/rho

# # two_a=Re*nu/uavg
# # a=(two_a/2)
# # nx=round(two_a)+1

# # pick nx, Re
# Re=10
# nx=15
# nu=(nx-1)*uavg/Re
# mu=nu*rho

# dPdL=umax*2*mu/((nx-2)/2)**2

# #g=3*uavg*nu/a**2
# #dPdL=g*rho

# S,cb=simSetup(dirx,dPdL,nu,channelWidth=nx,driven='pressure')
# S.sim(steps=1000,callbacks=cb)

# %%
#postprocessing(S,dirx=dirx)

# %%

#dPdL2=3*uavg*mu/((nx-2)/2)**2
# # %%

# umax=0.1
# uavg=2/3*umax
# # gravity
# #G=3uavg*mu/a^2=rho.g --> g=3uavg*nu/a^2
# # using definition of Re: 2a=Re*nu/u
# two_a=Re*nu/uavg
# a=(two_a/2)

# g=3*uavg*nu/a**2

# # %% pressure
# G=dPdL
# uavg=G*a**2/3/mu

# # %%


# G=dPdL
# #mu=
# uavg_theo=2/3*G/(2*mu)*a**2