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
import matplotlib.cm as cm

# matplotlib and animation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#https://stackoverflow.com/questions/42634997/how-do-i-properly-enable-ffmpeg-for-matplotlib-animation
plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\Ted\MyApps\FFmpeg\bin\ffmpeg.exe'
from matplotlib.animation import FFMpegWriter

# custom 
import importlib
import pylbm2
importlib.reload(pylbm2)
import boundaryConditions as BC
import plotlbm

nx=60;ny=30;
scl=1
nx=nx//scl;ny=ny//scl;

#dirx='we'
dirx='ns'

# WALLS
mx,my=np.meshgrid(range(nx),range(ny));
solid=np.zeros((ny,nx,1));

# INLET/OUTLET
inlet=np.zeros((ny,nx,1))
outlet=np.zeros((ny,nx,1))
if(dirx=='we'):
    solid[0,:]=1;solid[-1,:]=1
    inlet[1:-1,0]=1
    outlet[1:-1,-1]=1
elif(dirx=='ns'):
    solid[:,0]=1;solid[:,-1]=1
    inlet[-1,1:-1]=1
    outlet[0,1:-1]=1
else:
    raise Exception('not supported')

ki=np.where(inlet)
ko=np.where(outlet)
dp=1e-2
def cb_pres(self):
    if(dirx=='we'):
        self.fields['Fin']=BC.zhouHePressure(self.fields['Fin'], fromdir='w', k=ki, rho0=1+dp/2)
        self.fields['Fin']=BC.zhouHePressure(self.fields['Fin'], fromdir='e', k=ko, rho0=1-dp/2)
    elif(dirx=='ns'):
        self.fields['Fin']=BC.zhouHePressure(self.fields['Fin'], fromdir='s', k=ki, rho0=1+dp/2)
        self.fields['Fin']=BC.zhouHePressure(self.fields['Fin'], fromdir='n', k=ko, rho0=1-dp/2)
    
# define solid
if(1):
    xc=nx//5;yc=ny//2;R=5//scl
    r=((mx-xc)**2+(my-yc)**2)**0.5;
    k=np.where(r<=R)
    solid[k]=0.1;

#solid[10:,35]=1

# %%
nu=1./6
tau=3*nu+.5
omega=1/tau
S=pylbm2.LBM((ny,nx))
#S.omega=omega
S.tau=tau
# assign solid
S.fields['ns']=solid;
# init velocity (& particle distribution)
#S.v[:,:,1]=v0
#S.calcfeq();S.F=S.Feq.copy();
S.initDistribution();

# %%
# MJPG, DIVX, mp4v, XVID, X264 
fourcc=cv2.VideoWriter_fourcc(*'MP4V');fps=30
S.mov= cv2.VideoWriter('cv2_vor.mp4',fourcc, fps, (nx,ny),isColor=True)

cb={'postStream':[cb_pres],
    'postMacro':[]}
S.sim(steps=200,callbacks=cb)
S.mov.release()

# %%
plt.figure(figsize=(6,6))
plt.subplot(2,1,1)
if(dirx=='we'):
    plt.imshow(S.fields['v'][:,:,0]);
elif(dirx=='ns'):
    plt.imshow(S.fields['v'][:,:,1]);

plt.axis('off')
vx=S.fields['v'][:,:,0]
vy=S.fields['v'][:,:,1]
plt.streamplot(mx,my,vx,vy,density=1,linewidth=1,color=(1,0,0,1))
plt.subplot(2,1,2)
plt.plot(S.fields['v'][7,:,0])
plt.plot(S.fields['v'][15,:,0])
plt.plot(S.fields['v'][22,:,0])
plt.tight_layout()

# %%
plt.figure()
plotlbm.plotf(S.fields['Fin'][:,:,0,:],origin='upper')