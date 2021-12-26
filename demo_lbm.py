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

# matplotlib and animation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#https://stackoverflow.com/questions/42634997/how-do-i-properly-enable-ffmpeg-for-matplotlib-animation
plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\Ted\MyApps\FFmpeg\bin\ffmpeg.exe'
from matplotlib.animation import FFMpegWriter

# custom 
import importlib
import pylbm
importlib.reload(pylbm)

def plotf(F):
    """plot distribution"""
    pp=[(1,5),(2,1),(3,4),(4,2),(5,8),(6,0),(7,6),(8,3),(9,7)]
    for ip,ii in pp:
        plt.subplot(3,3,ip);plt.imshow(F[:,:,ii]);plt.axis('off');
#
#a=np.array(list(range(24))).reshape(2,3,4)
#if(__name__=='__main__'):


v0=0.1
def cb_velbc(self):
    """velocity boundary condition"""
    #v=v0*np.minimum(1,self.step/1000)
    self.v[:,0,1]=v0
    
def cb_mov(self):
    """movie callback"""
    if(self.step%20!=0): return
    
    # cv2
    img = cv2.normalize(self.v[:,:,1], None, 
                        alpha=0, beta=255, 
                        norm_type=cv2.NORM_MINMAX, 
                        dtype=cv2.CV_8U)
    self.mov.write(img)
    
    # ffmpeg
    plt.clf()
    plt.imshow(img);plt.axis('off');plt.title(self.step)
    plt.streamplot(mx,my,self.v[:,:,1],self.v[:,:,0],density=.5)
    self.mov2.grab_frame()
    
    #fig2=plt.figure()
    #plt.plot(S.v[ny//2,:,1])
    

ny=201;nx=801;

# define solid
xc=160#nx/4;
yc=100#ny/2;
fd=.2
R=20#ny*fd
mx,my=np.meshgrid(range(nx),range(ny));
r=((mx-xc)**2+(my-yc)**2)**0.5;
k=np.where(r<=R)
solid=np.zeros((ny,nx));solid[k]=1;
solid[0,:]=1;solid[-1,:]=1
omega=1.9
nu=1/3*(1/omega-.5)
Re=v0*(2*ny*fd)/nu
print('Re: %.3g'%Re)
# %%
nu=0.01
omega=1/(3*nu+.5)
S=pylbm.LBM((ny,nx))
S.omega=omega
# assign solid
S.solid=solid;
# init velocity (& particle distribution)
S.v[:,:,1]=v0
#S.calcfeq();S.F=S.Feq.copy();
S.initDistribution();



# MJPG, DIVX, mp4v, XVID, X264 
fourcc=cv2.VideoWriter_fourcc(*'MP4V');fps=30
S.mov= cv2.VideoWriter('cv2_v.mp4',fourcc, fps, (nx,ny),False)

#https://matplotlib.org/stable/gallery/animation/frame_grabbing_sgskip.html
f1 = plt.figure()
metadata = dict(title='lbm-v', artist='lbm',comment='Movie!')
S.mov2 = FFMpegWriter(fps=fps, metadata=metadata)
dpi=100
with S.mov2.saving(f1, "mpl_v.mp4", dpi):

    S.sim(steps=50000,callbacks=[cb_velbc,cb_mov])

S.mov.release()

# %%
plt.figure(figsize=(12,3))
plt.subplot(1,3,1);plt.imshow(S.rho);plt.axis('off');plt.title('rho')
plt.streamplot(mx,my,S.v[:,:,1],S.v[:,:,0],density=.5)
plt.subplot(1,3,2);plt.imshow(S.v[:,:,0]);plt.axis('off');plt.title('$v_y$')
plt.subplot(1,3,3);plt.imshow(S.v[:,:,1]);plt.axis('off');plt.title('$v_x$')

#plt.figure();plotf(S.Feq)
#plt.figure();plotf(S.F)

plt.figure()
plt.plot(S.v[ny//2,:,1])