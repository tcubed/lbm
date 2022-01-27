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
import pylbm
importlib.reload(pylbm)
import boundaryConditions as BC


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
    

nx=801;ny=201;
scl=5
nx=nx//scl;ny=ny//scl;

mx,my=np.meshgrid(range(nx),range(ny));
solid=np.zeros((ny,nx));
solid[0,:]=1;solid[-1,:]=1

inlet=np.zeros((ny,nx))
#inlet[:,0]=1
inlet[1:-1,0]=1
outlet=np.zeros((ny,nx))
outlet[1:-1,-1]=1
#outlet[:,-1]=1

ki=np.where(inlet)
ko=np.where(outlet)
dp=1e-3
def cb_pres(self):
    self.F=BC.zhouHePressure(self.F, fromdir='w', k=ki, rho0=1+dp/2)
    #self.F=BC.zhouHePressure(self.F, fromdir='e', k=ko, rho0=1-dp/2)
    pass

def cb_vel(self):
    """velocity boundary condition"""
    
    #
    # try this right after stream, before storing in bounceback?
    #
    #v=v0*np.minimum(1,self.step/1000)
    #self.v[:,0,0]=.1
    self.v[:,-1,0]=self.v[:,-2,0]
    
def cb_viz(self):
    if(self.step%20>0): return
    ugrad = np.gradient(self.v[:, :, 0])
    vgrad = np.gradient(self.v[:, :, 1])
    vor = ugrad[1] - vgrad[0]
    #print('vor:',vor.min(),vor.max())
    #plt.imshow(vor,vmin=-1e-4,vmax=1e-4)
    #plt.show()
    colors = [(1, 1, 0), (0.953, 0.490, 0.016), (0, 0, 0),
        (0.176, 0.976, 0.529), (0, 1, 1)]
    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'my_cmap', colors)
    vor_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(
        vmin=-0.01, vmax=0.01),cmap=my_cmap).to_rgba(vor)
    
    # cv2
    img = cv2.normalize(vor_img, None, 
                        alpha=0, beta=255, 
                        norm_type=cv2.NORM_MINMAX, 
                        dtype=cv2.CV_8U)
    #img=(500*vor_img[:,:,:3]+127).astype(np.uint8)
    self.mov.write(img[:,:,:3])
    
    
# define solid
if(1):
    xc=nx//5;yc=ny//2;R=20//scl
    r=((mx-xc)**2+(my-yc)**2)**0.5;
    k=np.where(r<=R)
    solid[k]=1;
#fd=.2
#R=20#ny*fd
#
#
#

#omega=1.5
#nu=1/3*(1/omega-.5)
#Re=v0*(2*ny*fd)/nu
#print('Re: %.3g'%Re)
# %%
nu=.01
omega=1/(3*nu+.5)
S=pylbm.LBM((ny,nx))
S.omega=omega
# assign solid
S.solid=solid;
# init velocity (& particle distribution)
#S.v[:,:,1]=v0
#S.calcfeq();S.F=S.Feq.copy();
S.initDistribution();

# %%


cb={'postStream':[cb_pres],'postMacro':[cb_vel]}
S.sim(steps=100,callbacks=cb)

# %%
def myplot(S):
    ugrad = np.gradient(S.v[:, :, 0])
    vgrad = np.gradient(S.v[:, :, 1])
    vor = ugrad[1] - vgrad[0]
    
    colors = [(1, 1, 0), (0.953, 0.490, 0.016), (0, 0, 0),
        (0.176, 0.976, 0.529), (0, 1, 1)]
    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'my_cmap', colors)
    vor_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(
        vmin=-0.01, vmax=0.01),cmap=my_cmap).to_rgba(vor)
    
    nrow=4
    plt.figure(figsize=(8,8))
    plt.subplot(nrow,2,1)
    plt.imshow(S.rho,origin='lower');plt.title('rho');plt.colorbar()
    plt.subplot(nrow,2,3)
    plt.imshow(S.v[:,:,0],origin='lower');plt.title('v0');plt.colorbar()
    plt.subplot(nrow,2,5)
    plt.imshow(S.v[:,:,1],origin='lower');plt.title('v1');plt.colorbar()
    plt.subplot(nrow,1,4)
    #plt.imshow(vor,origin='lower',vmin=-.01,vmax=0.01);plt.title('vor');plt.colorbar()
    #plt.subplot(nrow,1,5)
    plt.imshow(vor_img,origin='lower');plt.title('vor');plt.axis('off')
    
    plt.subplot(nrow,2,2);plt.plot(S.rho[ny//2,:])
    plt.subplot(nrow,2,4);plt.plot(S.v[ny//2,:,0])
    plt.subplot(nrow,2,6);plt.plot(S.v[ny//2,:,1])
    plt.tight_layout()
myplot(S)
# %%
# MJPG, DIVX, mp4v, XVID, X264 
fourcc=cv2.VideoWriter_fourcc(*'MP4V');fps=30
S.mov= cv2.VideoWriter('cv2_vor.mp4',fourcc, fps, (nx,ny),isColor=True)

cb={'postStream':[cb_pres],
    'postMacro':[cb_vel,cb_viz]}
S.sim(steps=10000,callbacks=cb)
S.mov.release()

# %%
myplot(S)
# #https://matplotlib.org/stable/gallery/animation/frame_grabbing_sgskip.html
# f1 = plt.figure()
# metadata = dict(title='lbm-v', artist='lbm',comment='Movie!')
# S.mov2 = FFMpegWriter(fps=fps, metadata=metadata)
# dpi=100
# with S.mov2.saving(f1, "mpl_v.mp4", dpi):

#     S.sim(steps=5000,callbacks=[cb_pres,cb_mov])

# S.mov.release()

# # %%
# plt.figure(figsize=(12,3))
# plt.subplot(1,3,1);plt.imshow(S.rho);plt.axis('off');plt.title('rho')
# plt.streamplot(mx,my,S.v[:,:,1],S.v[:,:,0],density=.5)
# plt.subplot(1,3,2);plt.imshow(S.v[:,:,0]);plt.axis('off');plt.title('$v_y$')
# plt.subplot(1,3,3);plt.imshow(S.v[:,:,1]);plt.axis('off');plt.title('$v_x$')

# #plt.figure();plotf(S.Feq)
# #plt.figure();plotf(S.F)

# plt.figure()
# plt.plot(S.v[ny//2,:,1])