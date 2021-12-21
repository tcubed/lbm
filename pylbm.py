# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 19:35:11 2021

https://exolete.com/lbm/
simple 2D LBM
@author: Ted
"""
import numpy as np
import time
class LBM():
    def __init__(self,sizeyx):
        ny,nx=sizeyx
        self.dim=(ny,nx)
        self.c=np.array([[0,1,0,-1,1,1,-1,-1,0],
                         [1,0,-1,0,1,-1,-1,1,0]]);
        t1=4/9;t2=1/9;t3=1/36;
        self.w=np.array([t2,t2,t2,t2,t3,t3,t3,t3,t1])
        self.ndir=9
        self.omega=1.
        
        # init velocity, density, and distribution fields
        self.v=np.zeros((ny,nx,2))
        self.rho=np.ones((ny,nx))
        self.initDistribution();
        
        self.solid=np.zeros((ny,nx))
        self.toreflect=[0,1,2,3,4,5,6,7,8]
        self.reflected=[2,3,0,1,6,7,4,5,8]
        self.bounced=self.F.copy()
    
    def initDistribution(self):
        self.Feq=np.zeros((*self.dim,self.ndir))
        self.calcfeq();
        self.F=self.Feq.copy()
        
    def calcfeq(self):
        c_squ=1/3;
        u2c=(self.v[:,:,0]**2+self.v[:,:,1]**2)/(2*c_squ);
        for ii in range(self.ndir):
            cuns=(self.c[0,ii]*self.v[:,:,0]+self.c[1,ii]*self.v[:,:,1])/c_squ
            self.Feq[:,:,ii]=self.w[ii]*self.rho*(1+cuns+0.5*cuns**2-u2c)
                
    def calcMacro(self):
        self.rho=self.F.sum(axis=2)
        with np.errstate(invalid='ignore'):
            self.v[:,:,0]=((self.F[:,:,1]+self.F[:,:,4]+self.F[:,:,5])-
                          (self.F[:,:,3]+self.F[:,:,6]+self.F[:,:,7]))/self.rho
            self.v[:,:,1]=((self.F[:,:,0]+self.F[:,:,4]+self.F[:,:,7])-
                          (self.F[:,:,2]+self.F[:,:,5]+self.F[:,:,6]))/self.rho
    def stream(self):
        for ii in range(self.ndir):
            self.F[:,:,ii]=np.roll(self.F[:,:,ii],
                                   (self.c[0,ii],self.c[1,ii]),
                                   axis=(0,1))
    def collide(self):
        self.F=self.omega*self.Feq+(1-self.omega)*self.F;
    def bounceback(self,ON):
        for dd in range(self.ndir):
            F=self.F[:,:,dd]
            B=self.bounced[:,:,self.reflected[dd]]
            F[ON]=B[ON]
            self.F[:,:,dd]=F#self.bounced[:,:,self.reflected[dd]]
    def sim(self,steps=10,callbacks=None):
        if(callbacks is None): callbacks=[]
        ON=np.where(self.solid)
        t0=time.time();
        tepoch=t0
        for ii in range(steps):
            self.step=ii
            if((ii>0) and (ii%100==0)):
                tnow=time.time()
                telapsed=tnow-t0
                mlups=np.prod(S.rho.shape)*ii/1e6/telapsed
                print('%d: %.3gmlups (%.1fsec/epoch)'%(ii,mlups,tnow-tepoch))
                tepoch=tnow
            self.stream()
            for dd in range(self.ndir):
                self.bounced[:,:,dd]=self.F[:,:,self.toreflect[dd]]
            self.calcMacro()
            # BC
            # -- solids
            self.rho[ON]=0;
            for jj in [0,1]:
                v0=self.v[:,:,jj];v0[ON]=0;self.v[:,:,jj]=v0
            # -- callbacks
            for cb in callbacks:
                cb(self)
            self.calcfeq();
            self.collide()
            self.bounceback(ON);
            
            #prevavu=avu;avu=sum(sum(UX))/numactivenodes; ts=ts+1;
            if(np.any(S.rho>10)):
                break
        print('done! (%.2fmin)'%((time.time()-t0)/60))
def plotf(F):
    pp=[(1,5),(2,1),(3,4),(4,2),(5,8),(6,0),(7,6),(8,3),(9,7)]
    for ip,ii in pp:
        plt.subplot(3,3,ip);plt.imshow(F[:,:,ii]);plt.axis('off');
#
#a=np.array(list(range(24))).reshape(2,3,4)
#if(__name__=='__main__'):
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#https://stackoverflow.com/questions/42634997/how-do-i-properly-enable-ffmpeg-for-matplotlib-animation
plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\Ted\MyApps\FFmpeg\bin\ffmpeg.exe'


import cv2
from matplotlib.animation import FFMpegWriter

v0=0.1
def cb_velbc(self):
    v=v0*np.minimum(1,self.step/200)
    self.v[:,0,1]=v
def cb_mov(self):
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
    

ny=200;nx=400;

# define solid
xc=nx/4;yc=ny/2;fd=.2
mx,my=np.meshgrid(range(nx),range(ny));
r=((mx-xc)**2+(my-yc)**2)**0.5;
k=np.where(r<(ny*fd))
solid=np.zeros((ny,nx));solid[k]=1;
#solid[0,:]=1;solid[-1,:]=1
omega=1.7
nu=1/3*(1/omega-.5)
Re=v0*(2*ny*fd)/nu
print('Re: %.3g'%Re)
# %%
S=LBM((ny,nx))
S.omega=omega
# assign solid
S.solid=solid;
# init velocity (& particle distribution)
#S.v[:,:,1]=v0
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

    S.sim(steps=2000,callbacks=[cb_velbc,cb_mov])

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