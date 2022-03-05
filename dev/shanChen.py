# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:29:09 2022

@author: Ted
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


# matplotlib and animation
import matplotlib
matplotlib.use("Agg")
#https://stackoverflow.com/questions/42634997/how-do-i-properly-enable-ffmpeg-for-matplotlib-animation
plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\Ted\MyApps\FFmpeg\bin\ffmpeg.exe'
from matplotlib.animation import FFMpegWriter

def ffplot(S):
    plt.subplot(1,2,1);plt.imshow(S.fields['rho'][...,0]);plt.axis('off');plt.title(0)
    plt.subplot(1,2,2);plt.imshow(S.fields['rho'][...,1]);plt.axis('off');plt.title(1)
    
def fluidFluidInteraction(self):
    assert hasattr(self,'shanChen'), "sim needs 'shanChen' dict for fluidFluidInteraction"
    SC=self.shanChen
    npair=len(SC['pairs'])
    
    psi=self.fields['rho']
    V=np.zeros((*self.dim,self.nphase,2))
    for ii in range(self.ndir):
        # roll in ii direction
        shift=(-self.c[0,ii],-self.c[1,ii],0)
        d0=np.roll(psi*self.w[ii],shift,axis=(0,1,2))
        for dd in [0,1]:
            V[...,dd]+=d0*self.c[dd,ii]
    # calc accel
    A=np.zeros((*self.dim,self.nphase,2))
    for jj in range(npair):
        for ii in [0,1]:
            Gtau=self.fields['G'][...,jj]*self.fields['tau'][...,SC['pairs'][jj][ii]]
            for dd in [0,1]:
                A[...,SC['pairs'][jj][ii],dd]-=Gtau*V[...,SC['pairs'][jj][1-ii],dd]
    return A
def ueqForcingSCFluidFluid(self):
    PSI=self.fields['density'].copy()
    return self.fluidFluidPotatialAccel(PSI)

def ueqForcingSCFluidSurface(self):
    PSI=self.fields['Gads']*self.fields['tau']*self.fields['rhoWall']
    return self.fluidOtherPotentialAccel(PSI)

def ueqForcingHSD(self):
    A=self.fields['gravity']*0
    for ii in [0,1]:
        v=self.fields['rho']*self.fields['gravity'][...,ii]/2
        A[...,ii]=v
    return A
def feqForcingHSD(self):
    feq=self.fields['feq']
    Ueq=self.fields['ueq']
    A=self.fields['gravity']
    fpop=feq*0
    for pp in range(self.nphase):
        for vv in [0,1]:
            accPhaseDir=A[...,pp,vv]
            for dd in range(self.ndir):
                fpop[...,pp,dd]=fpop[...,pp,dd]*(1-self.fields['tau'][...,pp]/2)*3* \
                (accPhaseDir*(self.c[vv,dd]-Ueq[...,pp,vv]))*feq[...,pp,dd]
    fpop[~np.isfinite(fpop)]=0
    self.fields['fpop']=fpop
    
def cb_postUeq(self):
    A=fluidFluidInteraction(self)
    #Ueq=np.zeros((*self.dim,self.nphase,2))
    #for pp in range(self.nphase):
    #    Ueq[...,pp,:]=self.fields['ueq']
    
    for ii in [0,1]:
        self.fields['ueq'][...,ii]+=A[...,ii]
    #self.fields['ueq']=Ueq
    #print('cb_postUeq!')
if(__name__=="__main__"):
    import pylbm2 as pylbm
    ny=30;nx=30
    S=pylbm.LBM((ny,nx),nphase=2)
    
    S.fields['rho'][10:20,10:20,0]=0;
    S.fields['rho'][...,1]=0;
    S.fields['rho'][10:20,10:20,1]=1;
    S.fluidPhases=[0,1]
    S.initDistribution()
    #ffplot(S)
    
    # define shanChen
    S.fields['G']=np.ones((ny,nx,1))*3
    S.shanChen={'pairs':[[0,1]]}
    #fluidFluidInteraction(S)
    
    
    def cb_mov(self):
        """movie callback"""
        if(self.step%20!=0): return
        
        # # cv2
        img = cv2.normalize(self.fields['v'][:,:,1], None, 
                            alpha=0, beta=255, 
                            norm_type=cv2.NORM_MINMAX, 
                            dtype=cv2.CV_8U)
        # self.mov.write(img)
        
        # ffmpeg
        plt.clf()
        plt.imshow(img);plt.axis('off');plt.title(self.step)
        plt.streamplot(mx,my,self.fields['v'][:,:,1],self.fields['v'][:,:,0],density=.5)
        self.mov.grab_frame()
    
    #cb={'postMacro':[cb_postMacro,cb_mov]}
    cb={'postUeq':[cb_postUeq]}
    #S.sim(steps=2,callbacks=cb)
    
# MJPG, DIVX, mp4v, XVID, X264 
#fourcc=cv2.VideoWriter_fourcc(*'MP4V');fps=30
#S.mov= cv2.VideoWriter('cv2_v.mp4',fourcc, fps, (nx,ny),False)

# fps=30
# mx,my=np.meshgrid(range(nx),range(ny));
# #https://matplotlib.org/stable/gallery/animation/frame_grabbing_sgskip.html
# f1 = plt.figure()
# metadata = dict(title='lbm-v', artist='lbm',comment='Movie!')
# S.mov = FFMpegWriter(fps=fps, metadata=metadata)
# dpi=100
# with S.mov.saving(f1, "mpl_v.mp4", dpi):

S.sim(steps=500,callbacks=cb)

#S.mov.release()

plt.figure()
plt.subplot(2,2,1)
plt.imshow(S.fields['rho'][:,:,0]);plt.axis('off');plt.title('rho0');plt.colorbar();
plt.subplot(2,2,2)
plt.imshow(S.fields['rho'][:,:,1]);plt.axis('off');plt.title('rho1');plt.colorbar();
plt.tight_layout()
#print(S.fields['v'].shape)