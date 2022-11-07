# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:29:09 2022

@author: Ted
"""
import os
import numpy as np

import matplotlib.pyplot as plt
import cv2

#========================================================
#                 STANDARD
#========================================================
# def cb_timing(self):
#     if((ii>0) and (ii%500==0)):
#         tnow=time.time()
#         mlups=np.prod(self.fields['rho'].shape)*500/1e6/(tnow-t0)
#         print('%d: %.3gmlups (%.2fsec/epoch)'%(ii,mlups,tnow-t0))
#         t0=tnow
def ueqForcingSC(self):
    # Guo, pg 67
    A=np.zeros(self.fields['gravity'].shape)
    for ii in [0,1,2]:
        A[...,ii]=self.fields['tau']*self.fields['gravity'][...,ii]
    self.fields['ueq']+=A
def ueqForcingHSD(self):
    # Guo, pg 67
    A=np.zeros(self.fields['gravity'].shape)
    for ii in [0,1,2]:
        v=self.fields['rho']*self.fields['gravity'][...,ii]/2
       # v=self.fields['tau']*self.fields['gravity'][...,ii]
        A[...,ii]=v
    self.fields['ueq']+=A
    
def feqForcingHSD(self):
    # Guo, pg 70
    Ueq=self.fields['ueq']
    A=self.fields['gravity']
    fpop=self.fields['Feq']*0
    for pp in range(self.nphase):
        for dd in range(self.ndir):
            term=0
            for vv in [0,1,2]:
                term+=A[...,pp,vv]*(self.c[vv,dd]-Ueq[...,pp,vv])
            fpop[...,pp,dd]=(1-1/(self.fields['tau'][...,pp]*2))*3*term*self.fields['Feq'][...,pp,dd]
    fpop[~np.isfinite(fpop)]=0
    self.fields['Fpop']=fpop

def history(self):
    assert hasattr(self,'history'),"history callback needs self.history dict"
    H=self.history
    if(self.step==0):
        # init logging
        if(os.path.exists(H['file'])):
            f=open(H['file'],'rt');f.close()
            os.remove(H['file'])
        H['fileobj']=open(H['file'],'wt');
        
        # default requests
        if('requests' not in H): H['requests']=[]
        defreq=['mass','maxu']
        for k in defreq:
            if(k not in H['requests']): H['requests'].append(k)
            
        # create header line
        hdr='step';
        for k in H['requests']:
            hdr+=','+k
        print(hdr,file=H['fileobj'])
    
    if(self.step>=0):
        val=str(self.step)
        for k in H['requests']:
            if(k=='mass'):
                val+=',%.3g'%self.fields['rho'].sum()
            elif(k=='maxu'):
                val+=',%.3g'%np.max(np.abs(self.fields['u']))
        #print(val)
        print(val,file=H['fileobj'])
    else:
        # end of simulation, step is set to -1; flush and close
        H['fileobj'].flush()
        H['fileobj'].close()

#========================================================
#                 SHAN-CHEN
#========================================================
def fluidFluidInteractionSCMP(self):
    """
    Shan-Chen fluid-fluid interaction
    
    This callback should be called in postUeq.
    
    If 'fluidFluidPotential' is a defined field, that will be used.
    Otherwise, the 'rho' field will be used.
    
    Returns
    -------
    None.

    """
    assert hasattr(self,'shanChen'), "sim needs 'shanChen' dict for fluidFluidInteraction"
    assert ('G' in self.fields), "Shan-Chen fluid-solid needs 'G' field."
    #SC=self.shanChen
    #npair=len(SC['pairs'])
    
    if('fluidFluidPotential' in self.fields):
        psi=self.fields['fluidFluidPotential']
    else:
        #psi=np.tile(np.expand_dims(self.fields['G'][...,0],axis=-1),(1,1,1,self.nphase))*self.fields['rho']
        psi=self.fields['rho']
    
    V=np.zeros((*self.dim,self.nphase,3))
    for ii in range(self.ndir):
        # roll in ii direction
        shift=(-self.c[0,ii],-self.c[1,ii],-self.c[2,ii],0)
        d0=np.roll(psi*self.w[ii],shift,axis=(0,1,2,3))
        for dd in [0,1,2]:
            V[...,dd]+=d0*self.c[dd,ii]
    # calc accel
    A=np.zeros((*self.dim,self.nphase,3))
    #for jj in range(npair):
    #    for ii in [0,1]:
            #Gtau=self.fields['G'][...,jj]*self.fields['tau'][...,SC['pairs'][jj][ii]]
    #taup=self.fields['tau'][...,0]
    #for dd in [0,1,2]:
    localTerms=np.tile(self.fields['G'][...,0],(1,1,1,self.nphase))*self.fields['tau']*psi/self.fields['rho']
    for dd in [0,1,2]:
        A[...,dd]-=localTerms*V[...,dd]
    #A-=self.fields['tau']*V
    self.fields['ueq']+=A
    
def fluidFluidInteractionMCMP(self):
    """
    Shan-Chen fluid-fluid interaction
    
    This callback should be called in postUeq.
    
    If 'fluidFluidPotential' is a defined field, that will be used.
    Otherwise, the 'rho' field will be used.
    
    Returns
    -------
    None.

    """
    assert hasattr(self,'shanChen'), "sim needs 'shanChen' dict for fluidFluidInteraction"
    assert ('G' in self.fields), "Shan-Chen fluid-solid needs 'G' field."
    SC=self.shanChen
    npair=len(SC['pairs'])
    
    if('fluidFluidPotential' in self.fields):
        psi=self.fields['fluidFluidPotential']
    else:
        #psi=np.tile(np.expand_dims(self.fields['G'][...,0],axis=-1),(1,1,1,self.nphase))*self.fields['rho']
        psi=self.fields['rho']
        
    V=np.zeros((*self.dim,self.nphase,3))
    for ii in range(self.ndir):
        # roll in ii direction
        shift=(-self.c[0,ii],-self.c[1,ii],-self.c[2,ii],0)
        d0=np.roll(psi[...,[1,0]]*self.w[ii],shift,axis=(0,1,2,3))
        for dd in [0,1,2]:
            V[...,dd]+=d0*psi*self.c[dd,ii]
    # calc accel
    # A=np.zeros((*self.dim,self.nphase,3))
    # for jj in range(npair):
    #     for ii in [0,1]:
    #         #Gtau=self.fields['G'][...,jj]*self.fields['tau'][...,SC['pairs'][jj][ii]]
    #         taup=self.fields['tau'][...,SC['pairs'][jj][ii]]
    #         for dd in [0,1,2]:
    #             A[...,SC['pairs'][jj][ii],dd]-=taup*V[...,SC['pairs'][jj][1-ii],dd]
    # self.fields['ueq']+=A
    F=np.zeros((*self.dim,self.nphase,3))
    for jj in range(npair):
        G=self.fields['G'][...,jj:(jj+1)]
        F-=np.expand_dims(G,-1)*V
        
    # calc accel
    du=np.zeros((*self.dim,self.nphase,3))
    for jj in range(npair):
        pr=SC['pairs'][jj]
        tau=self.fields['tau'][...,pr]#+1e-12
        rho=self.fields['rho'][...,pr]+1e-12
        #G=self.fields['G'][...,jj:(jj+1)]
        
        #with np.errstate(divide='ignore',invalid='ignore'):
        for dd in [0,1,2]:
            #du[...,pr,dd]+=G*V[...,pr,dd]*tau/rho
            du[...,pr,dd]+=F[...,pr,dd]*tau/rho
            #du[...,pr,:]+=F[...,pr,:]*np.expand_dims(tau/rho,-1)
        # F=np.expand_dims(self.fields['G'][...,jj:(jj+1)],-1)*V
        # with np.errstate(divide='ignore'):
        #     localTerms=self.fields['tau'][...,SC['pairs'][jj]] \
        #         /self.fields['rho'][...,SC['pairs'][jj]]
        #     #localTerms=np.expand_dims(tauRho,-1)
            
        #     for dd in [0,1,2]:
        #         du[...,dd]+=localTerms*F[...,dd]
    self.fields['ueq']+=du
    
    
def fluidSolidInteraction(self):
    """
    Shan-Chen fluid-solid interaction
    
    This callback should be called in postUeq.
    
    If 'fluidSolidPotential' is a defined field, that will be used.
    Otherwise, the field['Gads']*field['tau']*field['rhoWall'] will be used.
    
    Returns
    -------
    None.

    """
    #assert ('fluidSolidPotential' in self.fields), "Shan-Chen fluid-solid needs 'fluidSolidPotential' field."
    for k in ['Gads','rhoWall']:
        assert (k in self.fields), "Shan-Chen fluid-solid needs '%s' field."%k
    if('fluidSolidPotential' in self.fields):
        psi=self.fields['fluidSolidPotential']
    else:
        psi=self.fields['Gads']*self.fields['rhoWall']
    A=np.zeros((*self.dim,self.nphase,3))
    for ii in range(self.ndir):
        # roll in ii direction
        shift=(-self.c[0,ii],-self.c[1,ii],-self.c[2,ii],0)
        d0=np.roll(psi*self.w[ii],shift,axis=(0,1,2,3))
        for dd in [0,1,2]:
            A[...,dd]-=d0*self.c[dd,ii]*self.fields['tau']
    self.fields['ueq']+=A
    
def ueqForcingSCFluidFluid(self):
    PSI=self.fields['rho'].copy()
    return self.fluidFluidPotentialAccel(PSI)

def ueqForcingSCFluidSurface(self):
    PSI=self.fields['Gads']*self.fields['tau']*self.fields['rhoWall']
    return self.fluidOtherPotentialAccel(PSI)


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
        plt.streamplot(mx,my,self.fields['u'][:,:,1],self.fields['u'][:,:,0],density=.5)
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

# S.sim(steps=500,callbacks=cb)

# #S.mov.release()

# plt.figure()
# plt.subplot(2,2,1)
# plt.imshow(S.fields['rho'][:,:,0]);plt.axis('off');plt.title('rho0');plt.colorbar();
# plt.subplot(2,2,2)
# plt.imshow(S.fields['rho'][:,:,1]);plt.axis('off');plt.title('rho1');plt.colorbar();
# plt.tight_layout()
# #print(S.fields['v'].shape)