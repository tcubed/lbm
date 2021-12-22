"""
simple 2D LBM
Some inspiration from https://exolete.com/lbm/
@author: Ted Tower
"""
import numpy as np
import time

from numba import int32, int64,float32,float64    # import the types
from numba.experimental import jitclass

spec = [
    ('dim', int64[:]),
    ('ndir', int64),
    ('c', int64[:,:]), 
    ('w', float64[:]),
    ('omega', float64), 
    ('v', float64[:,:,:]), 
    ('rho', float64[:,:]), 
    ('F', float64[:,:,:]), 
    ('Feq', float64[:,:,:]), 
    ('solid', int64[:,:]), 
    ('toreflect', int64[:]), 
    ('reflected', int64[:]), 
    ('bounced', float64[:,:,:]), 
]

@jitclass(spec)
class LBM():
    def __init__(self,ny,nx,omega=1):
        self.dim=[ny,nx]
        self.ndir=9
        self.c=np.array([[0,1,0,-1,1,1,-1,-1,0],  # component in each dir
                         [1,0,-1,0,1,-1,-1,1,0]]);
        t1=4/9;t2=1/9;t3=1/36;
        self.w=np.array([t2,t2,t2,t2,t3,t3,t3,t3,t1]) # weights in each dir
        self.omega=omega
        
        # init velocity, density, and distribution fields
        self.v=np.zeros((self.dim[0],self.dim[1],2))
        #self.rho=np.ones(self.dim,dtype=np.float64)
        self.rho=np.ones(self.dim)
        self.initDistribution();
        
        self.solid=np.zeros(self.dim,dtype=np.int64)  # solid points for bouncebacko documentation available 
        self.toreflect=[0,1,2,3,4,5,6,7,8]
        self.reflected=[2,3,0,1,6,7,4,5,8]
        self.bounced=self.F.copy()
    
    def initDistribution(self):
        self.Feq=np.zeros((self.dim[0],self.dim[1],self.ndir),dtype=np.float64)
        self.calcFeq();
        self.F=self.Feq.copy()
        
    def calcFeq(self):
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
                mlups=np.prod(self.rho.shape)*ii/1e6/telapsed
                print('%d: %.3gmlups (%.1fsec/epoch)'%(ii,mlups,tnow-tepoch))
                tepoch=tnow
            self.stream()
            for dd in range(self.ndir):
                self.bounced[:,:,dd]=self.F[:,:,self.toreflect[dd]]
            self.calcMacro()
            
            # BC: solids bounceback
            self.rho[ON]=0;
            for jj in [0,1]:
                v0=self.v[:,:,jj];v0[ON]=0;self.v[:,:,jj]=v0
            # callbacks (e.g. vel BCs, report out)
            for cb in callbacks:
                cb(self)
            self.calcFeq();
            self.collide()
            self.bounceback(ON);
            if(np.any(self.rho>10)):
                break
        print('done! (%.2fmin)'%((time.time()-t0)/60))

if(__name__=='__main__'):
    import matplotlib.pyplot as plt
    S=LBM((30,60))
    S.solid[10:20,15:20]=1
    def cb_vel(self):
        self.v[:,0,1]=.1
    S.sim(steps=500,callbacks=[cb_vel])
    plt.imshow(S.v[:,:,1])