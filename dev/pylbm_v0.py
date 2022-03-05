"""
simple 2D LBM
Some inspiration from https://exolete.com/lbm/
@author: Ted Tower
"""
import numpy as np
import time
class LBM():
    def __init__(self,sizeyx,omega=1):
        self.dim=sizeyx
        self.ndir=9
        self.c=np.array([[0,1,0,-1, 0,1,-1,-1, 1],
                         [0,0,1, 0,-1,1, 1,-1,-1]]);
        t1=4/9;t2=1/9;t3=1/36;
        self.w=np.array([t1,t2,t2,t2,t2,t3,t3,t3,t3]) # weights in each dir
        self.omega=omega
        
        # init velocity, density, and distribution fields
        self.v=np.zeros((*sizeyx,2))
        self.rho=np.ones(sizeyx)
        self.initDistribution();
        
        self.solid=np.zeros(sizeyx)  # solid points for bouncebacko documentation available 
        self.toreflect=[0,1,2,3,4,5,6,7,8]
        self.reflected=[0,3,4,1,2,7,8,5,6]
        self.bounced=self.F.copy()
    
    def initDistribution(self):
        self.Feq=np.zeros((*self.dim,self.ndir))
        self.calcFeq();
        self.F=self.Feq.copy()
        
    def calcFeq(self):
        #c_squ=1/3;
        u2c=1.5*(self.v[:,:,0]**2+self.v[:,:,1]**2);
        for ii in range(self.ndir):
            cuns=3*(self.c[0,ii]*self.v[:,:,0]+self.c[1,ii]*self.v[:,:,1])#/c_squ
            self.Feq[:,:,ii]=self.w[ii]*self.rho*(1+cuns+0.5*cuns**2-u2c)
                
    def calcMacro(self):
        self.rho=self.F.sum(axis=-1)
        with np.errstate(invalid='ignore'):
            self.v[:,:,0]=((self.F[:,:,1]+self.F[:,:,5]+self.F[:,:,8])-
                          (self.F[:,:,3]+self.F[:,:,6]+self.F[:,:,7]))/self.rho
            self.v[:,:,1]=((self.F[:,:,2]+self.F[:,:,5]+self.F[:,:,6])-
                          (self.F[:,:,4]+self.F[:,:,7]+self.F[:,:,8]))/self.rho
    def stream(self):
        for ii in range(self.ndir):
            self.F[:,:,ii]=np.roll(self.F[:,:,ii],
                                   (self.c[1,ii],self.c[0,ii]),
                                   axis=(0,1))
    def collide(self):
        self.F=self.omega*self.Feq+(1-self.omega)*self.F;
    def bounceback(self,ON):
        for dd in range(self.ndir):
            F=self.F[:,:,dd]
            B=self.bounced[:,:,self.reflected[dd]]
            F[ON]=B[ON]
            self.F[:,:,dd]=F#self.bounced[:,:,self.reflected[dd]]
    def sim(self,steps=10,callbacks=None,verbose=False):
        if(callbacks is None): callbacks={}
        for k in ['postStream','postMacro']:
            if(k not in callbacks): callbacks[k]=[]
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
            # callbacks (e.g. vel BCs, report out)
            for cb in callbacks['postStream']:
                cb(self)
            self.calcMacro()
            for cb in callbacks['postMacro']:
                cb(self)
            # BC: solids bounceback
            self.rho[ON]=0;
            for jj in [0,1]:
                v0=self.v[:,:,jj];v0[ON]=0;self.v[:,:,jj]=v0
            
            self.calcFeq();
            self.collide()
            self.bounceback(ON);
            if(np.any(self.rho>10)):
                print('ack!: velocity too high! (step %d)'%self.step)
                break
        if(verbose):
            print('done! (%.2fmin)'%((time.time()-t0)/60))

if(__name__=='__main__'):
    import matplotlib.pyplot as plt
    S=LBM((30,60))
    S.solid[10:20,15:20]=1
    def cb_vel(self):
        self.v[:,0,1]=.1
    S.sim(steps=500,callbacks=[cb_vel])
    plt.imshow(S.v[:,:,1])