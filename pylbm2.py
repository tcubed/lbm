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
        self.fields={'v':np.zeros((*sizeyx,2)),
                     'rho':np.ones(sizeyx),
                     'solid':np.zeros(sizeyx),
                     'Fin':np.zeros((*sizeyx,self.ndir)),
                     'Fout':np.zeros((*sizeyx,self.ndir)),
                     'Feq':np.zeros((*sizeyx,self.ndir))}
        self.initDistribution();
        
        #self.solid=np.zeros(sizeyx)  # solid points for bouncebacko documentation available 
        self.toreflect=[0,1,2,3,4,5,6,7,8]
        self.reflected=[0,3,4,1,2,7,8,5,6]
        self.bounced=self.fields['Fin'].copy()
    
    def initDistribution(self):
        self.fields['Feq']=np.zeros((*self.dim,self.ndir))
        self.calcFeq();
        self.fields['Fin']=self.fields['Feq'].copy()
        
    def calcFeq(self):
        #c_squ=1/3;
        u2c=1.5*(self.fields['v']**2).sum(axis=-1)
        for ii in range(self.ndir):
            cuns=3*(self.c[0,ii]*self.fields['v'][:,:,0]+self.c[1,ii]*self.fields['v'][:,:,1])
            self.fields['Feq'][:,:,ii]=self.w[ii]*self.fields['rho']*(1+cuns+0.5*cuns**2-u2c)
                
    def calcMacro(self):
        F=self.fields['Fin']
        self.fields['rho']=F.sum(axis=-1)
        with np.errstate(invalid='ignore'):
            self.fields['v'][:,:,0]=((F[...,1]+F[...,5]+F[...,8])-
                                     (F[...,3]+F[...,6]+F[...,7]))/self.fields['rho']
            self.fields['v'][:,:,1]=((F[...,2]+F[...,5]+F[...,6])-
                                     (F[...,4]+F[...,7]+F[...,8]))/self.fields['rho']
    def stream(self):
        for ii in range(self.ndir):
            self.fields['Fin'][...,ii]=np.roll(self.fields['Fin'][:,:,ii],
                                   (self.c[1,ii],self.c[0,ii]),axis=(0,1))
    def collide(self):
        self.fields['Fout']=self.omega*self.fields['Feq']+(1-self.omega)*self.fields['Fin'];
    def bounceback(self,ON):
        for dd in range(self.ndir):
            F=self.fields['Fout'][:,:,dd]
            B=self.bounced[:,:,self.reflected[dd]]
            F[ON]=B[ON]
            self.fields['Fout'][:,:,dd]=F#self.bounced[:,:,self.reflected[dd]]
    def sim(self,steps=10,callbacks=None,verbose=False):
        if(callbacks is None): callbacks={}
        for k in ['postStream','postMacro']:
            if(k not in callbacks): callbacks[k]=[]
        ON=np.where(self.fields['solid'])
        t0=time.time();
        tepoch=t0
        for ii in range(steps):
            self.step=ii
            if((ii>0) and (ii%100==0)):
                tnow=time.time()
                telapsed=tnow-t0
                mlups=np.prod(self.fields['rho'].shape)*ii/1e6/telapsed
                print('%d: %.3gmlups (%.1fsec/epoch)'%(ii,mlups,tnow-tepoch))
                tepoch=tnow
            self.stream()
            for dd in range(self.ndir):
                self.bounced[...,dd]=self.fields['Fin'][...,self.toreflect[dd]]
            # callbacks (e.g. vel BCs, report out)
            for cb in callbacks['postStream']:
                cb(self)
            self.calcMacro()
            for cb in callbacks['postMacro']:
                cb(self)
            # BC: solids bounceback
            self.fields['rho'][ON]=0;
            for jj in [0,1]:
                v0=self.fields['v'][:,:,jj];v0[ON]=0;self.fields['v'][:,:,jj]=v0
            
            self.calcFeq();
            self.collide()
            self.bounceback(ON);
            if(np.any(self.fields['rho']>10)):
                print('ack!: velocity too high! (step %d)'%self.step)
                break
        if(verbose):
            print('done! (%.2fmin)'%((time.time()-t0)/60))

if(__name__=='__main__'):
    import matplotlib.pyplot as plt
    S=LBM((30,60))
    S.fields['solid'][10:20,15:20]=1
    def cb_vel(self):
        self.fields['v'][:,0,1]=.1
    S.sim(steps=500,callbacks={'postMacro':[cb_vel]})
    plt.imshow(S.fields['v'][:,:,0])
    plt.plot(S.fields['v'][15,:,0])