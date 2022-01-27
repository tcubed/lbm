"""
simple 2D LBM
Some inspiration from https://exolete.com/lbm/
@author: Ted Tower
"""
import numpy as np
import time
class LBM():
    def __init__(self,sizeyx,nphase=1,invtau=None):
        self.dim=sizeyx
        self.nphase=nphase
        self.ndir=9
        self.c=np.array([[0,1,0,-1, 0,1,-1,-1, 1],
                         [0,0,1, 0,-1,1, 1,-1,-1]]);
        t1=4./9;t2=1./9;t3=1./36;
        self.w=np.array([t1,t2,t2,t2,t2,t3,t3,t3,t3]) # weights in each dir
        
        # init velocity, density, and distribution fields
        self.fields={'invtau':np.ones((*self.dim,self.nphase)),
                     'v':np.zeros((*self.dim,2)),
                     'rho':np.ones((*self.dim,self.nphase)),
                     'ns':np.zeros((*self.dim,self.nphase)),
                     'Fin':np.zeros((*self.dim,self.nphase,self.ndir)),
                     'Fout':np.zeros((*self.dim,self.nphase,self.ndir)),
                     'Feq':np.zeros((*self.dim,self.nphase,self.ndir)),
                     'flowMode':np.zeros((*self.dim,self.nphase))+2}
        if(invtau is not None): self.fields['invtau']=invtau
        self.initDistribution();
        self.reflected=[0,3,4,1,2,7,8,5,6]
    
    def initDistribution(self):
        self.fields['Feq']=np.zeros((*self.dim,self.nphase,self.ndir))
        self.calcFeq();
        self.fields['Fout']=self.fields['Feq'].copy()
        
    def calcFeq(self):
        u2c=1.5*(self.fields['v']**2).sum(axis=-1)
        for pp in range(self.nphase):
            for ii in range(self.ndir):
                cuns=3*(self.c[0,ii]*self.fields['v'][...,0]+self.c[1,ii]*self.fields['v'][...,1])
                #self.fields['Feq'][...,pp,ii]=self.w[ii]*self.fields['rho'][...,pp]*(1+cuns+0.5*cuns**2-u2c)
                
                k0=np.where(self.fields['flowMode'][...,pp]==0)
                k1=np.where(self.fields['flowMode'][...,pp]==1)
                k2=np.where(self.fields['flowMode'][...,pp]>=1)
                if(k0[0].size):
                    self.fields['Feq'][k0+(pp,ii,)]=self.w[ii]*self.fields['rho'][k0+(pp,)]
                if(k1[0].size):
                    self.fields['Feq'][k1+(pp,ii,)]=self.w[ii]*self.fields['rho'][k1+(pp,)]*(1+cuns[k1])
                if(k2[0].size):
                    self.fields['Feq'][k2+(pp,ii,)]=self.w[ii]*self.fields['rho'][k2+(pp,)]*(1+cuns[k2]+0.5*cuns[k2]**2-u2c[k2])
    def calcMacro(self):
        F=self.fields['Fin']
        self.fields['rho']=F.sum(axis=-1);
        rhoTot=self.fields['rho'].sum(axis=-1)
        with np.errstate(invalid='ignore'):
            self.fields['v'][...,0]=((F[...,1]+F[...,5]+F[...,8])-
                                     (F[...,3]+F[...,6]+F[...,7])).sum(axis=-1)/rhoTot
            self.fields['v'][...,1]=((F[...,2]+F[...,5]+F[...,6])-
                                     (F[...,4]+F[...,7]+F[...,8])).sum(axis=-1)/rhoTot
    def stream(self):
        for ii in range(self.ndir):
            self.fields['Fin'][...,ii]=np.roll(self.fields['Fout'][...,ii],
                                   (self.c[1,ii],self.c[0,ii]),axis=(0,1))
    def collide(self):
        invtau=np.tile(self.fields['invtau'][...,np.newaxis],(1,1,1,9))
        self.fields['Fout']=invtau*self.fields['Feq']+(1-invtau)*self.fields['Fin'];
    def PBB(self,ON):
        F=self.fields['Fout']
        for ii in range(self.ndir):
            k=ON+(ii,)
            F[k]=F[k]+self.fields['ns'][ON]*(self.fields['Fin'][ON+(self.reflected[ii],)]-F[k])
        
    def sim(self,steps=1,callbacks=None,verbose=False):
        if(callbacks is None): callbacks={}
        for k in ['postStream','postMacro']:
            if(k not in callbacks): callbacks[k]=[]
        ON=np.where(self.fields['ns'])
        self.flowIdx=(np.where(self.fields['flowMode']==0),
                      np.where(self.fields['flowMode']==1),
                      np.where(self.fields['flowMode']>=1))
        t0=time.time();
        for ii in range(steps):
            self.step=ii
            if((ii>0) and (ii%100==0)):
                tnow=time.time()
                mlups=np.prod(self.fields['rho'].shape)*ii/1e6/(tnow-t0)
                print('%d: %.3gmlups (%.1fsec/epoch)'%(ii,mlups,tnow-t0))
                t0=tnow
            self.stream()
            # callbacks (e.g. vel BCs, report out)
            for cb in callbacks['postStream']: cb(self)
            self.calcMacro()
            for cb in callbacks['postMacro']: cb(self)
            self.calcFeq();
            self.collide()
            self.PBB(ON)
            
            if(np.any(self.fields['rho']>10)):
                print('ack!: velocity too high! (step %d)'%self.step)
                break
        if(verbose):
            print('done! (%.2fmin)'%((time.time()-t0)/60))

if(__name__=='__main__'):
    import matplotlib.pyplot as plt
    S=LBM((30,60),nphase=2)
    S.fields['ns'][5:25,15:40,0]=1
    def cb_vel(self):
        self.fields['v'][:,0,0]=.1
        self.fields['v'][:,-1,:]=self.fields['v'][:,-2,:]
    S.sim(steps=300,callbacks={'postMacro':[cb_vel]})
    plt.figure(figsize=(6,9))
    plt.subplot(2,1,1)
    plt.imshow(S.fields['v'][:,:,0]);plt.colorbar();
    plt.subplot(2,1,2)
    plt.imshow(S.fields['rho']);plt.colorbar()
    #plt.plot(S.fields['v'][15,:,0])