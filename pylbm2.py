""" simple 2D LBM: extendable to multi-component and thermal
@author: Ted Tower
"""
import numpy as np
import time
class LBM():
    def __init__(self,nzyx,nphase=1,tau=None):
        self.dim=nzyx
        self.nphase=nphase
        self.ndir=9
        self.c=np.array([[0,0,0, 0, 0,0, 0, 0, 0],  # z
                         [0,0,1, 0,-1,1, 1,-1,-1],  # y
                         [0,1,0,-1, 0,1,-1,-1, 1]]);# x
        t1=4./9;t2=1./9;t3=1./36;
        self.w=np.array([t1,t2,t2,t2,t2,t3,t3,t3,t3]) # weights in each dir
        # init velocity, density, and distribution fields
        self.fields={'tau':np.ones((*self.dim,self.nphase)),
                     'v':np.zeros((*self.dim,3)),
                     'ueq':np.zeros((*self.dim,self.nphase,3)),
                     'rho':np.ones((*self.dim,self.nphase)),
                     'ns':np.zeros((*self.dim,self.nphase)),
                     'Fin':np.zeros((*self.dim,self.nphase,self.ndir)),
                     'Fout':np.zeros((*self.dim,self.nphase,self.ndir)),
                     'Feq':np.zeros((*self.dim,self.nphase,self.ndir)),
                     'Fpop':np.zeros((*self.dim,self.nphase,self.ndir)),
                     'flowMode':np.zeros((*self.dim,self.nphase))+2}
        if(tau is not None): self.fields['tau']=tau
        self.flowIdx=[]
        self.getFlowIndex()
        self.fluidPhases=[0]
        self.initDistribution();
        self.reflected=[0,3,4,1,2,7,8,5,6]
        self.step=0
    def initDistribution(self):
        self.fields['Feq']=np.zeros((*self.dim,self.nphase,self.ndir))
        #for ii in range(self.nphase):
        #    self.fields['ueq'][...,ii,:]=self.fields['v']
        self.calcFeq();
        self.fields['Fout']=self.fields['Feq'].copy()
    def getFlowIndex(self):
        self.flowIdx=[]
        for pp in range(self.nphase):
            k0=np.where(self.fields['flowMode'][...,pp]==0)
            k1=np.where(self.fields['flowMode'][...,pp]==1)
            k2=np.where(self.fields['flowMode'][...,pp]>=1)
            self.flowIdx.append((k0,k1,k2))
            
    def calcFeq(self):
        #for ii in range(self.nphase):
        #    self.fields['ueq'][...,ii,:]=self.fields['v']
        #u2c=1.5*(self.fields['v']**2).sum(axis=-1)
        for pp in range(self.nphase):
            u2c=1.5*(self.fields['ueq'][...,pp,:]**2).sum(axis=-1)
            k0,k1,k2=self.flowIdx[pp]
            for ii in range(self.ndir):
                cuns=3*(self.c[0,ii]*self.fields['ueq'][...,pp,0]+
                        self.c[1,ii]*self.fields['ueq'][...,pp,1]+
                        self.c[2,ii]*self.fields['ueq'][...,pp,2])
                #cuns=3*(self.c[0,ii]*self.fields['v'][...,0]+self.c[1,ii]*self.fields['v'][...,1])
                #self.fields['Feq'][...,pp,ii]=self.w[ii]*self.fields['rho'][...,pp]*(1+cuns+0.5*cuns**2-u2c)
                if(k0[0].size): self.fields['Feq'][k0+(pp,ii,)]=self.w[ii]*self.fields['rho'][k0+(pp,)]
                if(k1[0].size): self.fields['Feq'][k1+(pp,ii,)]=self.w[ii]*self.fields['rho'][k1+(pp,)]*(1+cuns[k1])
                if(k2[0].size): self.fields['Feq'][k2+(pp,ii,)]=self.w[ii]*self.fields['rho'][k2+(pp,)]*(1+cuns[k2]+0.5*cuns[k2]**2-u2c[k2])
    def calcMacro(self):
        self.fields['rho']=self.fields['Fin'].sum(axis=-1);
        F=self.fields['Fin'][...,self.fluidPhases,:]
        rhoTot=self.fields['rho'][...,self.fluidPhases].sum(axis=-1)
        with np.errstate(invalid='ignore'):
            self.fields['v'][...,2]=((F[...,1]+F[...,5]+F[...,8])-
                                     (F[...,3]+F[...,6]+F[...,7])).sum(axis=-1)/rhoTot
            self.fields['v'][...,1]=((F[...,2]+F[...,5]+F[...,6])-
                                     (F[...,4]+F[...,7]+F[...,8])).sum(axis=-1)/rhoTot
    def calcUeq(self):
        for ii in range(self.nphase):
            self.fields['ueq'][...,ii,:]=self.fields['v']
    def stream(self):
        for ii in range(self.ndir):
            self.fields['Fin'][...,ii]=np.roll(self.fields['Fout'][...,ii],
                                   (self.c[0,ii],self.c[1,ii],self.c[2,ii],0),axis=(0,1,2,3))
    def collide(self):
        invtau=np.tile(1/self.fields['tau'][...,np.newaxis],(1,1,1,1,9))
        self.fields['Fout']=invtau*self.fields['Feq']+(1-invtau)*self.fields['Fin']+self.fields['Fpop'];
    def PBB(self,ON):
        F=self.fields['Fout']
        for ii in range(self.ndir):
            k=ON+(ii,)
            F[k]=F[k]+self.fields['ns'][ON]*(self.fields['Fin'][ON+(self.reflected[ii],)]-F[k])
    def sim(self,steps=1,callbacks=None,verbose=False):
        if(callbacks is None): callbacks={}
        for k in ['postStream','postMacro','postUeq','postFeq','init','final']:
            if(k not in callbacks): callbacks[k]=[]
        ON=np.where(self.fields['ns'])
        self.getFlowIndex()
        for cb in callbacks['init']: cb(self)
        t0=time.time();
        for ii in range(steps):
            self.step=ii
            if((ii>0) and (ii%500==0)):
                tnow=time.time()
                mlups=np.prod(self.fields['rho'].shape)*500/1e6/(tnow-t0)
                print('%d: %.3gmlups (%.2fsec/epoch)'%(ii,mlups,tnow-t0))
                t0=tnow
            self.stream()
            # callbacks (e.g. vel BCs, report out)
            for cb in callbacks['postStream']: cb(self) # set distributions here
            self.calcMacro()
            for cb in callbacks['postMacro']: cb(self)  # set v and rho here
            self.calcUeq()
            for cb in callbacks['postUeq']: cb(self)  # set v and rho here
            self.calcFeq();
            for cb in callbacks['postFeq']: cb(self) 
            self.collide()
            self.PBB(ON)
            if(np.any(self.fields['v']>1)):
                print('ack!: velocity too high! (step %d)'%self.step)
                break
        self.step=-1  # simple state flag
        for cb in callbacks['final']: cb(self)
        if(verbose):
            print('done! (%.2fmin)'%((time.time()-t0)/60))

if(__name__=='__main__'):
    import matplotlib.pyplot as plt
    S=LBM((30,60),nphase=1)
    S.fields['ns'][5:25,15:40,0]=1
    
    # callbacks
    # -- bc
    def cb_vel(self):
        self.fields['v'][:,0,0]=.1
        self.fields['v'][:,-1,:]=self.fields['v'][:,-2,:]
    # -- display
    def myplot(self):
        plt.imshow(self.fields['v'][:,:,0]);
    def cb_postMacro(self):
        plt.figure();myplot(self);plt.title('postMacro-v');plt.show()
    def cb_postStream(self):
        plt.figure();myplot(self);plt.title('postStream-v');plt.show()
    #cb={'postMacro':[cb_vel,cb_postMacro],'postStream':[cb_postStream]}
    cb={'postMacro':[cb_vel],'postStream':[]}
# %%
    S.sim(steps=2000,callbacks=cb)
    
    plt.figure(figsize=(6,9))
    plt.subplot(2,1,1)
    plt.imshow(S.fields['v'][:,:,0]);plt.colorbar();plt.axis('off');plt.title('vx')
    plt.subplot(2,1,2)
    plt.imshow(S.fields['rho']);plt.colorbar();plt.axis('off');plt.title('rho')
    #plt.plot(S.fields['v'][15,:,0])