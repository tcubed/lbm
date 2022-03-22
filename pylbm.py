"""simple LBM: 2D, that can be extended to 3D, multiphase, etc through callbacks
@author: Ted Tower
"""
import numpy as np
import time
class LBM():
    def __init__(self,nzyx,nphase=1,tau=None):
        self.dim=nzyx
        self.nphase=nphase
        self.ndir=9
        self.c=np.array([[0,0,0, 0, 0,0, 0, 0, 0],  # z -- direction vectors for D2Q9
                         [0,0,1, 0,-1,1, 1,-1,-1],  # y
                         [0,1,0,-1, 0,1,-1,-1, 1]]);# x
        t1=4./9;t2=1./9;t3=1./36;
        self.w=np.array([t1,t2,t2,t2,t2,t3,t3,t3,t3]) # D2Q9 weights in each dir
        self.fields={'tau':np.ones((*self.dim,self.nphase)),    # relaxation time
                     'u':np.zeros((*self.dim,3)),               # macroscopic velocity
                     'ueq':np.zeros((*self.dim,self.nphase,3)), # phase-velocity
                     'rho':np.ones((*self.dim,self.nphase)),    # density
                     'ns':np.zeros((*self.dim,self.nphase)),    # scattering (e.g. walls=1)
                     'Fin':np.zeros((*self.dim,self.nphase,self.ndir)),     # incoming dist
                     'Fout':np.zeros((*self.dim,self.nphase,self.ndir)),    # outgoing dist
                     'Feq':np.zeros((*self.dim,self.nphase,self.ndir)),     # equilibrium dist
                     'Fpop':np.zeros((*self.dim,self.nphase,self.ndir)),  # external forces
                     'flowMode':np.zeros((*self.dim,self.nphase))+2}
        if(tau is not None): self.fields['tau']=tau
        self.flowIdx=[]
        self.getFlowIndex()
        self.fluidPhases=[0]
        self.initDistribution();
        self.reflected=[0,3,4,1,2,7,8,5,6]  # reflection directions for D2Q9
        self.step=0
        self.status=(0,'ok')
    def initDistribution(self): # initialize the equilibrium distribution based on rho&v
        self.fields['Feq']=np.zeros((*self.dim,self.nphase,self.ndir))
        self.calcFeq();
        self.fields['Fout']=self.fields['Feq'].copy()
    def getFlowIndex(self):  # determine regions that diffuse, advect, fully N-S flow
        self.flowIdx=[]
        for pp in range(self.nphase):
            k0=np.where(self.fields['flowMode'][...,pp]==0)
            k1=np.where(self.fields['flowMode'][...,pp]==1)
            k2=np.where(self.fields['flowMode'][...,pp]>=1)
            self.flowIdx.append((k0,k1,k2))
    def calcFeq(self):  # calc the equilibrium distribution
        ueq=self.fields['ueq']
        for pp in range(self.nphase):
            u2c=1.5*(self.fields['ueq'][...,pp,:]**2).sum(axis=-1)
            k0,k1,k2=self.flowIdx[pp]
            for ii in range(self.ndir):
                cuns=3*(self.c[0,ii]*ueq[...,pp,0]+self.c[1,ii]*ueq[...,pp,1]+self.c[2,ii]*ueq[...,pp,2])
                if(k0[0].size): self.fields['Feq'][k0+(pp,ii,)]=self.w[ii]*self.fields['rho'][k0+(pp,)]
                if(k1[0].size): self.fields['Feq'][k1+(pp,ii,)]=self.w[ii]*self.fields['rho'][k1+(pp,)]*(1+cuns[k1])
                if(k2[0].size): self.fields['Feq'][k2+(pp,ii,)]=self.w[ii]*self.fields['rho'][k2+(pp,)]*(1+cuns[k2]+0.5*cuns[k2]**2-u2c[k2])
    def calcMacro(self):     # calculate macroscopic variables (hard-coded for D2Q9)
        self.fields['rho']=self.fields['Fin'].sum(axis=-1);
        F=self.fields['Fin'][...,self.fluidPhases,:]
        rhoTot=self.fields['rho'][...,self.fluidPhases].sum(axis=-1)
        with np.errstate(invalid='ignore'):
            self.fields['u'][...,2]=((F[...,1]+F[...,5]+F[...,8])-
                                     (F[...,3]+F[...,6]+F[...,7])).sum(axis=-1)/rhoTot
            self.fields['u'][...,1]=((F[...,2]+F[...,5]+F[...,6])-
                                     (F[...,4]+F[...,7]+F[...,8])).sum(axis=-1)/rhoTot
    def calcUeq(self): #  copy the velocity field 'v' into each phase of 'ueq'
        for ii in range(self.nphase):
            self.fields['ueq'][...,ii,:]=self.fields['u']
    def stream(self):   # stream outgoing distributions into incoming
        for ii in range(self.ndir):
            self.fields['Fin'][...,ii]=np.roll(self.fields['Fout'][...,ii],
                                   (self.c[0,ii],self.c[1,ii],self.c[2,ii],0),axis=(0,1,2,3))
    def collide(self):  # collide the incoming distributions into a new outgoing distribution
        invtau=np.tile(1/self.fields['tau'][...,np.newaxis],(1,1,1,1,9))
        self.fields['Fout']=invtau*self.fields['Feq']+(1-invtau)*self.fields['Fin']+self.fields['Fpop'];
    def PBB(self,ON):   # partial bounceback (Walsh-Bxxx-Saar)
        F=self.fields['Fout']
        for ii in range(self.ndir):
            k=ON+(ii,); # "on" spatial locations, append index for D2Q9 direction
            F[k]=F[k]+self.fields['ns'][ON]*(self.fields['Fin'][ON+(self.reflected[ii],)]-F[k])
    def sim(self,steps=1,callbacks=None,verbose=False):
        if(callbacks is None): callbacks={}
        for k in ['init','postStream','postMacro','postUeq','postFeq','postCollision','final']:
            if(k not in callbacks): callbacks[k]=[]
        ON=np.where(self.fields['ns'])
        self.getFlowIndex()
        for cb in callbacks['init']: cb(self)
        t0=time.time();
        for ii in range(steps):
            self.step=ii
            self.stream()
            for cb in callbacks['postStream']: cb(self) # modify distributions
            self.calcMacro()
            for cb in callbacks['postMacro']: cb(self)  # BC: set v and rho, explicitly; output
            self.calcUeq()
            for cb in callbacks['postUeq']: cb(self)    # BC: adjust ueq (e.g. Shan-Chen)
            self.calcFeq();
            for cb in callbacks['postFeq']: cb(self)    # BC: forcing functions (e.g. gravity)
            self.collide()
            for cb in callbacks['postCollision']: cb(self)    # BC: forcing functions (e.g. gravity)
            self.PBB(ON)
            if(np.any(self.fields['u']>1)):
                print('ack!: velocity too high! (step %d)'%self.step);break
        #self.step=-1  # simple state flag
        for cb in callbacks['final']: cb(self)
        if(verbose):
            print('done! (%.2fmin)'%((time.time()-t0)/60))

if(__name__=='__main__'):
    import matplotlib.pyplot as plt
    # This is an example of the workflow for a simple single-phase flow
    # with an obstacle.
    #
    # Dimensions specified/accessed in (nz,ny,nx) order.
    # instance LBM object
    S=LBM((1,30,60),nphase=1)
    
    # specify/reassign field variables such as scattering (z,y,x,phase)
    S.fields['ns'][0,10:20,10:20,0]=1
    
    # create callback to specify velocity boundary conditions
    # -- velocity is (z,y,x,3) for 3 components of velocity.  In a 2D simulation
    #    you will see [...,1] for y-component and [...,2] for x-component.
    def cb_vel(self):
        self.fields['u'][0,1:-1,0,2]=.1  # specified vel from left
        self.fields['u'][0,1:-1,-1,:]=self.fields['u'][0,1:-1,-2,:]  # "open" right
        
    # create a convenience function for plotting velocity
    def myplot(self,prefix):
        vmag=((self.fields['u'][0]**2).sum(axis=-1))**0.5
        plt.imshow(vmag);plt.title('%s:|u|'%prefix);plt.colorbar()
    
    # use that in callbacks at different stages of the simulation
    def cb_postMacro(self):
        if(self.step%50==0):
            plt.figure();myplot(self,prefix='postMacro(%d)'%self.step);plt.show()
    
    # gather callbacks to pass to the simulation method
    cb={'postMacro':[cb_vel,cb_postMacro]}
    
    # call the sim method to run for 500 steps
    S.sim(steps=500,callbacks=cb)
    
