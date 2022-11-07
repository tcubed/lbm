"""simple LBM: 2D, that can be extended to 3D, multiphase, etc through callbacks
@author: Ted Tower
"""
import numpy as np
import time

import numba
from numba import jit,njit
from numba import uint8,int8,int32, int64,uint16,uint64,float32,float64    # import the types
from numba.experimental import jitclass

from numba.typed import List, Dict

#
# Fields class
#
fldSpec=[('tau',float64[:,:,:,:]),
         ('rho',float64[:,:,:,:]),
         ('ns',float64[:,:,:,:]),
         
         ('u',float64[:,:,:,:]),
         ('ueq',float64[:,:,:,:,:]),
         
         ('Feq',float64[:,:,:,:,:]),
         ('Fin',float64[:,:,:,:,:]),
         ('Fout',float64[:,:,:,:,:]),
         ('Fpop',float64[:,:,:,:,:]),
         ]
@jitclass(fldSpec)
class Fields():
    nphase: uint8
    ndir: uint8
    
    def __init__(self,nzyx,nphase,ndir):
        nz,ny,nx=nzyx
        dph=(nz,ny,nx,nphase)
        dv=(nz,ny,nx,3)
        dphv=(nz,ny,nx,nphase,3)
        dF=(nz,ny,nx,nphase,ndir)
        
        self.tau=np.ones(dph,dtype=np.float64)
        self.rho=np.ones(dph,dtype=np.float64)
        self.ns=np.zeros(dph,dtype=np.float64)
        
        self.u=np.zeros(dv,dtype=np.float64) 
        self.ueq=np.zeros(dphv,dtype=np.float64) 
        
        self.Fin=np.zeros(dF,dtype=np.float64)     # incoming dist
        self.Fout=np.zeros(dF,dtype=np.float64)    # outgoing dist
        self.Feq=np.zeros(dF,dtype=np.float64)     # equilibrium dist
        self.Fpop=np.zeros(dF,dtype=np.float64)  # external forces
        



spec=[('c',int8[:,:]),
      ('w',float64[:]),
      ('reflected',uint8[:]),
      ('fluidPhases',uint8[:]),
      ]
@jitclass(spec)
class LBM():
    dim: numba.typeof((int64(1),int64(1),int64(1)))
    nphase: uint8
    ndir: uint8
    fields: Fields
    step: uint16
    status: uint8
    statusmsg: str
    
    def __init__(self,nzyx,nphase=1,ndir=9):
    #def __init__(self):    
        nzyx=(1,30,30);nphase=1;ndir=9
        nz,ny,nx=nzyx
        self.dim=(uint64(nz),uint64(ny),uint64(nx))
        self.nphase=uint8(nphase)
        self.ndir=9
        self.c=np.array([[0,0,0, 0, 0,0, 0, 0, 0],  # z -- direction vectors for D2Q9
                          [0,0,1, 0,-1,1, 1,-1,-1],  # y
                          [0,1,0,-1, 0,1,-1,-1, 1]],dtype=int8);# x
        t1=4./9;t2=1./9;t3=1./36;
        self.w=np.array([t1,t2,t2,t2,t2,t3,t3,t3,t3],dtype=float64) # D2Q9 weights in each dir
        
        self.fields=Fields(nzyx,nphase,ndir)
        
        # self.flowIdx=[]
        # self.getFlowIndex()
        
        self.fluidPhases=np.arange(nphase,dtype=uint8)
        self.initDistribution();
        self.reflected=np.array([0,3,4,1,2,7,8,5,6],dtype=uint8)  # reflection directions for D2Q9
        self.step=uint16(0)
        self.status=uint8(0)
        self.statusmsg=''
        
    
    def initDistribution(self): # initialize the equilibrium distribution based on rho&v
        self.fields.Feq=np.zeros((*self.dim,self.nphase,self.ndir),dtype=np.float64)
        self.calcFeq();
        self.fields.Fout=self.fields.Feq.copy()
    # def getFlowIndex(self):  # determine regions that diffuse, advect, fully N-S flow
    #     self.flowIdx=[]
    #     for pp in range(self.nphase):
    #         k0=np.where(self.fields['flowMode'][...,pp]==0)
    #         k1=np.where(self.fields['flowMode'][...,pp]==1)
    #         k2=np.where(self.fields['flowMode'][...,pp]>=1)
    #         self.flowIdx.append((k0,k1,k2))
    def calcFeq(self):  # calc the equilibrium distribution
        ueq=self.fields.ueq
        for pp in range(self.nphase):
            u2c=1.5*(self.fields.ueq[...,pp,:]**2).sum(axis=-1)
            #k0,k1,k2=self.flowIdx[pp]
            for ii in range(self.ndir):
                cuns=3*(self.c[0,ii]*ueq[...,pp,0]+self.c[1,ii]*ueq[...,pp,1]+self.c[2,ii]*ueq[...,pp,2])
                #if(k0[0].size): self.fields.Feq[k0+(pp,ii,)]=self.w[ii]*self.fields.rho[k0+(pp,)]
                #if(k1[0].size): self.fields.Feq[k1+(pp,ii,)]=self.w[ii]*self.fields.rho[k1+(pp,)]*(1+cuns[k1])
                #if(k2[0].size): self.fields.Feq[k2+(pp,ii,)]=self.w[ii]*self.fields.rho[k2+(pp,)]*(1+cuns[k2]+0.5*cuns[k2]**2-u2c[k2])
                self.fields.Feq[...,pp,ii]=self.w[ii]*self.fields.rho[...,pp]*(1+cuns+0.5*np.power(cuns,2)-u2c)
                
    def calcMacro(self):     # calculate macroscopic variables (hard-coded for D2Q9)
        self.fields.rho=self.fields.Fin.sum(axis=-1);
        F=self.fields.Fin[...,self.fluidPhases,:]
        tau=self.fields.tau[...,self.fluidPhases]
        rho=self.fields.rho[...,self.fluidPhases]
        rhoOmegaTot=(rho/tau).sum(axis=-1)
        # with np.errstate(invalid='ignore'):
        #     ux=((F[...,1]+F[...,5]+F[...,8])-(F[...,3]+F[...,6]+F[...,7]))
        #     self.fields['u'][...,2]=(ux/tau).sum(axis=-1)/rhoOmegaTot
        #     uy=((F[...,2]+F[...,5]+F[...,6])-(F[...,4]+F[...,7]+F[...,8]))
        #     self.fields['u'][...,1]=(uy/tau).sum(axis=-1)/rhoOmegaTot
        #with np.errstate(invalid='ignore'):
        ux=((F[...,1]+F[...,5]+F[...,8])-(F[...,3]+F[...,6]+F[...,7]))
        self.fields.u[...,2]=(ux/tau).sum(axis=-1)/(rhoOmegaTot+1e-12)
        uy=((F[...,2]+F[...,5]+F[...,6])-(F[...,4]+F[...,7]+F[...,8]))
        self.fields.u[...,1]=(uy/tau).sum(axis=-1)/(rhoOmegaTot+1e-12)
        
        # zero out velocity in solid
        for dd in [0,1,2]:
            self.fields.u[...,dd]*=(1-self.fields.ns[...,0])
        
    def calcUeq(self): #  copy the velocity field 'v' into each phase of 'ueq'
        for ii in range(self.nphase):
            self.fields.ueq[...,ii,:]=self.fields.u
    def stream(self):   # stream outgoing distributions into incoming
        Fout=self.fields.Fout
        nz,ny,nx,nph,ndir=Fout.shape
        iz=np.arange(nz)
        iy=np.arange(ny)
        ix=np.arange(nx)
        for ii in range(self.ndir):
            #self.fields['Fin'][...,ii]=np.roll(self.fields['Fout'][...,ii],
            #                        (self.c[0,ii],self.c[1,ii],self.c[2,ii],0),axis=(0,1,2,3))
            # manual roll...
            Fout[...,ii]=Fout[iz[(np.arange(nz)+self.c[0,ii])%nz],:,:,:,ii]
            Fout[...,ii]=Fout[:,iy[(np.arange(ny)+self.c[1,ii])%ny],:,:,ii]
            Fout[...,ii]=Fout[:,:,ix[(np.arange(nx)+self.c[2,ii])%nx],:,ii]
        self.fields.Fin=Fout  
            
    def collide(self):  # collide the incoming distributions into a new outgoing distribution
        #https://stackoverflow.com/questions/61686293/numba-compatible-implementation-of-np-tile
        #invtau=np.tile(1/self.fields['tau'][...,np.newaxis],(1,1,1,1,9))
        tau=self.fields.tau
        invtau=(1/tau).repeat(self.ndir).reshape((*tau.shape,int64(self.ndir)))
        self.fields.Fout=invtau*self.fields.Feq+(1-invtau)*self.fields.Fin+self.fields.Fpop;
    
    def PBB(self,ONF):   # partial bounceback (Walsh-Bxxx-Saar)
        nsf=self.fields.ns.flatten()
        for ii in range(self.ndir):
            FlatOut=self.fields.Fout[...,ii].flatten()      # outgoing
            # incoming, reflected
            FflatInRef=self.fields.Fin[...,self.reflected[ii]].flatten()
            # PBB
            FlatOut[ONF]=FlatOut[ONF]+nsf[ONF]*(FflatInRef[ONF]-FlatOut[ONF])
            self.fields.Fout[...,ii]=FlatOut.reshape(self.fields.Fout[...,ii].shape)
        
    def sim(self,steps,callbacks,verbose=False
            ):
        # https://numba.discourse.group/t/typed-list-of-jitted-functions-in-jitclass/413/2
        # https://stackoverflow.com/questions/68318702/numba-jit-with-list-of-functions-as-function-argument
        
        ONF=np.where(self.fields.ns.flatten())
        # #self.getFlowIndex()
        for cbt,cb in callbacks:
            if(cbt=='init'): cb(self)
            
        #t0=time.time();
        for ii in range(steps):
            self.step=uint16(ii)
            self.stream()               # STREAM
            for cbt,cb in callbacks:
                if(cbt=='postStream'): cb(self)
            self.calcMacro()            # CALCULATE MACROSCOPIC
            for cbt,cb in callbacks:    # ...BC: set v and rho, explicitly; output
                if(cbt=='postMacro'): cb(self) 
            self.calcUeq()              # EQUILIBRIUM VELOCITY
            for cbt,cb in callbacks:    # ...BC: adjust ueq (e.g. Shan-Chen)
                if(cbt=='postUeq'): cb(self)   
            self.calcFeq();             # EQUILIBRIUM DISTRIBUTION
            for cbt,cb in callbacks:    # ...BC: forcing functions (e.g. gravity)
                if(cbt=='postFeq'): cb(self)   
            self.collide()              # COLLISION
            for cbt,cb in callbacks:    # ...sinks & sources
                if(cbt=='postCollision'): cb(self)
        #     for cb in callbacks['postCollision']: cb(self)    
            self.PBB(ONF)
        #     if(np.any(self.fields['u']>1)):
        #         print('ack!: velocity too high! (step %d)'%self.step);break
        # #self.step=-1  # simple state flag
        for cbt,cb in callbacks:    # wrap up
            if(cbt=='final'): cb(self)
        if(verbose):
            print('done!')
        # if(verbose):
        #     print('done! (%.2fmin)'%((time.time()-t0)/60))



@njit()
def cb_test(self):
    print('test!')
    #return self

if(__name__=='__main__'):
    import matplotlib.pyplot as plt
    # This is an example of the workflow for a simple single-phase flow
    # with an obstacle.
    #
    # Dimensions specified/accessed in (nz,ny,nx) order.
    # instance LBM object
    #S=LBM((1,30,60),1,9)
    #S=LBM()
    # specify/reassign field variables such as scattering (z,y,x,phase)
    #S.fields.ns[0,10:20,10:20,0]=1
    
    # # create callback to specify velocity boundary conditions
    # # -- velocity is (z,y,x,3) for 3 components of velocity.  In a 2D simulation
    # #    you will see [...,1] for y-component and [...,2] for x-component.
    # def cb_vel(self):
    #     self.fields['u'][0,1:-1,0,2]=.1  # specified vel from left
    #     self.fields['u'][0,1:-1,-1,:]=self.fields['u'][0,1:-1,-2,:]  # "open" right
        
    # # create a convenience function for plotting velocity
    # def myplot(self,prefix):
    #     vmag=((self.fields['u'][0]**2).sum(axis=-1))**0.5
    #     plt.imshow(vmag);plt.title('%s:|u|'%prefix);plt.colorbar()
    
    # # use that in callbacks at different stages of the simulation
    # def cb_postMacro(self):
    #     if(self.step%50==0):
    #         plt.figure();myplot(self,prefix='postMacro(%d)'%self.step);plt.show()
    
    # # gather callbacks to pass to the simulation method
    # cb={'postMacro':[cb_vel,cb_postMacro]}
    
    # call the sim method to run for 500 steps
    #cb=[]
    
    
    #S.sim(steps=500,callbacks=None)
    #S.sim(steps=500)
    #S.sim(5)
    
    
    # DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
    start = time.perf_counter()
    S=LBM((1,300,600),1,9)
    f1=cb_test(S)
    calcFeq(S)
    
    
    
    #cb=Dict()
    from numba.core import types
    #cb = Dict.empty(
    #    key_type=types.unicode_type,
    #    value_type=List(),
    #)
    
    #cb['init']=List()
    #cb['init'].append(f1)
    cb=List()
    #cb.append(f1)
    cb.append(('init',cb_test))
    
    S.fields.ns[0,10:20,10:20,0]=1
    S.sim(1,cb)
    end = time.perf_counter()
    mlups=300*600*5000/1e6/(end-start)
    print("Elapsed (with compilation) = {}s ({:.1f}mlups)".format((end - start),mlups))
    
    # NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
    start = time.perf_counter()
    S=LBM((1,300,600),1,9)
    S.fields.ns[0,10:20,10:20,0]=1
    S.sim(5000,cb)
    end = time.perf_counter()
    mlups=300*600*5000/1e6/(end-start)
    print("Elapsed (after compilation) = {}s ({:.1f}mlups)".format((end - start),mlups))
