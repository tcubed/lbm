# LBM
Simple Lattice Boltzmann Method (LBM) package for python.

## Introduction

Simulating mass, heat, species transfer in fluids and solids is a huge part of engineering,
visualization, gaming, and more.  LBM is one approach to simulate simple fluids.  There are 
many powerful packages for simulating fluids, but that is often too complex for discovery.

*Goal*: provide a package that allows users to discover and be creative with fluids in a
(hopefully) easy way.

### Module Overview

The following are the main modules of interest:

 - pylbm.py: this is the LBM engine, with a simple flow scenario.  The engine is ~100 lines of code.
 - boundaryConditions.py: for more elaborate boundary conditions, such as Zhou-He pressure and velocity boundary conditions.
 - callbacks.py: several common callback examples for reporting and other ways of specifying boundary conditions.

In addition are some supplementary modules:

 - test_poiseuille.py: a unit test for pressure and gravity Poiseuille flow in north-south and east-west directions.  This is to confirm that we haven't broken some basic functionality.
 - plotlbm.py: a convenience plotting function (under development)
 - dev/ folder: where I store some scripts/modules also under development.

# A working example

This is an example of single-phase flow with an obstacle.  It is a more 
descriptive version of what is in pylbm.py

Dimensions are specified and accessed in (z,y,x) order.  After imports, 
we instance an LBM object with dimensions and optionally, the number of phases.
Here, we create an object for a 2D simulation with x=60, y=30 (and z=1) lattice points.

    import matplotlib.pyplot as plt
    from pylab import LBM
    # Dimensions specified/accessed in (nz,ny,nx) order.
    # instance LBM object
    S=LBM((1,30,60),nphase=1)

We can specify and reassign "field" variables (i.e. anything that varies over
the domain).  Fields are often phase-related with size (n_z,n_y,n_x,n_phase) or vectors
(e.g. velocity) with size (n_z,n_y,n_x,3) or special distribution fields with 
size (n_z,n_y,n_x,n_phase,n_quantizedDirections).  Here, we create an solid obstacle 
using the ns scattering parameter, where a 1 reflects a solid boundary (although 
a porous media can be modeled this way with a value between 0-1).

    # specify/reassign field variables such as scattering (z,y,x,phase)
    S.fields['ns'][0,10:20,10:20,0]=1

Additional functionality is provided through generic callback functions.  A callback
can be almost anything!  Common uses are setting boundary conditions (e.g. velocity
or pressure), adding new physics (e.g. Shan-Chen multiphase), or reporting and plotting.
Examples and some common ones are availably in the callbacks module, but feel free to 
write your own.

Here, we create a callback to specify velocity boundary conditions.  To specify an
incoming velocity on the left boundary, we just assign that velocity.  To specify an 
"open" condition on the right, we just copy the velocity just to the left.  This callback
will set these velocities every step of the simulation.

    def cb_vel(self):
        self.fields['v'][0,1:-1,0,2]=.1  # specified vel from left
        self.fields['v'][0,1:-1,-1,:]=self.fields['v'][0,1:-1,-2,:]  # "open" right

There isn't a formal callback template or mechanism.  However, callbacks expects
the caller "self" as the only argument.  You can easily provide other convenience
functions for possibly multiple callbacks.

    # create a convenience function for plotting velocity
    def myplot(self,prefix):
        vmag=((self.fields['v'][0]**2).sum(axis=-1))**0.5
        plt.imshow(vmag);plt.title('%s:|v|'%prefix);plt.colorbar()
    
Here, we wrap that plotting function inside our callback that will be called
after the macroscopic variables are calculated.  We'll only plot every 50
steps though.
    
    # use that in callbacks at different stages of the simulation
    def cb_postMacro(self):
        if(self.step%50==0):
            plt.figure();myplot(self,prefix='postMacro(%d)'%self.step);plt.show()

All callbacks are gathered together using one of the allowed entry points: 'init',
'postStream','postMacro','postUeq','postFeq','postCollision', and 'final'.

    # gather callbacks to pass to the simulation method
    cb={'postMacro':[cb_vel,cb_postMacro]}
    
Finally, we call the simulation method to run for 500 steps using our callbacks.
    
    # call the sim method to run for 500 steps
    S.sim(steps=500,callbacks=cb)
    
The aim for this is to be pretty simple and extensible to use in various projects.

Have fun!
