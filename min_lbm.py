# -*- coding: utf-8 -*-
"""
Minimal LBM simulation
@author: Ted
"""
import matplotlib.pyplot as plt
import importlib

import pylbm
importlib.reload(pylbm)

# instance LBM object with ny=30,nx=60
S=pylbm.LBM((30,60))

# assign a solid object
S.solid[10:20,15:20]=1

# create a callback to assign velocity at a point in the domain
def cb_vel(self):
    self.v[:,0,1]=.1
    
# simulate for 500 steps
S.sim(steps=500,callbacks=[cb_vel])

# plot the X velocity
plt.imshow(S.v[:,:,1])