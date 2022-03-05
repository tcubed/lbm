# -*- coding: utf-8 -*-
"""
Minimal LBM simulation
@author: Ted
"""
import matplotlib.pyplot as plt
from numba import njit
from numba.typed import List
import importlib

import pylbm_numba
importlib.reload(pylbm_numba)

# instance LBM object with ny=30,nx=60
S=pylbm_numba.LBM(30,60)

# assign a solid object
S.solid[10:20,15:20]=1

# create a callback to assign velocity at a point in the domain
@njit
def cb_vel(self):
    self.v[:,0,1]=.1
    
# simulate for 500 steps
#S.sim(steps=500,callbacks=[cb_vel])
cb=List()
cb.append(cb_vel)
S.sim(500,cb)

# plot the X velocity
plt.imshow(S.v[:,:,1])