# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 19:35:11 2021

https://exolete.com/lbm/
simple 2D LBM
@author: Ted
"""
import numpy as np
import time
import cv2
import matplotlib.cm as cm

# matplotlib and animation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#https://stackoverflow.com/questions/42634997/how-do-i-properly-enable-ffmpeg-for-matplotlib-animation
plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\Ted\MyApps\FFmpeg\bin\ffmpeg.exe'
from matplotlib.animation import FFMpegWriter

# custom 
import importlib
import pylbm2
importlib.reload(pylbm2)
import boundaryConditions as BC
import shanChen
import plotlbm

nx=30;ny=30;
scl=1
nx=nx//scl;ny=ny//scl;



# %%
nphase=2
nu=.16
tau=3*nu+.5
omega=1/tau
S=pylbm2.LBM((ny,nx),nphase=nphase)
S.omega=omega
S.fields['rho'][10:20,10:20,0]=0
S.fields['rho'][:,:,1]=1-S.fields['rho'][:,:,0]

S.initDistribution();


cb={'postStream':[],
    'postMacro':[]}
S.sim(steps=2,callbacks=cb)
