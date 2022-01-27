# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 22:05:06 2022

@author: Ted
"""
import matplotlib.pyplot as plt

def plotf(F,**kwargs):
    """plot distribution"""
    pp=[(1,6),(2,2),(3,5),(4,3),(5,0),(6,1),(7,7),(8,4),(9,8)]
    for ip,ii in pp:
        plt.subplot(3,3,ip);plt.imshow(F[:,:,ii],**kwargs);plt.axis('off');