# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 20:14:14 2022

@author: Ted

https://google.github.io/mediapipe/solutions/selfie_segmentation.html
https://stackoverflow.com/questions/72706073/attributeerror-partially-initialized-module-cv2-has-no-attribute-gapi-wip-gs
"""

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

#from numba.typed import List
import time
import pylbm_numba as pylbm

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation


#
# init and compile LBM
#
print('init LBM!')
start = time.perf_counter()
S=pylbm.LBM((1,48,64),1,9)
pylbm.stream(S)
pylbm.calcUeq(S)
pylbm.calcFeq(S)
#pylbm.callbacks(S,'test')

#rom numba.core import types
#cb=List()
#cb.append(('init',pylbm.cb_test))

#S.fields.ns[0,10:20,10:20,0]=1
S.sim(1)
end = time.perf_counter()
mlups=48*64*1/1e6/(end-start)
print("Elapsed (with compilation) = {}s ({:.1f}mlups)".format((end - start),mlups))
 
# DO LOOP OUTSIDE OF NUMBA
start = time.perf_counter()
S=pylbm.LBM((1,48,64),1,9)
#S.fields.ns[0,10:20,10:20,0]=1
for ii in range(10):
    S.sim(50)
end = time.perf_counter()
mlups=48*64*500/1e6/(end-start)
print("Elapsed,outside loop (after compilation) = {}s ({:.1f}mlups)".format((end - start),mlups))


# %%

# create initial grid of particles


#
# START CAMERA!
#
print('start camera!')
BG_COLOR = (192, 192, 192) # gray
cap = cv2.VideoCapture(0)
with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1) as selfie_segmentation:
    bg_image = None
    while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
    
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = selfie_segmentation.process(image)
    
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        RM=cv2.resize(results.segmentation_mask,(64,48))
        S.fields.ns[0,:,:,0]=(RM>.2).astype(float)
        S.fields.u[0,:,:,1]=0
        S.fields.u[0,:,:,2]=0
        S.sim(50)
        
        uy=S.fields.u[0,:,:,1];ux=S.fields.u[0,:,:,2]
        umag=(ux**2+uy**2)**0.5
        umag=np.clip(umag*1000,0,255)
        
        BG=cv2.resize(umag.astype(np.uint8),(640,480))
        
        bg_image = cv2.applyColorMap(BG, cv2.COLORMAP_VIRIDIS)
        
        #mx,my=np.meshgrid(range(nx),range(ny))
        #plt.figure()
        #plt.imshow(umag,cmap='viridis',origin='lower')
        #skip=2
        #plt.quiver(mx[0::skip,0::skip],my[0::skip,0::skip],ux[0::skip,0::skip],uy[0::skip,0::skip])
        
        
        
        # Draw selfie segmentation on the background image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack(
          (results.segmentation_mask,) * 3, axis=-1) > 0.1
        # The background can be customized.
        #   a) Load an image (with the same width and height of the input image) to
        #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
        #   b) Blur the input image by applying image filtering, e.g.,
        #      bg_image = cv2.GaussianBlur(image,(55,55),0)
        # if bg_image is None:
        #     #bg_image = np.zeros(image.shape, dtype=np.uint8)
        #     #bg_image[:] = BG_COLOR
        #     vmag=(S.fields.u[0,:,:,1]**2+S.fields.u[0,:,:,2]**2)**0.5
        #     bg_image=(vmag*100).astype(np.uint8)
        #bg_image=BG.reshape((480,640,1)).repeat(3,axis=2)
        output_image = np.where(condition, image, bg_image)
    
        cv2.imshow('MediaPipe Selfie Segmentation', output_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()