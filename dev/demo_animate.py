# -*- coding: utf-8 -*-
"""
Different approaches to animate simulation

@author: Ted
"""

# matplotlib and animation
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

#https://stackoverflow.com/questions/42634997/how-do-i-properly-enable-ffmpeg-for-matplotlib-animation
pnffmpeg=os.path.join(os.environ['USERPROFILE'],r'FFmpeg\bin\ffmpeg.exe')
plt.rcParams['animation.ffmpeg_path'] = pnffmpeg
from matplotlib.animation import FFMpegWriter

# %% using cv2

def cb_mov(self):
    """movie callback"""
    if(self.step%20!=0): return
    
    # cv2
    img = cv2.normalize(self.v[:,:,1], None, 
                        alpha=0, beta=255, 
                        norm_type=cv2.NORM_MINMAX, 
                        dtype=cv2.CV_8U)
    self.mov.write(img)
    
# MJPG, DIVX, mp4v, XVID, X264 
fourcc=cv2.VideoWriter_fourcc(*'MP4V');fps=30
S.mov= cv2.VideoWriter('cv2_vor.mp4',fourcc, fps, (nx,ny),isColor=True)
# simulate
# S.sim...
S.mov.release()

# %% using FFmpeg
# #https://matplotlib.org/stable/gallery/animation/frame_grabbing_sgskip.html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#https://stackoverflow.com/questions/42634997/how-do-i-properly-enable-ffmpeg-for-matplotlib-animation
pnffmpeg=os.path.join(os.environ['USERPROFILE'],r'FFmpeg\bin\ffmpeg.exe')
plt.rcParams['animation.ffmpeg_path'] = pnffmpeg
from matplotlib.animation import FFMpegWriter

def cb_mov(self):
    """movie callback"""
    if(self.step%20!=0): return
    
    img =self.fields['rho'][0,:,:,0]
    
    # ffmpeg
    plt.clf()
    plt.imshow(img);plt.axis('off');plt.title(self.step)
    self.mov.grab_frame()

fps=30;dpi=100
fig = plt.figure()
metadata = dict(title='lbm-v', artist='lbm',comment='Movie!')
S.mov = FFMpegWriter(fps=fps, metadata=metadata)

with S.mov.saving(fig, "ffmpeg_mov.mp4", dpi):
    S.sim(steps=100,callbacks=[cb_pres,cb_mov])

# %% Animating functions
#https://www.geeksforgeeks.org/create-an-animated-gif-using-python-matplotlib/
from matplotlib.animation import FuncAnimation

# function takes frame as an input
def AnimationFunction(frame):
    pass

anim_created = FuncAnimation(Figure, AnimationFunction, frames=100, interval=25)

video = anim_created.to_html5_video()
html = display.HTML(video)
display.display(html)
 
# good practice to close the plt object.
plt.close()

# %% PILLOW
#https://www.geeksforgeeks.org/create-and-save-animated-gif-with-python-pillow/

# %% imageio
# https://pysource.com/2021/03/25/create-an-animated-gif-in-real-time-with-opencv-and-python/
# https://www.tutorialexample.com/python-create-gif-with-images-using-imageio-a-complete-guide-python-tutorial/
imageio.mimwrite(giffile, images_data, format= '.gif', fps = 1)

