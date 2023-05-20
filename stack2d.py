# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:20:41 2023

@author: romax
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy import interpolate
from scipy import ndimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# create two 2D masks
mask_top = np.zeros((20, 20))
mask_top[2:20, 5:15] = 1

# Create a side mask
mask_side = np.zeros((20, 20))
mask_side[1:20, 5:15] = 1

#mask_side = np.rot90(mask_side)
volume = np.stack([mask_top, mask_side])



#visualize the 3D volume from 2D images
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

verts, faces, _, _ = measure.marching_cubes_lewiner(volume, level=0)
mesh = Poly3DCollection(verts[faces], alpha=0.3)
mesh.set_facecolor([0, 1, 0])
ax.add_collection3d(mesh)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim(0, volume.shape[1])
ax.set_ylim(0, volume.shape[0])
ax.set_zlim(0, volume.shape[2])

# plt.show()