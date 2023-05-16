import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy import interpolate
from scipy import ndimage

# create two 2D masks
mask_top = np.array([[1, 1, 0, 1], 
                     [1, 0, 0, 0], 
                     [1, 0, 0, 0], 
                     [0, 0, 0, 0]])

mask_side = np.array([[1, 0, 0, 0], 
                      [1, 0, 0, 1], 
                      [1, 0, 0, 0], 
                      [1, 0, 0, 0]])

# Interpolate the 2D masks
x = np.arange(mask_top.shape[0])
y = np.arange(mask_top.shape[1])
x_new = np.linspace(0, mask_top.shape[0]-1, 100)
y_new = np.linspace(0, mask_top.shape[1]-1, 100)
f_top = interpolate.interp2d(x, y, mask_top, kind='linear')
f_side = interpolate.interp2d(x, y, mask_side, kind='linear')
mask_top_interp = f_top(x_new, y_new)
mask_side_interp = f_side(x_new, y_new)

# stack them to create a 3D volume [No Interpolation]
#volume = np.stack([mask_top, mask_side])

# Create the 3D volume using 2D images andd interpolation
volume = np.zeros((100, 100, 100))
for i in range(volume.shape[0]):
    for j in range(volume.shape[1]):
        for k in range(volume.shape[2]):
            if mask_top_interp[i, j] > k/100 and mask_side_interp[i, j] > k/100:
                volume[i, j, k] = 1

# visualize the 3D volume [Full Volume]

#Plot the 3D volume with less pixels

# Remove empty voxels
eroded = ndimage.binary_erosion(volume)
nonzero_voxels = np.transpose(np.nonzero(eroded))
xmin, ymin, zmin = nonzero_voxels.min(axis=0)
xmax, ymax, zmax = nonzero_voxels.max(axis=0)
volume = volume[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot voxels
ax.voxels(volume, edgecolor='k')

# Set limits and show plot
ax.set_xlim3d(0, volume.shape[0])
ax.set_ylim3d(0, volume.shape[1])
ax.set_zlim3d(0, volume.shape[2])
plt.show()

# calculate Volume Based on Voxels (We need to re-arange it.)

dx, dy, dz = 1.0, 1.0, 1.0  # Voxel dimensions in some units x y z 
occupied_voxels = np.count_nonzero(volume)
total_volume = occupied_voxels * dx * dy * dz
print("Total voxel volume of object:", total_volume)










# Plot the 3D volume
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.voxels(volume, edgecolor='k')
# plt.show()


# visualize the 3D volume from 2D images
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# verts, faces, _, _ = measure.marching_cubes_lewiner(volume, level=0)
# mesh = Poly3DCollection(verts[faces], alpha=0.3)
# mesh.set_facecolor([0, 1, 0])
# ax.add_collection3d(mesh)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# ax.set_xlim(0, volume.shape[1])
# ax.set_ylim(0, volume.shape[0])
# ax.set_zlim(0, volume.shape[2])

# plt.show()

