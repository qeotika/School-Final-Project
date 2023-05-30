import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_3d_volume(top_mask, side_mask):
    # Find the non-zero indices in the masks
    top_indices = np.nonzero(top_mask)
    side_indices = np.nonzero(side_mask)

    # Determine the dimensions of the 3D volume
    volume_shape = (max(np.max(top_indices[0]), np.max(side_indices[0])) + 1,
                    max(np.max(top_indices[1]), np.max(side_indices[1])) + 1,
                    max(np.max(top_indices[1]), np.max(side_indices[1])) + 1)

    # Create an empty 3D volume
    volume = np.zeros(volume_shape, dtype=bool)

    # Fill the top mask in the volume array
    volume[top_indices] = True

    # Fill the side mask in the volume array
    volume[side_indices[0], side_indices[1], -side_indices[1]] = True

    return volume


# Example usage
mask_top = np.array([[0, 0, 0, 0],
                     [0, 0, 0, 1], #Right is the TOP RIGHT UPPER BUILD 
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])

mask_side = np.array([[0, 0, 0, 1], #Right colum is the lower x,ys
                      [0, 0, 0, 1],
                      [0, 0, 0, 1],
                      [0, 0, 0, 1]])

# Create the 3D object
volume = create_3d_volume(mask_top, mask_side)

# Create meshgrid for X, Y, Z axes
X, Y, Z = np.meshgrid(np.arange(volume.shape[2]),
                       np.arange(volume.shape[1]),
                       np.arange(volume.shape[0]))

# Visualize the 3D volume
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.voxels(volume, facecolors='green', edgecolors='k')  # Plot the voxels

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

