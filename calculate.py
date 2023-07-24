import numpy as np
import matplotlib.pyplot as plt
import cv2
#from skimage import measure
# from scipy import interpolate
# from scipy import ndimage


# import numpy as np
# from skimage import measure
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
def resize_masks_to_max_dimension(top_mask, side_mask):
    # Find the maximum dimensions among the top and side masks
    top_height, top_width = top_mask.shape[:2]
    side_height, side_width = side_mask.shape[:2]
    max_width = max(top_width, side_width)
    max_height = max(top_height, side_height)

    # Set the desired size to the maximum dimensions
    desired_size = (max_width, max_height)

    # Convert masks to np.uint8 data type
    top_mask = (top_mask * 255).astype(np.uint8)
    side_mask = (side_mask * 255).astype(np.uint8)

    # Resize the top mask to the desired size
    resized_top_mask = cv2.resize(top_mask, desired_size, interpolation=cv2.INTER_LINEAR)

    # Resize the side mask to the desired size
    resized_side_mask = cv2.resize(side_mask, desired_size, interpolation=cv2.INTER_LINEAR)

    # Normalize the resized masks back to the range of 0 and 1
    resized_top_mask = resized_top_mask.astype(np.float32) / 255.0
    resized_side_mask = resized_side_mask.astype(np.float32) / 255.0

    return resized_top_mask, resized_side_mask

def create_3d_volume(top_mask, side_mask):
    # Find the non-zero indices in the masks
    top_mask,side_mask = resize_masks_to_max_dimension(top_mask, side_mask)
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

#Calculate the volume on the object 
def calculate_volume(volume, voxel_size):
    # Calculate the volume of the object
    voxel_volume = np.prod(voxel_size)
    object_volume = np.sum(volume) * voxel_volume
    return object_volume

#Create 3D recreation of that object
def create_3d_plot(volume):
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



