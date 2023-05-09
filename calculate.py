import numpy as np
import pyvista as pv
from skimage import measure


# Create a mesh using the Marching Cubes algorithm
def create_mesh(x_pixels, y_pixels):
    # Create a binary image from the x,y pixels
    image = np.zeros((y_pixels.max()+1, x_pixels.max()+1), dtype=np.bool)
    image[y_pixels, x_pixels] = True

    # Use the Marching Cubes algorithm to create a mesh from the binary image
    verts, faces, _, _ = measure.marching_cubes(image, level=0.5)

    # Create a PyVista mesh object from the vertices and faces
    mesh = pv.PolyData(verts, faces)

    return mesh


mask1 = [[1,2],[3,4],[12,52],[12,34]]
mask2= [[7,44],[8,12],[15,18],[19,42]]
# Create meshes from the mask data
mesh1 = create_mesh(mask1[:, 0], mask1[:, 1])
mesh2 = create_mesh(mask2[:, 0], mask2[:, 1])
# Combine the meshes
mesh = mesh1 + mesh2

# create a pyvista mesh object
pv_mesh = pv.PolyData(mesh.points, mesh.faces)

# plot the mesh
pv_mesh.plot()