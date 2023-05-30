import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import numpy as np
from calculate import calculate_volume
from calculate import create_3d_volume
from calculate import create_3d_plot

def resize_image(image, max_dimension):
    height, width = image.shape[:2]
    if max(height, width) <= max_dimension:
        return image
    if height > width:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    else:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


#Only use non-zero elements.
def reduce_mask_size(mask, margin):
    # Find the non-zero indices of the mask
    nonzero_indices = np.nonzero(mask)
    rows, cols = nonzero_indices

    # Determine the bounding box of non-zero elements
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # Add a margin to the bounding box
    min_row -= margin
    max_row += margin
    min_col -= margin
    max_col += margin

    # Create a smaller mask array with the reduced size
    reduced_mask = np.zeros((max_row - min_row + 1, max_col - min_col + 1), dtype=mask.dtype)

    # Copy the non-zero elements and the margin from the original mask to the reduced mask
    reduced_mask[rows - min_row, cols - min_col] = mask[rows, cols]

    return reduced_mask, (min_row, min_col)


# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

CLASS_NAMES = ['BG', 'salad','rice','chicken breast']

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model.
model.load_weights(filepath="C:/logdir/train/mask_rcnn_object_0087.h5", 
                   by_name=True)

# load the input image, convert it from BGR to RGB channel
image_top = cv2.imread("C:/Users/romax/Downloads/chickentop.jpeg")

image_top = cv2.cvtColor(image_top, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = model.detect([image_top], verbose=0)

# Get the results for the first image.
r = r[0]

# Visualize the detected objects.
mrcnn.visualize.display_instances(image=image_top, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'])
mask_data_top = {"x_y_values": [], "class_id": [], "class_masks": []}

for i in range(r["masks"].shape[2]):
    mask = r["masks"][:, :, i]
    y_pixels, x_pixels = np.where(mask)
    class_id=r["class_ids"][i]  # or results["class_ids"][i] for the class ID of the mask
    # Combine the x,y coordinates and add them to the dictionary
    x_y_values = np.zeros((len(x_pixels), 2), dtype=np.float32)
    x_y_values[:, 0] = x_pixels
    x_y_values[:, 1] = y_pixels
    mask_data_top["class_masks"].append(mask)
    mask_data_top["x_y_values"].append(x_y_values)
    mask_data_top["class_id"].append(class_id)
   
    
#Side picture
# load the input image, convert it from BGR to RGB channel
image_side = cv2.imread("C:/Users/romax/Downloads/chickenside.jpeg")
#image_side =resize_image(image_side, 1000)
# image_side = cv2.imread("C:/Users/romax/Downloads/chickenside.jpeg")
image_side = cv2.cvtColor(image_side, cv2.COLOR_BGR2RGB)

    # Perform a forward pass of the network to obtain the results
r = model.detect([image_side], verbose=0)

    # Get the results for the first image.
r = r[0]

    # Visualize the detected objects.
mrcnn.visualize.display_instances(image=image_side, 
                                      boxes=r['rois'], 
                                      masks=r['masks'], 
                                      class_ids=r['class_ids'], 
                                      class_names=CLASS_NAMES, 
                                      scores=r['scores'])
mask_data_side = {"x_y_values": [], "class_id": [], "class_masks": []}

for i in range(r["masks"].shape[2]):
        mask = r["masks"][:, :, i]
        y_pixels, x_pixels = np.where(mask)
        class_id=r["class_ids"][i]  # or results["class_ids"][i] for the class ID of the mask
        # Combine the x,y coordinates and add them to the dictionary
        x_y_values = np.zeros((len(x_pixels), 2), dtype=np.float32)
        x_y_values[:, 0] = x_pixels
        x_y_values[:, 1] = y_pixels
        mask_data_side["class_masks"].append(mask)
        mask_data_side["x_y_values"].append(x_y_values)
        mask_data_side["class_id"].append(class_id)
        
  
margin = 10
# Reduce the size of the top mask
reduced_mask_top, offset_top = reduce_mask_size(mask_data_top['class_masks'][0], margin)

# Reduce the size of the side mask
reduced_mask_side, offset_side = reduce_mask_size(mask_data_side['class_masks'][0], margin)

volume3d = create_3d_volume(reduced_mask_top, reduced_mask_side)
create_3d_plot(volume3d)



