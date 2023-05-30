
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import numpy as np
import sys
sys.path.append('C:\Programming PYTHON\MarkRcnn37Dig\Mask-RCNN-TF2\kangaroo-transfer-learning')
from calculatepixelstocmratio import calculate_pixels_to_cm_ratio
import math


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


def calculate_longest_distance(mask):
    # Convert the boolean mask to uint8
    mask_uint8 = mask.astype(np.uint8)

    # Find the contours of the mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the contour with the maximum area
    max_contour = max(contours, key=cv2.contourArea)

    # Get the coordinates of the contour points
    contour_points = max_contour.squeeze()

    # Initialize variables to store the maximum distance and corresponding points
    max_distance = 0.0
    max_point_1 = None
    max_point_2 = None

    # Iterate over all pairs of points and calculate the Euclidean distance
    for i in range(len(contour_points)):
        for j in range(i+1, len(contour_points)):
            point_1 = contour_points[i]
            point_2 = contour_points[j]
            distance = np.sqrt(np.sum((point_1 - point_2) ** 2))

            # Update the maximum distance and corresponding points if a longer distance is found
            if distance > max_distance:
                max_distance = distance
                max_point_1 = point_1
                max_point_2 = point_2

    return max_distance, max_point_1, max_point_2




# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

CLASS_NAMES = ['BG', 'plate']

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
model.load_weights(filepath="C:/logdir/train/mask_rcnn_plateob_0082.h5", 
                   by_name=True)

# load the input image, convert it from BGR to RGB channel
image = cv2.imread("C:/Users/romax/Downloads/emptyplate.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



# Perform a forward pass of the network to obtain the results
r = model.detect([image], verbose=0)


# Get the results for the first image.
r = r[0]


# Filter out instances with confidence score below 0.82
high_conf_indices = np.where(r['scores'] >= 0.82)[0]
filtered_boxes = r['rois'][high_conf_indices]
filtered_masks = r['masks'][:, :, high_conf_indices]
filtered_class_ids = r['class_ids'][high_conf_indices]
filtered_scores = r['scores'][high_conf_indices]

# Visualize the detected objects.
mrcnn.visualize.display_instances(image=image, 
                                  boxes=filtered_boxes, 
                                  masks=filtered_masks, 
                                  class_ids=filtered_class_ids, 
                                  class_names=CLASS_NAMES, 
                                  scores=filtered_scores)

#Get X,Y of first mask
# Get the mask for the first object instance

#loop through dict to class_ids length(for each)
mask = r['masks'][:, :, 0]

# Find the indices where the mask values are non-zero
nonzero_indices = np.nonzero(mask)

# Extract the x and y coordinates
x_coords = nonzero_indices[1]
y_coords = nonzero_indices[0]

pixel_to_cm_info = calculate_pixels_to_cm_ratio(r,27.5,1)#Mask,Plate_Width_Cm,PlateID
pixel_to_cm_ratio = pixel_to_cm_info[0]
pixel_width_plate = pixel_to_cm_info[1]

print("Pixel to CM: :" ,pixel_to_cm_ratio)
#longest_pixels_distance = calculate_longest_distance(mask)
#print(longest_pixels_distance) #In Cm. [length in cm,max x, max y]
print("your plate is ", pixel_width_plate/pixel_to_cm_ratio , " cm")
print("your plate is ", pixel_width_plate , " pixels")


