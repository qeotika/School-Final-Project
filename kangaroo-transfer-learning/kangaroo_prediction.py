import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import numpy as np

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
model.load_weights(filepath="C:/logs/train/mask_rcnn_object_0087.h5", 
                   by_name=True)

# load the input image, convert it from BGR to RGB channel
image = cv2.imread("testttttttt.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#added
padding =5

#added

# Perform a forward pass of the network to obtain the results
r = model.detect([image], verbose=0)

#added to load mask


#added

# Get the results for the first image.
r = r[0]

# Visualize the detected objects.
mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'])

#Get X,Y of first mask
# Get the mask for the first object instance

#loop through dict to class_ids length(for each)
mask = r['masks'][:, :, 0]

# Find the indices where the mask values are non-zero
nonzero_indices = np.nonzero(mask)

# Extract the x and y coordinates
x_coords = nonzero_indices[1]
y_coords = nonzero_indices[0]