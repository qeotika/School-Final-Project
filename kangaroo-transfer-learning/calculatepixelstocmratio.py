import cv2
import numpy as np
import math

#Find the class ID for "Plate"
class_id = 0 
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

#How many pixels per cm 
def calculate_pixels_to_cm_ratio(mask_data,plate_width_cm,class_id):
    # inch_to_cm_ratio=2.54
    class_masks = mask_data['masks'][:, :, 0]
    nonzero_indices = np.nonzero(class_masks)
    x_coords = nonzero_indices[1]
    print(x_coords)
    class_ids = mask_data['class_ids']
    #Check whether there's a plate in the image
    class_indicies = np.where(class_ids == class_id)[0]
    if len(class_indicies) == 0:
        print("No plate has been found")
        return None
    
    mask = class_masks  #The masks present   
    plate_width_pixels = calculate_longest_distance(mask)[0] #Take max_distance ONLY
    pixels_to_cm_ratio = plate_width_pixels/plate_width_cm #Ratio of pixels to CM
    return pixels_to_cm_ratio,plate_width_pixels #X Pixels = 1 CM


