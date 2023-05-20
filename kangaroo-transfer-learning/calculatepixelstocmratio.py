import cv2
import numpy as np

#Find the class ID for "Plate"
class_id = 0 

def calculate_pixels_to_cm_ratio(mask_data,plate_width_inches,class_id):
    class_masks = mask_data['class_masks']
    class_ids = mask_data['class_id']
    #Check whether there's a plate in the image
    class_indicies = np.where(class_ids == class_id)[0]
    if len(class_indicies) == 0:
        print("No plate has been found")
        return None
    
    mask = class_masks[:,: class_indicies[0]]
    
    #Calculate the pixel-to-cm-ratio - using the outline of the plate
    #We use the arc of the plate in order to calculate the ratio (pixel^3 to cm^3)
    plate_outline, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL(), cv2.CHAIN_APPROX_SIMPLE())
    plate_outline = plate_outline[0]
    plate_width_pixels = cv2.arcLength(plate_outline, True)
    pixel_to_cm_ratio = plate_width_pixels/plate_width_inches


