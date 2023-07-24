#My plan:
    #1.  - Wait for 2 pictures - plate width
    #1.2 - Identify plate and width
    #2.  - The moment you get those pictures - use mask rcnn to locate and recieve masks
    #3.  - Use the masks to create the volume
    #4   -Call API with name & weight
import cv2 
import sys
import math
import json
sys.path.append('C:\Programming PYTHON\MarkRcnn37Dig\Mask-RCNN-TF2\kangaroo-transfer-learning')
from kangaroo_prediction import identify_volume_maskrcnn
from recognize_plate import voxel_plate
from nutri import insert_grams_and_food

#Process 2 images and return the nutritions on the plate. 
#The processing: 1) Pixel to cm size of plate 
#2) Voxel Size compared to the item 
#3)Calculate Weight
def proccess_images_data(image_top,image_side):
    foodID = {1:"salad", 2:"rice",3:"chicken breast"}
    density = {"rice":0.6,"chicken breast":1.1,"salad":0.6}
        
    print("Loading...")

#Check for ratios of plates and items(Pixel wise)
    pixel_to_cm_size_top = voxel_plate(image_top)
    pixel_to_cm_size_side= voxel_plate(image_side)
    voxel_size = max(math.ceil(pixel_to_cm_size_top), math.ceil(pixel_to_cm_size_side))
    voxel_size = 1/voxel_size
    volume3d = identify_volume_maskrcnn(image_top, image_side,(voxel_size-0.050,voxel_size-0.050,voxel_size-0.050)) #voxel_size - 0.033)
    print("Volume of the voxels:" ,volume3d)
    
    #Small Refactor for readabillity
    density_of_chicken = density["chicken breast"]
    density_of_rice = density["rice"]
    density_of_salad =density["salad"]
    #For each 3d object do:
    weight = volume3d * density_of_chicken
    weight = math.floor(weight)
    weight = str(weight) 
    json_data = []
    food_items = ["salad","chicken breast" , "salad"]
    keys = {'name','calories','serving_size_g','protein_g','sugar','cholesterol_mg'}
    food_data = insert_grams_and_food(weight, foodID[3])
    json_data.append(food_data)

    json_str = json.dumps(json_data)
    print(json_str)
    return json_str


