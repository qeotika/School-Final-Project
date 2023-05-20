from PIL import Image
import os

#resize all my images

def resize_images(directory, output_directory, width, height):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            output_path = os.path.join(output_directory, filename)
            try:
                with Image.open(image_path) as image:
                    resized_image = image.resize((width, height))
                    resized_image.save(output_path)
                    print(f"Resized {filename} successfully.")
            except Exception as e:
                print(f"Error resizing {filename}: {str(e)}")

# Example usage:
input_directory = "C:/Program Files/Final Project Sapir/School-Final-Project/DatasetP/DatasetPlatesVal"
output_directory = "C:/Program Files/Final Project Sapir/School-Final-Project/DatasetP/val"
width = 256
height = 197

resize_images(input_directory, output_directory, width, height)