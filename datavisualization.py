import imghdr
from PIL import Image
from data_extraction import data_extraction
import os, random

def datavisualization():
    images = data_extraction()
    print("Images:", images)
    
    # List all files that end with .jpg
    jpg_files = [f for f in os.listdir(images) if f.endswith(".jpg")]
    random_files = random.sample(jpg_files, min(6, len(jpg_files)))
    print("============", random_files)

    for file_name in random_files:
        img_path = os.path.join(images, file_name)
        
        # Check if the file is a valid image
        if imghdr.what(img_path) is None:
            print(f"Skipping invalid image file: {img_path}")
            continue
        
        try:
            img = Image.open(img_path)
            img_name = os.path.splitext(file_name)[0]
            output_image_path = os.path.join(os.getcwd(), f"{img_name}.jpg")
            img.save(output_image_path)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

datavisualization()
