from PIL import Image
import os, random
import pandas as pd

def datavisualization():
    images = "./valid_images"
    #print("Images:", images)
    jpg_files = [f for f in os.listdir(images) if f.endswith(".jpg")]
    random_files = random.sample(jpg_files, min(6, len(jpg_files)))
    print("============", random_files)
    
    for file_name in random_files:
        img_path = os.path.join(images, file_name)
        img = Image.open(img_path)
        img_name = os.path.splitext(file_name)[0]
        output_image_path = os.path.join(os.getcwd(), f"{img_name}.jpg")
        img.save(output_image_path)

datavisualization()
