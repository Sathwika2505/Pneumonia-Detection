import os
import numpy as np

def data_extraction():
    file_path = os.path.join(os.getcwd(), "output_images")
    
    for img in os.listdir(file_path):
        img_path = os.path.join(file_path, img)
    print("========",file_path)  
    return file_path
    
data_extraction()