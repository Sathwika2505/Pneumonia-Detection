import boto3
import os
import zipfile
from io import BytesIO

# Initialize S3 client
s3 = boto3.client('s3')

def data_extraction(bucket_name, zip_key, target_folder):
    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)
    
    # Download the zip file from S3 into memory
    zip_buffer = BytesIO()
    s3.download_fileobj(bucket_name, zip_key, zip_buffer)
    zip_buffer.seek(0)  # Move the pointer to the start of the file
    
    extracted_files = []
    
    # Open the zip file
    with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
        # Extract all files to the target folder
        zip_ref.extractall(target_folder)
        extracted_files = [os.path.join(target_folder, name) for name in zip_ref.namelist()]

    print(f"Extraction complete. Files saved to {target_folder}")
    return extracted_files

# Usage example
bucket_name = 'deeplearning-mlops-demo'
image_zip_key = 'valid_images.zip'  # Path to the images zip file
xml_zip_key = 'valid_xml.zip'  # Path to the XML files zip file
image_target_folder = './'  # Local folder to save extracted images
xml_target_folder = './'  # Local folder to save extracted XMLs

# Extract images and get file paths
extracted_image_files = data_extraction(bucket_name, image_zip_key, image_target_folder)

# Extract XML files and get file paths
extracted_xml_files = data_extraction(bucket_name, xml_zip_key, xml_target_folder)

# Print the lists of extracted files
print("Extracted Image Files:", extracted_image_files)
print("Extracted XML Files:", extracted_xml_files)