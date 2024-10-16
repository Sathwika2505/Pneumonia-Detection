import torch
from torch.utils.data import Dataset
import os
import glob
import cv2
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import ToPILImage
import dill as pickle
from xml.etree import ElementTree as ET

def transform_data():
    def get_train_transform():
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            T.ToTensor(),
        ])

    def get_valid_transform():
        return T.Compose([
            T.ToTensor(),
        ])

    def is_valid_xml(file_path):
        try:
            tree = ET.parse(file_path)
            return True
        except ET.ParseError as e:
            print(f"XML ParseError for {file_path}: {e}")
            return False

    class CustomDataset(Dataset):
        def __init__(self, images_path, labels_path, directory, width, height, classes, transforms=None):
            self.transforms = transforms
            self.images_path = images_path
            self.labels_path = labels_path
            self.directory = directory
            self.height = height
            self.width = width
            self.classes = classes
            self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
            self.all_image_paths = []

            for file_type in self.image_file_types:
                self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
            self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
            self.all_images = sorted(self.all_images)
            print("Number of images: ", len(self.all_images))

        def load_img(self, img_path):
            print("img_path-------------------", img_path)
            image = cv2.imread(img_path)

            if image is None:
                raise ValueError(f"Image not found or unable to load: {img_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return image

        def normalize_bbox(self, bboxes, rows, cols):
            norm_bboxes = np.zeros_like(bboxes)
            if cols > 0 and rows > 0:
                norm_bboxes[:, 0] = bboxes[:, 0] / cols
                norm_bboxes[:, 1] = bboxes[:, 1] / rows
                norm_bboxes[:, 2] = bboxes[:, 2] / cols
                norm_bboxes[:, 3] = bboxes[:, 3] / rows
            return norm_bboxes
        
        def __getitem__(self, index):
            print(f"Index: {index}, Length of all_images: {len(self.all_images)}")
            if index >= len(self.all_images):
                raise IndexError("Index out of range.")
            
            image_id = self.all_images[index][:-4]
            annot_file_path = os.path.join(self.labels_path, f"{image_id}.xml")
            image_path = os.path.join(self.images_path, self.all_images[index])
        
            # Validate XML before parsing
            if not is_valid_xml(annot_file_path):
                print(f"Deleting corrupted or empty XML: {annot_file_path}")
                os.remove(annot_file_path)  # Delete XML
                if os.path.exists(image_path):
                    print(f"Deleting associated image: {image_path}")
                    os.remove(image_path)  # Delete associated image
                return None  # Return None after deletion
        
            try:
                # Parse the annotation file
                tree = ET.parse(annot_file_path)
                root = tree.getroot()
        
                # Attempt to load the image
                image = self.load_img(image_path)
        
            except ValueError as e:
                print(f"Error loading image: {e}")
                return None  # Skip this image
        
            boxes = []
            labels = []
            for member in root.findall('object'):
                # Check if the expected elements are present and not None
                target_elem = member.find('Target')
                x_elem = member.find('x')
                y_elem = member.find('y')
                width_elem = member.find('width')
                height_elem = member.find('height')
        
                if target_elem is None or x_elem is None or y_elem is None or width_elem is None or height_elem is None:
                    print(f"Missing values in XML for image_id: {image_id}. Deleting entry and associated files.")
                    os.remove(annot_file_path)  # Delete XML
                    if os.path.exists(image_path):
                        os.remove(image_path)  # Delete associated image
                    return None  # Return None after deletion
        
                label = target_elem.text
                if label not in self.classes:
                    continue  # Skip labels not in the classes
                
                labels.append(self.classes.index(label))                
                
                # Convert values to float and check for NoneType before using
                try:
                    x_center = float(x_elem.text)
                    y_center = float(y_elem.text)
                    width = float(width_elem.text)
                    height = float(height_elem.text)
                except (ValueError, TypeError) as e:
                    print(f"Value error for image_id: {image_id}, Error: {e}. Deleting entry and associated files.")
                    os.remove(annot_file_path)  # Delete XML
                    if os.path.exists(image_path):
                        os.remove(image_path)  # Delete associated image
                    return None  # Return None after deletion
        
                width += x_center
                height += y_center
        
                if not any(np.isnan([x_center, y_center, width, height])) and width > 0 and height > 0:
                    boxes.append([x_center, y_center, width, height])
        
            boxes = np.array(boxes) if boxes else np.empty((0, 4))  # Use empty array if no valid boxes
            area = boxes[:, 2] * boxes[:, 3] if boxes.size > 0 else torch.tensor([0], dtype=torch.float32)
            area = torch.as_tensor(area, dtype=torch.float32)
        
            labels = torch.tensor(labels, dtype=torch.long) if labels else torch.tensor([], dtype=torch.long)
        
            image = (image * 255).astype(np.uint8)
        
            image_pil = ToPILImage()(image)
        
            if self.transforms:
                image = self.transforms(image_pil)
        
            _, h, w = image.shape
            norm_boxes = self.normalize_bbox(boxes, rows=h, cols=w)
        
            valid_indices = (norm_boxes[:, 2] > 0) & (norm_boxes[:, 3] > 0)
            valid_boxes = norm_boxes[valid_indices]
        
            target = {}
            if valid_boxes.size > 0:
                target['boxes'] = torch.as_tensor(valid_boxes, dtype=torch.float32)
                target['labels'] = labels[valid_indices]
            else:
                target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                target['labels'] = torch.empty((0,), dtype=torch.long)
        
            target['image_id'] = torch.tensor([index])
            target['area'] = area[valid_indices] if valid_indices.any() else torch.tensor([0], dtype=torch.float32)
        
            return image, target


        def __len__(self):
            return len(self.all_images)

    IMAGE_WIDTH = 800
    IMAGE_HEIGHT = 680
    classes = ['0', '1']

    train_dataset = CustomDataset(
        os.path.join(os.getcwd(), "valid_images"),
        os.path.join(os.getcwd(), "valid_xml"),
        os.getcwd(),
        IMAGE_WIDTH, IMAGE_HEIGHT,
        classes,
        get_train_transform()
    )
    print("Train Dataset: ", train_dataset)

    valid_dataset = CustomDataset(
        os.path.join(os.getcwd(), "valid_images"),
        os.path.join(os.getcwd(), "valid_xml"),
        os.getcwd(),
        IMAGE_WIDTH, IMAGE_HEIGHT,
        classes,
        get_valid_transform()
    )
    print("Valid Dataset: ", valid_dataset)
    
    for j in range(len(train_dataset)):
        result = train_dataset[j]
        if result is not None:  # Check if the result is not None
            i, a = result  # Unpack only if result is valid
            print("Image: ", i)  # Optionally print the image if needed
            print("Annotations: ", a)
        else:
            print(f"Skipping index {j} due to None result.")

    # Save datasets
    with open('train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)

    with open('valid_dataset.pkl', 'wb') as f:
        pickle.dump(valid_dataset, f)

    return train_dataset

transform_data()