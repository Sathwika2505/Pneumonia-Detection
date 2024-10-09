import torch
from torch.utils.data import Dataset
import os
from xml.etree import ElementTree as ET
import glob as glob
import torch
import cv2
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as transforms
import dill as pickle

def transform_data():
    # os.system("python3.10 annot2xml.py")
    # Define the training tranforms
    def get_train_aug():
        return A.Compose([
            A.MotionBlur(blur_limit=3, p=0.5),
            A.Blur(blur_limit=3, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, p=0.5
            ),
            A.ColorJitter(p=0.5),
            # A.Rotate(limit=10, p=0.2),
            A.RandomGamma(p=0.2),
            A.RandomFog(p=0.2),
            # A.RandomSunFlare(p=0.1),
            # `RandomScale` for multi-res training,
            # `scale_factor` should not be too high, else may result in 
            # negative convolutional dimensions.
            # A.RandomScale(scale_limit=0.15, p=0.1),
            # A.Normalize(
            #     (0.485, 0.456, 0.406),
            #     (0.229, 0.224, 0.225)
            # ),
            ToTensorV2(p=1.0),
        ], bbox_params={
            'format': 'pascal_voc',
            'label_fields': ['labels']
        })

    def get_train_transform():
        return A.Compose([
            # A.Normalize(
            #     (0.485, 0.456, 0.406),
            #     (0.229, 0.224, 0.225)
            # ),
            ToTensorV2(p=1.0),
        ], bbox_params={
            'format': 'pascal_voc',
            'label_fields': ['labels']
        })

    # Define the validation transforms
    def get_valid_transform():
        return A.Compose([
            # A.Normalize(
            #     (0.485, 0.456, 0.406),
            #     (0.229, 0.224, 0.225)
            # ),
            ToTensorV2(p=1.0),
        ], bbox_params={
            'format': 'pascal_voc', 
            'label_fields': ['labels']
        })


    class CustomDataset(Dataset):
        def __init__(
            self, images_path, labels_path, labels_txt, directory,
            width, height, classes, transforms=None, 
            use_train_aug=False,
            train=False, mosaic=False
        ):
            self.transforms = transforms
            self.use_train_aug = use_train_aug
            self.images_path = images_path
            self.labels_path = labels_path
            self.directory = directory
            self.labels_txt = labels_txt
            self.height = height
            self.width = width
            self.classes = classes
            self.train = train
            self.mosaic = mosaic
            self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
            self.all_image_paths = []
            
            # get all the image paths in sorted order
            for file_type in self.image_file_types:
                self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
            self.all_annot_paths = glob.glob(os.path.join(self.labels_path, '*.xml'))
            self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
            self.all_images = sorted(self.all_images)
            print("Number of images:-----------------", len(self.all_images))
            # Remove all annotations and images when no object is present.

        def load_image_and_labels(self, index):
            if index >= len(self.all_images):
                raise IndexError("Index out of range")
            image_name = self.all_images[index]
            image_path = os.path.join(self.images_path, image_name)

            # Read the image.
            image = cv2.imread(image_path)
            # Convert BGR to RGB color format.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image_resized = cv2.resize(image, (self.width, self.height))
            image_resized /= 255.0

            # Capture the corresponding XML file for getting the annotations.
            annot_filename = image_name[:-4] + '.xml'
            annot_file_path = os.path.join(self.labels_path, annot_filename)

            boxes = []
            orig_boxes = []
            labels = []
            tree = ET.parse(annot_file_path)
            root = tree.getroot()

            # Get the height and width of the image.
            image_width = image.shape[1]
            image_height = image.shape[0]

            # Box coordinates for xml files are extracted and corrected for image size given.
            for member in root.findall('object'):
                labels.append(self.classes.index(member.find('Target').text))
                x_center = float(member.find('x').text)
                y_center = float(member.find('y').text)
                width = float(member.find('width').text)
                height = float(member.find('height').text)
                xmin = (x_center - width / 2) * image_width
                ymin = (y_center - height / 2) * image_height
                xmax = (x_center + width / 2) * image_width
                ymax = (y_center + height / 2) * image_height

                # Ensure xmax and ymax are not greater than the image dimensions
                ymax, xmax = self.check_image_and_annotation(xmax, ymax, image_width, image_height)
                orig_boxes.append([xmin, ymin, xmax, ymax])

                xmin_final = (xmin/image_width) * self.width
                xmax_final = (xmax/image_width) * self.width
                ymin_final = (ymin/image_height) * self.height
                ymax_final = (ymax/image_height) * self.height

                # Skip bounding boxes containing NaN values
                if not np.isnan([xmin_final, ymin_final, xmax_final, ymax_final]).any():
                    boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

            # Bounding box to tensor.
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            # Check for empty boxes
            if len(boxes) == 0:
                # Handle the case where no valid boxes are available.
                return image, image_resized, [], [], [], None, None, (image_width, image_height)

            # Area of the bounding boxes.
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            # No crowd instances.
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

            # Labels to tensor.
            labels = torch.as_tensor(labels, dtype=torch.int64)

            return image, image_resized, orig_boxes, boxes, labels, area, iscrowd, (image_width, image_height)


        def check_image_and_annotation(self, xmax, ymax, width, height):
            """
            Check that all x_max and y_max are not more than the image
            width or height.
            """
            if ymax > height:
                ymax = height
            if xmax > width:
                xmax = width
            return ymax, xmax


        def load_cutmix_image_and_boxes(self, index, resize_factor=512):
            """ 
            Adapted from: https://www.kaggle.com/shonenkov/oof-evaluation-mixup-efficientdet
            """
            image, _, _, _, _, _, _, _ = self.load_image_and_labels(index=index)
            orig_image = image.copy()
            # Resize the image according to the `confg.py` resize.
            image = cv2.resize(image, resize_factor)
            h, w, c = image.shape
            s = h // 2
        
            xc, yc = [int(random.uniform(h * 0.25, w * 0.75)) for _ in range(2)]  # center x, y
            indexes = [index] + [random.randint(0, len(self.all_images) - 1) for _ in range(3)]
            
            # Create empty image with the above resized image.
            result_image = np.full((h, w, 3), 1, dtype=np.float32)
            result_boxes = []
            result_classes = []

            for i, index in enumerate(indexes):
                image, image_resized, orig_boxes, boxes, \
                labels, area, iscrowd, dims = self.load_image_and_labels(
                index=index
                )
                # Resize the current image according to the above resize,
                # else `result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]`
                # will give error when image sizes are different.
                image = cv2.resize(image, resize_factor)
                if i == 0:
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
                padw = x1a - x1b
                padh = y1a - y1b

                boxes[:, 0] += padw
                boxes[:, 1] += padh
                boxes[:, 2] += padw
                boxes[:, 3] += padh
                
                result_boxes.append(boxes)
                for class_name in labels:
                    result_classes.append(class_name)

            final_classes = []
            result_boxes = np.concatenate(result_boxes, 0)
            np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
            result_boxes = result_boxes.astype(np.int32)
            for idx in range(len(result_boxes)):
                if ((result_boxes[idx,2]-result_boxes[idx,0])*(result_boxes[idx,3]-result_boxes[idx,1])) > 0:
                    final_classes.append(result_classes[idx])
            result_boxes = result_boxes[
                np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)
            ]
            return orig_image, result_image/255., torch.tensor(result_boxes), \
                torch.tensor(np.array(final_classes)), area, iscrowd, dims

        def __getitem__(self, idx):
            print("Index-----------------------", idx)
            # Capture the image name and the full image path.
            if not self.mosaic:
                image, image_resized, orig_boxes, boxes, \
                    labels, area, iscrowd, dims = self.load_image_and_labels(
                    index=idx
                )

            if self.train and self.mosaic:
                while True:
                    image, image_resized, boxes, labels, \
                        area, iscrowd, dims = self.load_cutmix_image_and_boxes(
                        idx, resize_factor=(self.height, self.width)
                    )
                    if len(boxes) > 0:
                        break
            
            # visualize_mosaic_images(boxes, labels, image_resized, self.classes)

            # Prepare the final `target` dictionary.
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["area"] = area
            target["iscrowd"] = iscrowd
            image_id = torch.tensor([idx])
            target["image_id"] = image_id
            if self.use_train_aug: # Use train augmentation if argument is passed.
                train_aug = get_train_aug()
                sample = train_aug(image=image_resized,
                                        bboxes=target['boxes'],
                                        labels=labels)
                image_resized = sample['image']
                target['boxes'] = torch.Tensor(sample['bboxes'])
            else:
                sample = self.transforms(image=image_resized,
                                        bboxes=target['boxes'],
                                        labels=labels)
                image_resized = sample['image']
                target['boxes'] = torch.Tensor(sample['bboxes'])
            
                
            return image_resized, target
            
        def __len__(self):
            return len(self.all_images)

    IMAGE_WIDTH = 800
    IMAGE_HEIGHT = 680
    classes = ['0', '1']
    # Create datasets
    train_dataset = CustomDataset(os.path.join(os.getcwd(),"output_images"),os.path.join(os.getcwd(),"xml_labels"), os.path.join(os.getcwd(),"txt_labels"), "Pnemonia",IMAGE_WIDTH, IMAGE_HEIGHT, classes, get_train_transform())
    print("one-------------",train_dataset)
    valid_dataset = CustomDataset(os.path.join(os.getcwd(),"output_images"),os.path.join(os.getcwd(),"xml_labels"), os.path.join(os.getcwd(),"txt_labels"), "Pnemonia",IMAGE_WIDTH, IMAGE_HEIGHT, classes, get_valid_transform())
    print("-------------",valid_dataset)
    i, a = train_dataset[20]
    print("iiiiii:",i)
    print("aaaaa:",a)
    with open('train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open('valid_dataset.pkl', 'wb') as f:
        pickle.dump(valid_dataset, f)

    return train_dataset

transform_data()
