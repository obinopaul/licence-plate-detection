from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
import os

class licenceplate(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and labels.
            split (string): One of 'train', 'test', or 'valid' to specify the dataset split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.images_dir = os.path.join(self.root_dir, 'images')
        self.labels_dir = os.path.join(self.root_dir, 'labels')
        self.image_files = [f for f in os.listdir(self.images_dir) if os.path.isfile(os.path.join(self.images_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label_name = os.path.splitext(self.image_files[idx])[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)
        boxes = self.parse_labels(label_path)

        sample = {'image': image, 'boxes': boxes}

        return sample

    def parse_labels(self, label_path):
        """
        Parse the label file and return the target format expected by the model.
        The label file contains lines in the format: class x_center y_center width height.
        """
        boxes = []
        with open(label_path, 'r') as file:
            for line in file:
                class_label, x_center, y_center, width, height = map(float, line.split())
                boxes.append([class_label, x_center, y_center, width, height])
        return torch.tensor(boxes)

        