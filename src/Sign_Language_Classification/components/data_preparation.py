import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision.datasets import DatasetFolder
from torchvision import transforms
from ..config.configuration import Configuration

class VideoFramesDataset(DatasetFolder):
    def __init__(self, root, labels, transform=None):
        self.labels = labels
        super().__init__(root, self.load_image, extensions=("jpg",), transform=transform)
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        frame = self.loader(path)
        if self.transform is not None:
            frame = self.transform(frame)
        return frame, target
    
    @staticmethod
    def load_image(path):
        frame = cv2.imread(path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

class DataPreparation:
    def __init__(self, config: Configuration):
        self.config = config
        
    def get_transforms(self):
        """Define the image transformations for model input"""
        resize_dims = self.config.image_transform.resize_dimensions
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize_dims),
            transforms.ToTensor(),
        ])
    
    def create_dataset(self):
        """Create and return the dataset for training"""
        frames_dir = self.config.paths.output_dir
        labels = self.config.dataset.labels
        transform = self.get_transforms()
        
        dataset = VideoFramesDataset(
            root=frames_dir,
            labels=labels,
            transform=transform
        )
        
        return dataset
    
    def create_data_collator(self):
        """Create a collator function for batching data"""
        def collate_fn(batch):
            pixel_values = torch.stack([example[0] for example in batch])
            labels = torch.tensor([example[1] for example in batch])
            return {
                'pixel_values': pixel_values,
                'labels': labels
            }
        return collate_fn
