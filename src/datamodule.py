import os
import numpy as np
import albumentations as A
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from albumentations.pytorch import ToTensorV2

from .dataset import SegmentationDataset

class SegmentationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Semantic Segmentation.
    Handles dataset splitting, creation of Datasets, and DataLoaders.

    Args:
        config (dict): Configuration dictionary containing data paths, batch size, etc.
        label_suffix (str, optional): Suffix for label files. Defaults to ''.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.label_suffix = config['data']['label_suffix']
        self.image_dir = config['data']['image_dir']
        self.label_dir = config['data']['label_dir']
        self.train_split_ratio = config['data'].get('train_split', 0.8)
        self.batch_size = config['dataloader']['batch_size']
        self.num_workers = config['dataloader']['num_workers']
        self.img_height = config['augmentation']['height']
        self.img_width = config['augmentation']['width']
        
        # Pre-split directories (optional)
        self.train_image_dir = config['data'].get('train_image_dir')
        self.train_label_dir = config['data'].get('train_label_dir')
        self.val_image_dir = config['data'].get('val_image_dir')
        self.val_label_dir = config['data'].get('val_label_dir')
        self.test_image_dir = config['data'].get('test_image_dir')
        self.test_label_dir = config['data'].get('test_label_dir')
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Define augmentations
        self.train_transform = self._get_transforms(train=True)
        self.val_test_transform = self._get_transforms(train=False)
    
    def setup(self, stage=None):
        # Assign train/val/test datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            if self.train_image_dir and self.train_label_dir and self.val_image_dir and self.val_label_dir:
                # Use pre-split directories
                self.train_dataset = SegmentationDataset(self.train_image_dir, self.train_label_dir, transform=self.train_transform, label_suffix=self.label_suffix)
                self.val_dataset = SegmentationDataset(self.val_image_dir, self.val_label_dir, transform=self.val_test_transform, label_suffix=self.label_suffix)
                print(f"Using pre-split data: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}")
            else:
                # Split from a single source directory
                full_dataset = SegmentationDataset(self.image_dir, self.label_dir, transform=None, label_suffix=self.label_suffix)
                total_len = len(full_dataset)
                train_len = int(total_len * self.train_split_ratio)
                val_len = total_len - train_len
                
                if train_len == 0 or val_len == 0:
                    raise ValueError(f"Train/Val split resulted in zero samples for one set (Train: {train_len}, Val: {val_len}). Check dataset size and split ratio.")
                
                # Split indices and create Subsets with transforms applied
                indices = list(range(total_len))
                train_indices, val_indices = train_test_split(indices, train_size=train_len, test_size=val_len, random_state=0)
                
                # Create Subset datasets with the appropriate transforms
                base_train_dataset = SegmentationDataset(self.image_dir, self.label_dir, transform=self.train_transform, label_suffix=self.label_suffix)
                base_val_dataset = SegmentationDataset(self.image_dir, self.label_dir, transform=self.val_test_transform, label_suffix=self.label_suffix)
                
                self.train_dataset = Subset(base_train_dataset, train_indices)
                self.val_dataset = Subset(base_val_dataset, val_indices)
                print(f"Splitting data: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}")
                
        if stage == 'test' or stage is None:
            if self.test_image_dir and self.test_label_dir:
                self.test_dataset = SegmentationDataset(self.test_image_dir, self.test_label_dir, transform=self.val_test_transform, label_suffix=self.label_suffix)
                print(f"Found Test data: {len(self.test_dataset)}")
            elif stage == 'test': 
                # Only raise error if explicitly testing and no test set defined
                print("Warning: Test data directory not specified in config. No test dataloader will be created.")
                self.test_dataset = None
    
    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        if self.test_dataset is None:
            print("Test dataset not available, returning None for test_dataloader.")
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    # Optional: Predict dataloader if you need to run inference on data without labels
    # def predict_dataloader(self):
    #     return DataLoader(
    #         self.predict_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         pin_memory=True,
    #         persistent_workers=True if self.num_workers > 0 else False
    #     )
    
    def _get_transforms(self, train=False):
        transforms = [
            A.Resize(self.img_height, self.img_width)
        ]
        if train:
            # Add training-specific augmentations from config if desired
            if self.config['augmentation'].get('horizontal_flip_prob', 0.0) > 0:
                transforms.append(A.HorizontalFlip(p=self.config['augmentation']['horizontal_flip_prob']))
            if self.config['augmentation'].get('vertical_flip_prob', 0.0) > 0:
                transforms.append(A.VerticalFlip(p=self.config['augmentation']['vertical_flip_prob']))
            if self.config['augmentation'].get('random_brightness_contrast_prob', 0.0) > 0:
                transforms.append(A.RandomBrightnessContrast(p=self.config['augmentation']['random_brightness_contrast_prob']))
            # Add more augmentations here based on config
            pass
        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        return A.Compose(transforms)