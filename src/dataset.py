import os
import numpy as np
import cv2
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    """
    Semantic Segmentation Dataset Class.
    Reads images and masks from specified directories, applies augmentations,
    and prepares them for PyTorch models.

    Args:
        image_dir (str): Path to the directory containing images.
        label_dir (str): Path to the directory containing label masks.
        label_suffix (str, optional): Suffix for label files relative to image files.
                                      e.g., if image is 'img1.png' and label is 'img1_mask.png', suffix is '_mask'.
                                      If names are identical, use ''.
        transform (A.Compose, optional): Albumentations transform pipeline. Defaults to None.
    """
    def __init__(self, image_dir, label_dir, label_suffix, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_suffix = label_suffix
        self.transform = transform
        self.image_filenames = sorted([
            f for f in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, f)) and not f.startswith('.') # Ignore hidden files
        ])
        
        # Filter out images that don't have a corresponding label
        self.valid_filenames = []
        for img_fname in self.image_filenames:
            base_name, img_ext = os.path.splitext(img_fname)
            label_fname = f"{base_name}{self.label_suffix}{img_ext}"
            label_path = os.path.join(self.label_dir, label_fname)
            if os.path.exists(label_path):
                self.valid_filenames.append(img_fname)
            else:
                # Try common mask extensions if default fails
                found = False
                for mask_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                    label_fname_alt = f"{base_name}{self.label_suffix}{mask_ext}"
                    label_path_alt = os.path.join(self.label_dir, label_fname_alt)
                    if os.path.exists(label_path_alt):
                        self.valid_filenames.append(img_fname)
                        found = True
                        break
                if not found:
                    print(f"Warning: Label not found for image {img_fname}. Expected pattern: {label_fname} or similar. Skipping.")
                    
        print(f"Found {len(self.valid_filenames)} image-label pairs in {image_dir} and {label_dir}")
        
    def __len__(self):
        return len(self.valid_filenames)
    
    def __getitem__(self, idx):
        img_filename = self.valid_filenames[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        
        base_name, img_ext = os.path.splitext(img_filename)
        label_fname_base = f"{base_name}{self.label_suffix}"
        label_path = None
        
        # Find the correct label file path (handling different extensions)
        for mask_ext in [img_ext, '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            potential_label_path = os.path.join(self.label_dir, f"{label_fname_base}{mask_ext}")
            if os.path.exists(potential_label_path):
                label_path = potential_label_path
                break
        
        if label_path is None:
            raise FileNotFoundError(f"Label for image {img_filename} not found during __getitem__.")
        
        # Read image and mask
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB
        
        mask = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise IOError(f"Failed to load mask: {label_path}")
        # Handle masks that might have multiple channels (e.g., RGBA or 3-channel grayscale)
        if mask.ndim == 3:
            print(f"Warning: Mask {label_path} has {mask.shape[-1]} channels. Using the first channel.")
            mask = mask[..., 0]
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        mask = mask.long()
        
        return image, mask, img_filename