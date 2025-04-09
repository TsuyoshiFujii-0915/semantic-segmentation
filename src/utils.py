import os
import numpy as np
import cv2
import yaml

def load_config(config_path='config/config.yaml'):
    """
    Loads the YAML configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
            raise
    return config

def visualize_mask(mask, class_colors):
    """
    Visualizes a segmentation mask using specified colors for each class.

    Args:
        mask (np.ndarray): A 2D numpy array (H, W) with integer class indices.
        class_colors (list): A list of RGB color tuples or lists, e.g., [[R, G, B], ...].
                             The index corresponds to the class index.

    Returns:
        np.ndarray: A 3D numpy array (H, W, 3) representing the colorized mask (BGR).
    """
    h, w = mask.shape
    # Create an empty BGR image
    viz_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx, color_rgb in enumerate(class_colors):
        # Find pixels belonging to the current class
        class_pixels = (mask == class_idx)
        # Convert RGB to BGR for OpenCV
        color_bgr = tuple(reversed(color_rgb))
        viz_mask[class_pixels] = color_bgr
    
    return viz_mask

def combine_visualizations(image, gt_mask_viz, pred_mask_viz, separator_width=10, separator_color=(255, 255, 255)):
    """
    Combines original image, ground truth mask visualization, and predicted mask visualization
    horizontally into a single image with separators.

    Args:
        image (np.ndarray): Original image (H, W, 3) in BGR format.
        gt_mask_viz (np.ndarray): Colorized ground truth mask (H, W, 3) in BGR format.
        pred_mask_viz (np.ndarray): Colorized predicted mask (H, W, 3) in BGR format.
        separator_width (int): Width of the white separator line between images.
        separator_color (tuple): BGR color of the separator line.

    Returns:
        np.ndarray: Combined image (H, W_combined, 3) in BGR format.
    """
    h, w, _ = image.shape
    # Ensure all images have the same height (they should if processed correctly)
    assert h == gt_mask_viz.shape[0] == pred_mask_viz.shape[0], "Heights must match"
    assert w == gt_mask_viz.shape[1] == pred_mask_viz.shape[1], "Widths must match"
    
    # Create separators
    separator = np.full((h, separator_width, 3), separator_color, dtype=np.uint8)
    
    # Concatenate horizontally: Image | Separator | GT Mask | Separator | Pred Mask
    combined_image = cv2.hconcat([image, separator, gt_mask_viz, separator, pred_mask_viz])
    
    return combined_image

def save_scores(scores, output_path):
    """
    Saves evaluation scores (e.g., F1, IoU) to a text file.
    """
    with open(output_path, 'w') as f:
        for key, value in scores.items():
            f.write(f"{key}: {value:.4f}\n")