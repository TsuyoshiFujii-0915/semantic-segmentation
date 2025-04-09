# Semantic Segmentation

This repository provides a simple, configurable pipeline for semantic segmentation tasks using PyTorch Lightning and segmentation-models-pytorch.

## Features

- **Configurable:** Most parameters (data paths, training settings, model architecture, classes) are controlled via a YAML configuration file (`config/config.yaml`).
- **PyTorch Lightning:** Leverages PyTorch Lightning for cleaner code structure, multi-GPU training, mixed-precision, logging, checkpointing, and more.
- **Segmentation Models PyTorch (SMP):** Easily utilizes various segmentation model architectures (U-Net, FPN, DeepLabV3+, etc.) and pre-trained encoders.
- **Clear Structure:** Code is organized into modules for dataset handling, data loading, model/lightning logic, training, prediction, and utilities.
- **Basic Logging:** Logs metrics to CSV.
- **Checkpointing:** Saves the best model based on validation IoU.
- **Visualization:** Generates colorized prediction masks and comparison images (Input | Prediction).

## Directory Structure

```
.
├── config/
│   └── config.yaml                 # Configuration file
├── data/
│   ├── images/                     # Directory for input images
│   └── labels/                     # Directory for corresponding label masks
│   └── test/                       # Optional: Directory for test images/labels
│       ├── images/
│       └── labels/
├── logs/                           # Stores training logs and checkpoints
├── models/                         # Stores best model state dictionaries
├── results/                        # Stores test results including predictions and visualizations
│   ├── combined                    # Combines original image, ground truth mask, and predicted mask
│   └── masks/                      # Predicted masks
├── src/
│   ├── datamodule.py               # PyTorch Lightning DataModule
│   ├── dataset.py                  # PyTorch Dataset class
│   ├── lightning_module.py         # PyTorch Lightning Module (model, train/val/test logic)
│   ├── semantic_segmentation.py    # Main training script
│   └── utils.py                    # Utility functions (config loading, visualization)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/TsuyoshiFujii-0915/semantic-segmentation
    cd semantic-segmentation
    ```

2.  **Create a Python environment (recommended using uv):**

    ```bash
    uv sync
    . .venv/bin/activate
    ```

3.  **Install dependencies (if not using uv sync):**

    ```bash
    uv add -r requirements.txt
    ```
    
    _Note: Ensure you have a compatible version of PyTorch installed for your system (CPU/GPU). See [pytorch.org](https://pytorch.org/) for details._

## Dataset Preparation

1.  Place your training images in the directory specified by `data.image_dir` in `config.yaml` (default: `data/images/`).
2.  Place the corresponding ground truth label masks in the directory specified by `data.label_dir` (default: `data/labels/`).
3.  **Naming Convention:** The code assumes that for each image `imagename.ext` in the `image_dir`, there is a corresponding mask `imagename<label_suffix>.ext` (or `.png`, etc.) in the `label_dir`. The `label_suffix` (e.g., `_mask`) can be set in `data.label_suffix` in `config.yaml` (currently hardcoded as '').
4.  **Mask Format:** Label masks should be grayscale images where each pixel's value represents the class index (e.g., 0 for background, 1 for class A, 2 for class B, ...). These values **must** correspond to the `value` specified for each class in `config.yaml`. **Crucial:** The order here defines the class index (0, 1, 2...). The background is typically index 0.
5.  **Train/Validation Split:**
    - By default, the code splits the data found in `image_dir` and `label_dir` into training and validation sets based on `data.train_split` ratio in `config.yaml`.
    - Alternatively, you can pre-split your data and specify separate directories (`train_image_dir`, `train_label_dir`, `val_image_dir`, `val_label_dir`) in `config.yaml`.
6.  **Test Set (Optional):** If you have a separate test set, place images and labels in directories specified by `data.test_image_dir` and `data.test_label_dir` in `config.yaml`.

## Configuration (`config/config.yaml`)

Modify `config/config.yaml` to adjust parameters:

- **`data`:** Paths to image/label directories, train/validation split ratio.
- **`classes`:** Define your object classes.
  - `name`: Human-readable name.
  - `color`: RGB color for visualization.
  - `value`: The pixel value corresponding to this class in your label masks. **Crucial:** The order here defines the class index (0, 1, 2...). The background is typically index 0.
  - `ignore_in_metrics` (optional, default: false): If set to `true` for the background class (index 0), it will be ignored during F1/IoU calculation.
- **`dataloader`:** Batch size, number of workers
- **`training`:** Epochs, accelerator (`cpu`, `gpu`, `auto`), devices (`auto` or number), precision (`32`, `16`), log steps (how often to log metrics during training).
- **`model`:** Segmentation model architecture (`Unet`, `FPN`, etc.), encoder (`resnet34`, etc.), encoder weights (`imagenet`, `None`), learning rate, save directory. See [segmentation-models-pytorch docs](https://github.com/qubvel/segmentation_models.pytorch) for options.
- **`augmentation`:** Image height/width for resizing, and probabilities for other augmentations (add more in `datamodule.py` as needed).
- **`predict`:** Checkpoint to use for prediction (`best` or path), output directory, whether to save combined visualizations.

## Training

Run the training script from the root directory of the project:

```bash
python -m src.semantic_segmentation --config config/config.yaml
```

- You can specify a different config file using the `--config` argument.
- Logs (CSV) and model checkpoints (`.ckpt` files) will be saved under the `logs/` directory, organized by experiment name (e.g., `logs/Unet_resnet34/`).