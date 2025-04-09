import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics import F1Score, JaccardIndex

from . import utils

class SegmentationLightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module for Semantic Segmentation.
    Combines model, loss, optimizer, training, validation, and testing logic.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.learning_rate = config['model']['learning_rate']
        self.num_classes = len(config['classes'])
        self.class_names = [c['name'] for c in config['classes']]
        self.class_colors = [c['color'] for c in config['classes']] # For visualization
        self.predict_dir = config['predict']['output_dir'] # For saving test results
        
        # Determine mode based on number of classes
        if self.num_classes < 3:
            # Binary segmentation: model outputs 1 channel, use sigmoid activation later
            self.mode = 'binary'
            # Use BCEWithLogitsLoss or DiceLoss/JaccardLoss with mode='binary'
            self.loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
            
            self.train_f1 = F1Score(task='binary', num_classes=2, threshold=0.5)
            self.val_f1 = F1Score(task='binary', num_classes=2, threshold=0.5)
            self.test_f1 = F1Score(task='binary', num_classes=2, threshold=0.5)
            self.train_iou = JaccardIndex(task='binary', num_classes=2, threshold=0.5)
            self.val_iou = JaccardIndex(task='binary', num_classes=2, threshold=0.5)
            self.test_iou = JaccardIndex(task='binary', num_classes=2, threshold=0.5)
        
        else:
            # Multiclass segmentation: model outputs num_classes channels
            self.mode = 'multiclass'
            # Use CrossEntropyLoss or DiceLoss/JaccardLoss/FocalLoss with mode='multiclass'
            self.loss_fn = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
            
            ignore_index = self.config['classes'][0].get('ignore_in_metrics', False) # Configurable ignore index
            ignore_index_val = 0 if ignore_index else None # Assuming background is class 0 if ignored
            
            # Metrics need num_classes for multiclass
            self.train_f1 = F1Score(task='multiclass', num_classes=self.num_classes, ignore_index=ignore_index_val, average='macro')
            self.val_f1 = F1Score(task='multiclass', num_classes=self.num_classes, ignore_index=ignore_index_val, average='macro')
            self.test_f1 = F1Score(task='multiclass', num_classes=self.num_classes, ignore_index=ignore_index_val, average='macro')
            self.train_iou = JaccardIndex(task='multiclass', num_classes=self.num_classes, ignore_index=ignore_index_val)
            self.val_iou = JaccardIndex(task='multiclass', num_classes=self.num_classes, ignore_index=ignore_index_val)
            self.test_iou = JaccardIndex(task='multiclass', num_classes=self.num_classes, ignore_index=ignore_index_val)
        
        # Create the segmentation model using smp
        self.model = smp.create_model(
            arch=config['model']['architecture'],
            encoder_name=config['model']['encoder_name'],
            encoder_weights=config['model']['encoder_weights'],
            in_channels=3, # Assuming RGB images
            classes=self.num_classes
        )
        
        # To save hyperparameters with the checkpoint
        self.save_hyperparameters(config)
        
        # For storing test outputs
        self.test_step_outputs = []
    
    def forward(self, x):
        return self.model(x)
    
    def _common_step(self, batch, batch_idx, stage):
        images, masks, filenames = batch # Assumes dataloader returns filename
        
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        
        # Log loss
        self.log(f'{stage}_loss', loss, on_step=(stage == 'train'), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # Calculate and log metrics
        # Need predictions (argmax or sigmoid > 0.5) for metrics
        if self.mode == 'binary':
            preds = (torch.sigmoid(logits) > 0.5).int()
            metric_masks = (masks > 0).int()
        else: # Multiclass
            preds = torch.argmax(logits, dim=1)
            metric_masks = masks
        
        # Get the correct metric object based on stage
        f1_metric = getattr(self, f'{stage}_f1', None)
        iou_metric = getattr(self, f'{stage}_iou', None)
        
        if f1_metric:
            f1 = f1_metric(preds, metric_masks)
            self.log(f'{stage}_f1', f1, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        if iou_metric:
            iou = iou_metric(preds, metric_masks)
            self.log(f'{stage}_iou', iou, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        return {'loss': loss, 'preds': preds, 'masks': masks, 'filenames': filenames}
    
    def training_step(self, batch, batch_idx):
        result = self._common_step(batch, batch_idx, 'train')
        return result['loss']
    
    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        result = self._common_step(batch, batch_idx, 'test')
        # Store results for processing in test_epoch_end
        self.test_step_outputs.append({
            'preds': result['preds'].cpu().numpy(),
            'masks': result['masks'].cpu().numpy(),
            'filenames': result['filenames']
        })
    
    def on_test_epoch_end(self):
        # Aggregate metrics
        print(f"Final Test F1: {self.test_f1.compute():.4f}")
        print(f"Final Test IoU: {self.test_iou.compute():.4f}")
        self.test_f1.reset()
        self.test_iou.reset()
        
        # --- Save predicted masks and visualizations ---
        os.makedirs(self.predict_dir, exist_ok=True)
        os.makedirs(os.path.join(self.predict_dir, 'masks'), exist_ok=True)
        if self.config['predict']['save_combined']:
            os.makedirs(os.path.join(self.predict_dir, 'combined'), exist_ok=True)
        
        print(f"Saving test predictions to {self.predict_dir}...")
        
        # Save predicted masks and potentially combined images if labels are available
        for output_batch in self.test_step_outputs:
            preds_np = output_batch['preds']
            masks_np = output_batch['masks'] # Ground truth masks
            filenames = output_batch['filenames']
            
            for i in range(len(filenames)):
                filename = filenames[i]
                pred_mask = preds_np[i].astype(np.uint8) # Predicted mask (class indices)
                gt_mask = masks_np[i].astype(np.uint8) # Ground truth mask
                
                # --- Save raw prediction mask ---
                pred_mask_path = os.path.join(self.predict_dir, 'masks', filename)
                base, _ = os.path.splitext(pred_mask_path)
                pred_mask_path = base + ".png"
                cv2.imwrite(pred_mask_path, pred_mask)
                
                # --- Save combined visualization (optional) ---
                if self.config['predict']['save_combined']:
                    try:
                        img_path = self._find_original_image_path(filename)
                        if img_path and os.path.exists(img_path):
                            image = cv2.imread(img_path) # BGR
                            # Resize image if necessary to match mask size
                            h, w = pred_mask.shape[:2]
                            image = cv2.resize(image, (w, h))
                            
                            # Colorize masks
                            pred_viz = utils.visualize_mask(pred_mask, self.class_colors)
                            gt_viz = utils.visualize_mask(gt_mask, self.class_colors)
                            
                            # Combine: Image | Ground Truth | Prediction
                            combined_img = utils.combine_visualizations(image, gt_viz, pred_viz)
                            
                            combined_path = os.path.join(self.predict_dir, 'combined', filename)
                            base_comb, _ = os.path.splitext(combined_path)
                            combined_path = base_comb + ".png"
                            cv2.imwrite(combined_path, combined_img)
                        else:
                            print(f"Warning: Could not find original image for {filename} to create combined viz.")
                    except Exception as e:
                        print(f"Error creating combined visualization for {filename}: {e}")
        
        self.test_step_outputs.clear() # Free memory
    
    def _find_original_image_path(self, filename):
        # Try finding the image in the test directory specified in config
        test_img_dir = self.config['data'].get('test_image_dir')
        if test_img_dir:
            path = os.path.join(test_img_dir, filename)
            if os.path.exists(path):
                return path
        # Fallback: try the main image dir (if not using separate test dir)
        main_img_dir = self.config['data'].get('image_dir')
        if main_img_dir:
            path = os.path.join(main_img_dir, filename)
            if os.path.exists(path):
                return path
        return None
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # Optional: Add learning rate scheduler
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        return optimizer