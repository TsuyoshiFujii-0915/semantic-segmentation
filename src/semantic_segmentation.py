import os
import torch
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# Import modules from src package
from .utils import load_config
from .datamodule import SegmentationDataModule
from .lightning_module import SegmentationLightningModule

def main(config_path='config/config.yaml'):
    """
    Main training function.
    Loads config, initializes datamodule and model, sets up trainer, and starts training.
    """
    # Load configuration
    config = load_config(config_path)
    print("Configuration loaded:")
    print(config)
    
    # Create unique experiment name/version for logging and checkpoints
    experiment_name = f"{config['model']['architecture']}_{config['model']['encoder_name']}"
    log_dir = 'logs'
    
    # Initialize DataModule
    data_module = SegmentationDataModule(config)
    
    # Initialize LightningModule
    lightning_module = SegmentationLightningModule(config)
    
    # Configure callbacks
    # --- Model Checkpointing ---
    # Saves the best model based on validation IoU score.
    checkpoint_dir = os.path.join(log_dir, experiment_name, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch}-{val_f1:.4f}-{val_iou:.4f}',
        save_top_k=1,
        monitor='val_iou',
        mode='max',
        save_last=True,
        verbose=True
    )
    
    # --- Learning Rate Monitoring ---
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # --- Early Stopping ---
    early_stopping_callback = EarlyStopping(
        monitor='val_iou',
        patience=10,
        mode='max',
        verbose=True
    )
    
    callbacks = [checkpoint_callback, lr_monitor, early_stopping_callback]
    
    # Configure logger
    csv_logger = CSVLogger(save_dir=log_dir, name=experiment_name)
    
    # Initialize Trainer
    trainer = Trainer(
        **config['training'],
        callbacks=callbacks,
        logger=csv_logger
    )
    
    # Start training
    print("Starting training...")
    trainer.fit(model=lightning_module, datamodule=data_module)
    print("Training finished.")
    
    # Save the state_dict of the best model found during training
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Found best model checkpoint at: {best_model_path}")
        # Load the best model checkpoint
        best_lightning_module = SegmentationLightningModule.load_from_checkpoint(
            best_model_path,
            config=config
        )
        # Define path to save the state_dict
        best_model_state_dict_path = os.path.join(config['model']['save_dir'], f"{experiment_name}_best_model.pth")
        # Save the state_dict
        os.makedirs(config['model']['save_dir'], exist_ok=True)
        torch.save(best_lightning_module.model.state_dict(), best_model_state_dict_path)
        print(f"Best model state_dict saved to {best_model_state_dict_path}")
    
    # Optional: Run Testing after training using the best model
    data_module.setup('test') # Ensure test dataset is loaded
    if data_module.test_dataloader() is not None:
        print(f"Starting testing using the saved best model state_dict: {best_model_state_dict_path}")
        
        # Create a new instance of the LightningModule for testing
        test_model = SegmentationLightningModule(config)
        
        # Load the saved state_dict into the model attribute
        try:
            state_dict = torch.load(best_model_state_dict_path)
            test_model.model.load_state_dict(state_dict)
            print("Successfully loaded state_dict into the model for testing.")
            
            # Run testing with the loaded model
            test_results = trainer.test(model=test_model,datamodule=data_module)
            print("Test results:", test_results)
            
            # Save test scores
            if test_results and isinstance(test_results, list):
                scores = test_results[0] # Assuming single test dataloader
                scores_path = os.path.join(log_dir, experiment_name, 'test_scores.txt')
                try:
                    from .utils import save_scores # Import here to avoid circular dependency if utils imports train stuff
                    save_scores(scores, scores_path)
                    print(f"Test scores saved to {scores_path}")
                except ImportError:
                    print("Could not import save_scores from utils to save test scores.")
                except Exception as e:
                    print(f"Error saving test scores: {e}")
        except AttributeError:
            print("Could not find 'model' attribute in the LightningModule. Assuming the saved state_dict is for the entire LightningModule.")
            try:
                # Attempt to load the state_dict into the entire LightningModule
                state_dict = torch.load(best_model_state_dict_path)
                test_model.load_state_dict(state_dict)
                print("Successfully loaded state_dict into the LightningModule for testing.")
                # Run testing
                test_results = trainer.test(model=test_model, datamodule=data_module)
                print("Test results:", test_results)
            except Exception as e:
                print(f"Error loading state_dict into the LightningModule or running test: {e}")
        except Exception as e:
            print(f"Error loading state_dict or running test: {e}")
            
    else:
        print("No test dataset configured. Skipping testing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Semantic Segmentation Model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to the configuration file (default: config/config.yaml)'
    )
    args = parser.parse_args()
    
    main(config_path=args.config)