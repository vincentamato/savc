import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb
import argparse
from datetime import datetime
import numpy as np
import logging
import sys
from accelerate import Accelerator
import gc
import math
from torch.optim.lr_scheduler import LambdaLR

from src.models.music_transformer import MusicTransformer
from src.data.dataset import get_dataset, MIDITokenizer
from src.losses.music_generation_loss import MusicGenerationLoss


class DynamicLossWeighter:
    def __init__(self, initial_weights, decay_factor=0.95, min_weight=0.1):
        self.weights = initial_weights
        self.decay_factor = decay_factor
        self.min_weight = min_weight
        self.moving_averages = {k: 0.0 for k in initial_weights}
        
    def update(self, loss_dict):
        # Update moving averages
        for k, v in loss_dict.items():
            if k in self.moving_averages:
                self.moving_averages[k] = self.moving_averages[k] * self.decay_factor + v * (1 - self.decay_factor)
        
        # Calculate relative magnitudes
        max_loss = max(self.moving_averages.values())
        if max_loss > 0:
            for k in self.weights:
                relative_magnitude = self.moving_averages[k] / max_loss
                # Increase weight for smaller losses, decrease for larger ones
                self.weights[k] = max(
                    self.min_weight,
                    self.weights[k] * (2.0 - relative_magnitude)
                )
        
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
        
        return self.weights

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

def parse_args():
    parser = argparse.ArgumentParser(description='Train Art-to-Music Generation Model')
    
    # Model configuration
    parser.add_argument('--vit_model', type=str, default="google/vit-base-patch16-384",
                       help='ViT model name from HuggingFace')
    parser.add_argument('--d_model', type=int, default=1024,
                       help='Model dimension')
    parser.add_argument('--nhead', type=int, default=16,
                       help='Number of attention heads')
    parser.add_argument('--num_decoder_layers', type=int, default=16,  # Increased from 8
                       help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=4096,
                       help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.15,  # Slightly increased
                       help='Dropout rate')
    parser.add_argument('--max_seq_length', type=int, default=1024,  # Increased for longer sequences
                       help='Maximum sequence length')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=16,  # Adjusted for stability
                       help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=20,  # Increased
                       help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=150,  # Increased for better convergence
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5,  # Reduced from 1e-4
                   help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,  # Adjusted
                    help='Minimum learning rate')
    parser.add_argument('--warmup_steps', type=int, default=4000,  # Increased from 0
                    help='Warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.02,  # Increased
                       help='Weight decay')
    parser.add_argument('--label_smoothing', type=float, default=0.2,
                       help='Label smoothing factor')
    parser.add_argument('--grad_clip', type=float, default=0.5,
                       help='Gradient clipping')
    
    # Add new loss weighting arguments
    parser.add_argument('--token_loss_weight', type=float, default=1.0,
                       help='Weight for token prediction loss')
    parser.add_argument('--harmony_loss_weight', type=float, default=0.5,
                       help='Weight for harmony loss')
    parser.add_argument('--rhythm_loss_weight', type=float, default=0.5,
                       help='Weight for rhythm loss')
    parser.add_argument('--contour_loss_weight', type=float, default=0.3,
                       help='Weight for melodic contour loss')
    
    # Add dynamic weighting arguments
    parser.add_argument('--use_dynamic_weighting', type=bool, default=True,
                       help='Use dynamic loss weighting')
    parser.add_argument('--weight_decay_factor', type=float, default=0.95,
                       help='Decay factor for loss weights')
    
    # Logging and evaluation
    parser.add_argument('--log_every', type=int, default=10,
                       help='Log every N batches')
    parser.add_argument('--eval_every', type=int, default=500,
                       help='Evaluate every N steps')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save every N epochs')
    
    # System and optimization
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                       help='Mixed precision mode')
    parser.add_argument('--gradient_checkpointing', type=bool, default=True,
                   help='Enable gradient checkpointing')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Save directory')
    parser.add_argument('--project_name', type=str, default='art-to-music',
                       help='WandB project name')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of workers')
    parser.add_argument('--max_duration', type=float, default=30.0,
                       help='Maximum MIDI duration in seconds')
    
    return parser.parse_args()
    
def setup_logging(save_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(save_dir / 'training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    """
    def lr_lambda(current_step: int):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)

def setup_wandb_logging(args):
    """Enhanced WandB configuration with comprehensive metrics"""
    run = wandb.init(
        project=args.project_name,
        config={
            "model_config": {
                "vit_model": args.vit_model,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_decoder_layers": args.num_decoder_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
                "max_seq_length": args.max_seq_length
            },
            "training_config": {
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "epochs": args.epochs,
                "lr": args.lr,
                "min_lr": args.min_lr,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps
            },
            "loss_config": {
                "token_loss_weight": args.token_loss_weight,
                "harmony_loss_weight": args.harmony_loss_weight,
                "rhythm_loss_weight": args.rhythm_loss_weight,
                "contour_loss_weight": args.contour_loss_weight,
                "use_dynamic_weighting": args.use_dynamic_weighting,
                "weight_decay_factor": args.weight_decay_factor
            }
        }
    )
    
    # Define metric groupings
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("val/*", step_metric="val/step")
    wandb.define_metric("epoch", summary="max")
    wandb.define_metric("epoch_*", step_metric="epoch")
    
    return run

def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, accelerator, args, current_epoch, loss_weighter=None, logger=None):
    model.train()
    total_batches = len(train_loader)
    
    all_losses = {
        'token_loss': 0.0,
        'harmony_loss': 0.0,
        'rhythm_loss': 0.0,
        'contour_loss': 0.0,
        'total_loss': 0.0
    }
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {current_epoch+1}/{args.epochs}")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            with accelerator.autocast():
                outputs = model(
                    batch['image'],
                    batch['tokens'],
                    attention_mask=batch['attention_mask']
                )

                # Get loss dictionary from criterion
                loss_dict = criterion(outputs, batch['tokens'], attention_mask=batch['attention_mask'])
                
                # Extract total loss
                loss = loss_dict['total_loss']
                
                # Check for NaNs
                for loss_name, loss_value in loss_dict.items():
                    if torch.isnan(loss_value):
                        logger.error(f"NaN detected in {loss_name} at batch {batch_idx} of epoch {current_epoch+1}")
                        raise ValueError(f"NaN detected in {loss_name}")
                
                # Scale loss for gradient accumulation if needed
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
            
            # Backward pass
            accelerator.backward(loss)
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update metrics
            loss_value = loss.item() * (args.gradient_accumulation_steps if args.gradient_accumulation_steps > 1 else 1)
            for k, v in loss_dict.items():
                all_losses[k] += v.item()
            
            # Log progress on the progress bar
            progress_bar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Detailed logging at every step
            if accelerator.is_main_process:
                global_step = batch_idx + total_batches * current_epoch
                log_dict = {
                    'train/step': global_step,
                    'train/loss': loss_value,
                    'train/learning_rate': scheduler.get_last_lr()[0]
                }
                
                # Log individual losses
                for k in loss_dict:
                    log_dict[f'train/losses/{k}'] = loss_dict[k].item()
                
                wandb.log(log_dict, step=global_step)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                accelerator.print('OOM occurred, clearing cache and skipping batch')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
        except ValueError as ve:
            accelerator.print(f"Training stopped due to NaN loss: {ve}")
            return None  # Return None to indicate training should stop
    
    # Calculate averages
    avg_metrics = {k: v / total_batches for k, v in all_losses.items()}
    return avg_metrics


def validate(model, val_loader, criterion, accelerator, args, current_epoch, loss_weighter=None, logger=None):
    model.eval()
    total_batches = len(val_loader)
    
    all_losses = {
        'token_loss': 0.0,
        'harmony_loss': 0.0,
        'rhythm_loss': 0.0,
        'contour_loss': 0.0,
        'total_loss': 0.0
    }
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Validation Epoch {current_epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            outputs = model(
                batch['image'],
                batch['tokens'],
                attention_mask=batch['attention_mask']
            )
            
            # Get individual losses (tensors)
            loss_dict = criterion(
                outputs,
                batch['tokens'],
                attention_mask=batch['attention_mask']
            )
            
            # Check for NaNs in loss components
            for loss_name, loss_value in loss_dict.items():
                if torch.isnan(loss_value):
                    logger.error(f"NaN detected in {loss_name} at batch {batch_idx} of epoch {current_epoch+1}")
                    raise ValueError(f"NaN detected in {loss_name}")
            
            # Use total_loss directly
            loss = loss_dict['total_loss']
            
            # Update totals
            for k, v in loss_dict.items():
                all_losses[k] += v.item()
            
            # Log validation step metrics at every step
            if accelerator.is_main_process:
                global_step = batch_idx + total_batches * current_epoch
                log_dict = {
                    'val/step': global_step,
                    'val/step_loss': loss.item(),
                }
                
                # Log individual validation losses
                for k, v in loss_dict.items():
                    log_dict[f'val/step_losses/{k}'] = v.item()
                
                wandb.log(log_dict, step=global_step)
            
            progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
    
    # Calculate averages
    avg_metrics = {k: v / total_batches for k, v in all_losses.items()}
    return avg_metrics

def create_datasets(args, logger):
    """Create training and validation datasets with proper error handling"""
    logger.info("Initializing datasets...")
    
    # Setup tokenizer parameters
    tokenizer_params = {
        'max_notes': 128,
        'max_velocity': 32,
        'time_step': 0.125,
        'max_time_shift': 100,
        'max_duration': args.max_duration,
        'special_tokens': {
            'PAD': 0,
            'BOS': 1,
            'EOS': 2
        }
    }
    
    try:
        # Create training dataset
        logger.info("Creating training dataset...")
        train_dataset = get_dataset(
            base_dir=args.data_dir,
            split='train',
            num_workers=args.num_workers,
            tokenizer_params=tokenizer_params,
            max_seq_length=args.max_seq_length,
            vit_model_name=args.vit_model,
            max_duration=args.max_duration
        )
        logger.info(f"Training dataset size: {len(train_dataset)}")
        
        # Create validation dataset
        logger.info("Creating validation dataset...")
        val_dataset = get_dataset(
            base_dir=args.data_dir,
            split='val',
            num_workers=args.num_workers,
            tokenizer_params=tokenizer_params,
            max_seq_length=args.max_seq_length,
            vit_model_name=args.vit_model,
            max_duration=args.max_duration
        )
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        logger.error(f"Error creating datasets: {str(e)}")
        raise


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
    pin_memory: bool = True
) -> DataLoader:
    """Create DataLoader with proper settings for training/validation."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=drop_last
    )

def create_tokenizer(args):
    """Create and initialize tokenizer with given parameters"""
    tokenizer_params = {
        'max_notes': 128,
        'max_velocity': 32,
        'time_step': 0.125,
        'max_time_shift': 100,
        'max_duration': args.max_duration,
        'special_tokens': {
            'PAD': 0,
            'BOS': 1,
            'EOS': 2
        }
    }
    
    return MIDITokenizer(**tokenizer_params), tokenizer_params

def main():
    args = parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Setup directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if main process
    if accelerator.is_main_process:
        wandb_run = setup_wandb_logging(args)
        args.save_dir = str(save_dir / wandb_run.name)
        os.makedirs(args.save_dir, exist_ok=True)
    
    logger = setup_logging(Path(args.save_dir))
    logger.info(f"Starting training with arguments: {args}")
    
    # Create tokenizer first
    tokenizer, tokenizer_params = create_tokenizer(args)
    
    # Calculate vocab size from tokenizer
    vocab_size = (
        len(tokenizer.special_tokens) +  # Special tokens
        2 * tokenizer.max_notes +        # Note on/off
        tokenizer.max_velocity +         # Velocity
        tokenizer.max_time_shift         # Time shift
    )
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Create datasets
    try:
        # Create training dataset
        logger.info("Creating training dataset...")
        train_dataset, val_dataset = create_datasets(args, logger)
        
    except Exception as e:
        logger.error(f"Error creating datasets: {str(e)}")
        raise
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    
    # Create model
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        vit_model=args.vit_model,
        freeze_vit=True
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Initialize newly added layers
    def initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    model.apply(initialize_weights)
    
    # Calculate total steps for scheduler
    total_steps = len(train_loader) * args.epochs
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
        min_lr=args.min_lr
    )
    
    # Initialize loss weighter if using dynamic weighting
    loss_weighter = None
    if args.use_dynamic_weighting:
        initial_weights = {
            'token_loss': args.token_loss_weight,
            'harmony_loss': args.harmony_loss_weight,
            'rhythm_loss': args.rhythm_loss_weight,
            'contour_loss': args.contour_loss_weight
        }
        loss_weighter = DynamicLossWeighter(
            initial_weights,
            decay_factor=args.weight_decay_factor
        )
    
    # Initialize criterion
    criterion = MusicGenerationLoss(
        vocab_size=vocab_size,
        pad_token_id=tokenizer.special_tokens['PAD']
    )
    
    # Prepare for distributed training
    model, optimizer, train_loader, val_loader, scheduler, criterion = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler, criterion
    )
    
    # Training loop
    early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
    best_val_loss = float('inf')
    
    logger.info("Starting training loop")
    
    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.epochs}")
        
        # Training
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            accelerator=accelerator,
            args=args,
            current_epoch=epoch,
            loss_weighter=loss_weighter,
            logger=logger  # Pass logger here
        )
        
        # Validation
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            accelerator=accelerator,
            args=args,
            current_epoch=epoch,
            loss_weighter=loss_weighter,
            logger=logger  # And here
        )
        
        # Log metrics
        if accelerator.is_main_process:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['total_loss'],  # Fixed
                'train/token_loss': train_metrics['token_loss'],
                'train/harmony_loss': train_metrics['harmony_loss'],
                'train/rhythm_loss': train_metrics['rhythm_loss'],
                'train/contour_loss': train_metrics['contour_loss'],
                'val/loss': val_metrics['total_loss'],  # Fixed
                'val/token_loss': val_metrics['token_loss'],
                'val/harmony_loss': val_metrics['harmony_loss'],
                'val/rhythm_loss': val_metrics['rhythm_loss'],
                'val/contour_loss': val_metrics['contour_loss'],
                'learning_rate': scheduler.get_last_lr()[0]
            })
            
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:  # Changed from 'loss' to 'total_loss'
                best_val_loss = val_metrics['total_loss']
                logger.info(f"New best validation loss: {best_val_loss:.4f}")
                accelerator.save_state(os.path.join(args.save_dir, 'best_model.pt'))
                wandb.run.summary['best_val_loss'] = best_val_loss
            
            # Regular checkpoint saving
            if (epoch + 1) % args.save_every == 0:
                save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                accelerator.save_state(save_path)
                logger.info(f"Saved checkpoint: {save_path}")
        
        # Early stopping check
        if early_stopping(val_metrics['total_loss']):  # Changed from 'loss' to 'total_loss'
            logger.info(f"Early stopping triggered after epoch {epoch+1}")
            break
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    # Final cleanup and logging
    if accelerator.is_main_process:
        wandb.run.summary.update({
            'final_train_loss': train_metrics['loss'],
            'final_val_loss': val_metrics['loss'],
            'total_epochs': epoch + 1,
            'best_val_loss': best_val_loss
        })
        wandb.finish()
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()