import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
from datetime import datetime
import numpy as np
import math
import sys
import logging
from accelerate import Accelerator
import gc

from src.models.model import MusicGenerationTransformer
from src.data.dataset import get_dataset, MIDITokenizer

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.min_validation_loss = float('inf') if mode == 'min' else float('-inf')
    
    def __call__(self, validation_loss):
        if self.mode == 'min':
            if validation_loss < self.min_validation_loss - self.min_delta:
                self.min_validation_loss = validation_loss
                self.counter = 0
            else:
                self.counter += 1
        else:
            if validation_loss > self.min_validation_loss + self.min_delta:
                self.min_validation_loss = validation_loss
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop

class GradientFlow:
    def __init__(self, model):
        self.model = model
        self.gradients = {}
        
    def log_gradients(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.gradients[f"gradients/{name}"] = {
                    'mean': param.grad.abs().mean().item(),
                    'std': param.grad.std().item(),
                    'max': param.grad.abs().max().item()
                }
        return self.gradients

def parse_args():
    parser = argparse.ArgumentParser(description='Music Generation Training Script')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Number of gradient accumulation steps (default: 4)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate (default: 2e-4)')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate (default: 1e-6)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW (default: 0.01)')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                        help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping (default: 10)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor (default: 0.1)')
    
    # Model configuration
    parser.add_argument('--d_model', type=int, default=768,
                        help='Transformer dimension (default: 768)')
    parser.add_argument('--nhead', type=int, default=12,
                        help='Number of attention heads (default: 12)')
    parser.add_argument('--num_encoder_layers', type=int, default=6,
                        help='Number of encoder layers (default: 6)')
    parser.add_argument('--num_decoder_layers', type=int, default=6,
                        help='Number of decoder layers (default: 6)')
    parser.add_argument('--dim_feedforward', type=int, default=3072,
                        help='Feedforward dimension (default: 3072)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    
    # System parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the dataset')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers (default: 8)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value (default: 1.0)')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--max_duration', type=float, default=30.0,
                        help='Maximum duration of MIDI files in seconds (default: 30.0)')
    
    # Wandb configuration
    parser.add_argument('--project_name', type=str, default='savc',
                        help='WandB project name (default: savc)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='WandB run name (default: timestamp)')
    
    parser.add_argument('--max_seq_length', type=int, default=1024,  # Added max sequence length
                       help='Maximum sequence length (default: 512)')
    
    args = parser.parse_args()
    print(f"Parsed args: {vars(args)}")
    
    return args

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

def create_model(args):
    from transformers import AutoConfig

    # Configure ViT to be more memory efficient
    vit_config = AutoConfig.from_pretrained(
        "google/vit-large-patch16-384",
        output_attentions=False,
        output_hidden_states=False,
        output_scores=False,
        return_dict=False
    )

    tokenizer = MIDITokenizer(max_duration=args.max_duration)
    model = MusicGenerationTransformer(
        vit_name="google/vit-large-patch16-384",
        vit_config=vit_config,  # Pass the config
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,  # Make sure this matches
        tokenizer=tokenizer,
        label_smoothing=args.label_smoothing
    )

    # Enable gradient checkpointing only for the Vision Transformer if needed
    if hasattr(model.vit, "gradient_checkpointing_enable"):
        model.vit.gradient_checkpointing_enable()

    return model

def scheduled_teacher_forcing(epoch, min_ratio=0.2, max_ratio=1.0, num_epochs=100):
    """Calculate teacher forcing ratio based on training progress."""
    ratio = max_ratio - (max_ratio - min_ratio) * (epoch / num_epochs)
    return max(min_ratio, ratio)

def get_lr_scheduler(optimizer, args, total_steps):
    """Creates a learning rate scheduler with linear warmup and cosine decay."""
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        
        progress = float(current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return max(args.min_lr / args.lr, cosine_decay)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_batch(model, batch, optimizer, scheduler, criterion, accelerator, epoch, args, accumulating=True):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    images = batch['image']
    tokens = batch['tokens']
    
    teacher_ratio = scheduled_teacher_forcing(
        epoch,
        min_ratio=0.2,
        max_ratio=1.0,
        num_epochs=args.epochs
    )
    
    with accelerator.autocast():
        outputs = model(images, tokens, teacher_forcing_ratio=teacher_ratio)
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = tokens[..., 1:].contiguous()
        
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
    
    if accumulating:
        loss = loss / args.gradient_accumulation_steps
    
    accelerator.backward(loss)
    
    if not accumulating:
        accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
    
    return loss.item() * (args.gradient_accumulation_steps if accumulating else 1)


def validate(model, val_loader, criterion, accelerator):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image']
            tokens = batch['tokens']
            
            outputs = model(images, tokens)
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = tokens[..., 1:].contiguous()
            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
            
    return total_loss / num_batches

def train(args):
    # Memory management setup
    torch.cuda.empty_cache()
    gc.collect()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    accelerator = Accelerator()
    
    # Setup
    if not hasattr(args, 'run_name') or args.run_name is None:
        args.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(save_dir)
    
    if accelerator.is_main_process:
        wandb.init(
            project=args.project_name,
            name=args.run_name,
            config=vars(args)
        )
    
    # Data loading
     # Data loading
    train_dataset = get_dataset(
        args.data_dir, 
        split='train',
        max_duration=args.max_duration,
        max_seq_length=args.max_seq_length  # Make sure this is passed
    )
    val_dataset = get_dataset(
        args.data_dir, 
        split='val',
        max_duration=args.max_duration,
        max_seq_length=args.max_seq_length  # Make sure this is passed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Model setup
    model = create_model(args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_lr_scheduler(optimizer, args, total_steps)
    
    # Prepare for distributed training
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    if accelerator.is_main_process:
        wandb.watch(model, log_freq=100)
    
    early_stopping = EarlyStopping(patience=args.patience)
    gradient_flow = GradientFlow(model)
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        
        with tqdm(train_loader, desc=f'Epoch {epoch} - Train', disable=not accelerator.is_main_process) as train_pbar:
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_pbar):
                is_accumulating = (batch_idx + 1) % args.gradient_accumulation_steps != 0
                
                loss = train_one_batch(
                    model=model,
                    batch=batch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    accelerator=accelerator,
                    epoch=epoch,
                    args=args,
                    accumulating=is_accumulating
                )
                
                train_losses.append(loss)
                
                if not is_accumulating and accelerator.is_main_process:
                    grad_metrics = gradient_flow.log_gradients()
                    wandb.log({
                        'train/loss': loss,
                        'train/learning_rate': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + batch_idx / len(train_loader),
                        **grad_metrics
                    })
                
                train_pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
        
        # Validation
        val_loss = validate(model, val_loader, criterion, accelerator)
        
        # Logging
        train_loss = np.mean(train_losses)
        logger.info(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        if accelerator.is_main_process:
            wandb.log({
                'epoch': epoch,
                'train/epoch_loss': train_loss,
                'val/epoch_loss': val_loss,
            })
            
            # Model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                accelerator.save_state(save_dir / 'best_model.pt')
                wandb.log({'val/best_loss': best_val_loss})
            
            if (epoch + 1) % args.save_every == 0:
                accelerator.save_state(save_dir / f'checkpoint_epoch_{epoch}.pt')
        
        # Early stopping check
        if early_stopping(val_loss):
            logger.info(f'Early stopping triggered after epoch {epoch}')
            break
    
    if accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    train(args)