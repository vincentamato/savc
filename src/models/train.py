import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.amp as amp
from tqdm import tqdm
import wandb
import argparse
from datetime import datetime
import numpy as np
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import sys
import logging
from contextlib import nullcontext

from src.models.model import MusicGenerationTransformer, MIDITokenizer
from src.data.dataset import get_dataset

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
    
    # Enhanced training hyperparameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Number of gradient accumulation steps (default: 1)')
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
    
    # H100 optimizations
    parser.add_argument('--precision', type=str, default='bf16', 
                        choices=['fp32', 'fp16', 'bf16'],
                        help='Training precision (default: bf16)')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save memory')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() for faster training')
    
    # Model configuration
    parser.add_argument('--d_model', type=int, default=1024,
                        help='Transformer dimension (default: 1024)')
    parser.add_argument('--nhead', type=int, default=16,
                        help='Number of attention heads (default: 16)')
    parser.add_argument('--num_encoder_layers', type=int, default=8,
                        help='Number of encoder layers (default: 8)')
    parser.add_argument('--num_decoder_layers', type=int, default=8,
                        help='Number of decoder layers (default: 8)')
    parser.add_argument('--dim_feedforward', type=int, default=4096,
                        help='Feedforward dimension (default: 4096)')
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
    
    # Wandb configuration
    parser.add_argument('--project_name', type=str, default='savc',
                        help='WandB project name (default: svac)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='WandB run name (default: timestamp)')
    
    args = parser.parse_args()
    print(f"Parsed args: {vars(args)}") 
    
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

def create_model(args):
    tokenizer = MIDITokenizer()
    model = MusicGenerationTransformer(
        vit_name="google/vit-large-patch16-384",
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_length=1024,
        tokenizer=tokenizer,
        label_smoothing=args.label_smoothing
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    if args.compile:
        model = torch.compile(model)
    
    return model

def scheduled_teacher_forcing(epoch, min_ratio=0.2, max_ratio=1.0, num_epochs=100):
    """Calculate teacher forcing ratio based on training progress.
    
    Args:
        epoch (int): Current training epoch
        min_ratio (float): Minimum teacher forcing ratio (default: 0.2)
        max_ratio (float): Maximum teacher forcing ratio (default: 1.0)
        num_epochs (int): Total number of training epochs (default: 100)
    
    Returns:
        float: Teacher forcing ratio for current epoch
    """
    ratio = max_ratio - (max_ratio - min_ratio) * (epoch / num_epochs)
    return max(min_ratio, ratio)

def get_lr_scheduler(optimizer, args, total_steps):
    """Creates a learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: The optimizer to schedule
        args: Training arguments containing warmup_steps and min_lr
        total_steps: Total number of training steps
    
    Returns:
        torch.optim.lr_scheduler: Learning rate scheduler
    """
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        
        # Cosine decay with minimum learning rate
        progress = float(current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        # Scale between 1.0 and minimum learning rate
        return max(args.min_lr / args.lr, cosine_decay)
    
    return LambdaLR(optimizer, lr_lambda)

def train_one_batch(model, batch, optimizer, scheduler, criterion, device, epoch, args, accumulating=True):
    images = batch['image'].to(device, non_blocking=True)
    tokens = batch['tokens'].to(device, non_blocking=True)
    
    teacher_ratio = scheduled_teacher_forcing(
        epoch,
        min_ratio=0.2,
        max_ratio=1.0,
        num_epochs=args.epochs
    )
    
    outputs = model(images, tokens, teacher_forcing_ratio=teacher_ratio)
    shift_logits = outputs[..., :-1, :].contiguous()
    shift_labels = tokens[..., 1:].contiguous()
    loss = criterion(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    if accumulating:
        loss = loss / args.gradient_accumulation_steps
    
    loss.backward()
    
    if not accumulating:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    
    return loss.item() * (args.gradient_accumulation_steps if accumulating else 1)

def validate(model, val_loader, criterion, device, args):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device, non_blocking=True)
            tokens = batch['tokens'].to(device, non_blocking=True)
            
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
    # Setup
    if not hasattr(args, 'run_name') or args.run_name is None:
        args.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(save_dir)
    
    wandb.init(
        project=args.project_name,
        name=args.run_name,
        config=vars(args)
    )
    
    # Device setup
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Data loading
    train_dataset = get_dataset(args.data_dir, split='train')
    val_dataset = get_dataset(args.data_dir, split='val')
    
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
    model = model.to(device)
    
    wandb.watch(model, log_freq=100)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_lr_scheduler(optimizer, args, total_steps)
    early_stopping = EarlyStopping(patience=args.patience)
    gradient_flow = GradientFlow(model)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        
        with tqdm(train_loader, desc=f'Epoch {epoch} - Train') as train_pbar:
            optimizer.zero_grad(set_to_none=True)
            
            for batch_idx, batch in enumerate(train_pbar):
                is_accumulating = (batch_idx + 1) % args.gradient_accumulation_steps != 0
                
                loss = train_one_batch(
                    model=model,
                    batch=batch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    device=device,
                    epoch=epoch,
                    args=args,
                    accumulating=is_accumulating
                )
                
                train_losses.append(loss)
                
                if not is_accumulating:
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
        val_loss = validate(model, val_loader, criterion, device, args)
        
        # Logging
        train_loss = np.mean(train_losses)
        logger.info(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        wandb.log({
            'epoch': epoch,
            'train/epoch_loss': train_loss,
            'val/epoch_loss': val_loss,
        })
        
        # Model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'args': args
            }, save_dir / 'best_model.pt')
            
            wandb.log({'val/best_loss': best_val_loss})
        
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'args': args
            }, save_dir / f'checkpoint_epoch_{epoch}.pt')
        
        # Early stopping check
        if early_stopping(val_loss):
            logger.info(f'Early stopping triggered after epoch {epoch}')
            break
    
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    train(args)