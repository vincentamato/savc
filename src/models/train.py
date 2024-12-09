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

from src.models.model import MusicGenerationTransformer, MIDITokenizer
from src.data.dataset import get_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Music Generation Training Script')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help='Number of gradient accumulation steps (default: 4)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW (default: 0.01)')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps for learning rate scheduler')
    
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
    
    return parser.parse_args()

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
        tokenizer=tokenizer
    )
    return model

def get_lr_scheduler(optimizer, args, total_steps):
    from transformers import get_linear_schedule_with_warmup
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

# Add this at the top with other imports
def scheduled_teacher_forcing(epoch, min_ratio=0.2, max_ratio=1.0, num_epochs=100):
    """Calculate teacher forcing ratio based on training progress"""
    ratio = max_ratio - (max_ratio - min_ratio) * (epoch / num_epochs)
    return max(min_ratio, ratio)

# Update train_one_batch function
def train_one_batch(model, batch, optimizer, scheduler, scaler, criterion, device, epoch, args, accumulating=True):
    images = batch['image'].to(device)
    tokens = batch['tokens'].to(device)
    
    with amp.autocast('cuda'):
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
        # Scale loss for gradient accumulation
        if accumulating:
            loss = loss / args.gradient_accumulation_steps
    
    scaler.scale(loss).backward()
    
    # Only optimize after accumulating enough gradients
    if not accumulating:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
    
    return loss.item() * (args.gradient_accumulation_steps if accumulating else 1)

def train(args):
    # Initialize wandb
    if args.run_name is None:
        args.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    wandb.init(
        project=args.project_name,
        name=args.run_name,
        config=vars(args)
    )
    
    # Setup
    device = torch.device('cuda')
    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    train_dataset = get_dataset(args.data_dir, split='train')
    val_dataset = get_dataset(args.data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model
    model = create_model(args)
    
    # Initialize ViT weights properly
    if hasattr(model.vit, 'pooler'):
        # Initialize pooler weights if they exist
        nn.init.normal_(model.vit.pooler.dense.weight, std=0.02)
        nn.init.zeros_(model.vit.pooler.dense.bias)
    
    model = model.to(device)
    wandb.watch(model, log_freq=100)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scaler = amp.GradScaler('cuda')
    scheduler = get_lr_scheduler(optimizer, args, len(train_loader) * args.epochs)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch} - Train')
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for batch_idx, batch in enumerate(train_pbar):
            # Determine if we should accumulate or optimize
            is_accumulating = (batch_idx + 1) % args.gradient_accumulation_steps != 0
            
            loss = train_one_batch(
                model, batch, optimizer, scheduler, scaler, criterion, 
                device, epoch, args, accumulating=is_accumulating
            )
            train_losses.append(loss)
            
            # Clear GPU memory cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
            
            # Update progress bar and wandb only on optimization steps
            if not is_accumulating:
                train_pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                if (batch_idx // args.gradient_accumulation_steps) % 10 == 0:
                    wandb.log({
                        'train/loss': loss,
                        'train/learning_rate': scheduler.get_last_lr()[0],
                        'train/epoch': epoch + batch_idx / len(train_loader)
                    })
        
        # Validation loop
        model.eval()
        val_losses = []
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch} - Val')
        
        with torch.no_grad():
            for batch in val_pbar:
                # Use smaller batch size for validation if needed
                images = batch['image'].to(device)
                tokens = batch['tokens'].to(device)
                
                outputs = model(images, tokens)  # No teacher forcing in validation
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = tokens[..., 1:].contiguous()
                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                val_losses.append(loss.item())
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Clear cache after each validation batch
                torch.cuda.empty_cache()
        
        # Calculate epoch metrics
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train/epoch_loss': train_loss,
            'val/epoch_loss': val_loss,
        })
        
        # Save best model and checkpoints
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

    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    train(args)