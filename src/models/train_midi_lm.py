#!/usr/bin/env python3
import argparse
import sys
import random
import logging
from pathlib import Path
from datetime import datetime
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm
import wandb
import pretty_midi

from src.models.midi_lm import MidiLanguageModel

# ========================================
# MIDI Tokenizer
# ========================================
class MIDITokenizer:
    def __init__(
        self,
        max_notes=128,
        max_velocity=32,
        time_step=0.125,
        max_time_shift=100,
        max_duration=30.0,
        special_tokens=None
    ):
        self.max_notes = max_notes
        self.max_velocity = max_velocity
        self.time_step = time_step
        self.max_time_shift = max_time_shift
        self.max_duration = max_duration

        # Offsets for token types
        self.NOTE_ON_OFFSET = 0
        self.NOTE_OFF_OFFSET = self.NOTE_ON_OFFSET + max_notes
        self.VELOCITY_OFFSET = self.NOTE_OFF_OFFSET + max_notes
        self.TIME_SHIFT_OFFSET = self.VELOCITY_OFFSET + max_velocity

        # Special tokens
        if special_tokens is None:
            special_tokens = {
                'PAD': 0,
                'BOS': 1,
                'EOS': 2
            }
        self.special_tokens = special_tokens
        self.pad_token_id = special_tokens['PAD']
        self.bos_token_id = special_tokens['BOS']
        self.eos_token_id = special_tokens['EOS']

        self.vocab_size = (
            2 * max_notes
            + max_velocity
            + self.max_time_shift
            + len(self.special_tokens)
        )

    def quantize_time(self, seconds: float):
        steps = int(round(seconds / self.time_step))
        return min(steps, self.max_time_shift - 1)

    def quantize_velocity(self, velocity: int):
        # map velocity 0-127 into 0-(max_velocity-1)
        return min(velocity * self.max_velocity // 128, self.max_velocity - 1)

    def encode_midi(self, midi_path: str):
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        tokens = [self.bos_token_id]
        current_time = 0.0

        # Collect notes (non-drum)
        notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                notes.extend(instrument.notes)
        notes.sort(key=lambda x: (x.start, x.pitch))

        for note in notes:
            if note.start >= self.max_duration:
                break
            # time shift until note start
            time_diff = note.start - current_time
            if time_diff > 0:
                time_steps = self.quantize_time(time_diff)
                if time_steps > 0:
                    tokens.append(self.TIME_SHIFT_OFFSET + time_steps)

            # velocity and note on
            velocity_token = self.quantize_velocity(note.velocity)
            tokens.append(self.VELOCITY_OFFSET + velocity_token)
            tokens.append(self.NOTE_ON_OFFSET + note.pitch)

            # note off (time shift to note end)
            note_end = min(note.end, self.max_duration)
            time_diff = note_end - note.start
            if time_diff > 0:
                time_steps = self.quantize_time(time_diff)
                if time_steps > 0:
                    tokens.append(self.TIME_SHIFT_OFFSET + time_steps)
            tokens.append(self.NOTE_OFF_OFFSET + note.pitch)
            current_time = note_end
            if current_time >= self.max_duration:
                break

        tokens.append(self.eos_token_id)
        return tokens

# ========================================
# Dataset
# ========================================

class MidiDataset(Dataset):
    def __init__(
        self,
        data_dir,
        tokenizer: MIDITokenizer,
        split='train',
        max_seq_len=512,
        test_ratio=0.1,
        val_ratio=0.1,
        seed=42,
        cache_dir="cache"  # Directory to store cached data
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        midi_files = list(self.data_dir.glob('*.mid'))
        if len(midi_files) == 0:
            raise ValueError(f"No MIDI files found in {self.data_dir}")

        random.seed(seed)
        random.shuffle(midi_files)

        total = len(midi_files)
        test_count = int(total * test_ratio)
        val_count = int(total * val_ratio)
        train_count = total - test_count - val_count

        if split == 'train':
            self.files = midi_files[:train_count]
        elif split == 'val':
            self.files = midi_files[train_count:train_count+val_count]
        else:
            self.files = midi_files[train_count+val_count:]

        print(f"Processing {split} MIDI files...")

        self.encoded_data = []
        for file in tqdm(self.files, desc=f"Processing {split} MIDI"):
            cache_file = self.cache_dir / f"{file.stem}.pkl"

            if cache_file.exists():
                # Load from cache if available
                with open(cache_file, "rb") as f:
                    tokens = pickle.load(f)
            else:
                # Encode and cache the result
                tokens = self._encode_file(file)
                if tokens is not None:
                    with open(cache_file, "wb") as f:
                        pickle.dump(tokens, f)

            if tokens is not None:
                self.encoded_data.append(tokens)

    def _encode_file(self, file_path):
        try:
            tokens = self.tokenizer.encode_midi(file_path)
            # Truncate or pad to max_seq_len
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            else:
                pad_len = self.max_seq_len - len(tokens)
                tokens = tokens + [self.tokenizer.pad_token_id] * pad_len
            return torch.tensor(tokens, dtype=torch.long)
        except Exception as e:
            print(f"Failed to encode {file_path}: {e}")
            return None

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return {'tokens': self.encoded_data[idx]}


# ========================================
# EarlyStopping
# ========================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.early_stop = False
        self.min_validation_loss = float('inf') if mode == 'min' else float('-inf')

    def __call__(self, val_loss):
        if self.mode == 'min':
            if val_loss < self.min_validation_loss - self.min_delta:
                self.min_validation_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
        else:
            if val_loss > self.min_validation_loss + self.min_delta:
                self.min_validation_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop

# ========================================
# Training and Evaluation
# ========================================
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, gradient_accumulation_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
        tokens = batch['tokens'].to(device)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        logits = model(inputs)
        loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))

        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        avg_loss = total_loss / (batch_idx + 1)

        # Update tqdm bar
        progress_bar.set_postfix({'loss': avg_loss})

    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Evaluating")
    for batch_idx, batch in enumerate(progress_bar):
        tokens = batch['tokens'].to(device)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        # Update tqdm bar
        progress_bar.set_postfix({'loss': avg_loss})

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train a MIDI Transformer Language Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing MIDI files')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Warmup steps')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--split_val', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--split_test', type=float, default=0.1, help='Test ratio')
    parser.add_argument('--project_name', type=str, default='midi_lm', help='WandB project name')
    parser.add_argument('--run_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--save_every', type=int, default=5, help='Save model every N epochs')
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = Path('checkpoints') / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(save_dir / 'training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    # Initialize wandb
    wandb.init(project=args.project_name, name=args.run_name, config=vars(args))

    # Tokenizer
    tokenizer = MIDITokenizer(max_duration=30.0)

    # Datasets
    train_dataset = MidiDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        split='train',
        max_seq_len=args.max_seq_len,
        test_ratio=args.split_test,
        val_ratio=args.split_val,
        cache_dir="cache/train"
    )

    val_dataset = MidiDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        split='val',
        max_seq_len=args.max_seq_len,
        test_ratio=args.split_test,
        val_ratio=args.split_val,
        cache_dir="cache/val"
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MidiLanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        pad_token_id=tokenizer.pad_token_id
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    early_stopping = EarlyStopping(patience=args.patience)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch} starting...")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, args.grad_accum_steps)
        val_loss = evaluate(model, val_loader, criterion, device)

        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        wandb.log({'train/loss': train_loss, 'val/loss': val_loss, 'epoch': epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / 'best_model.pt')
            wandb.log({'val/best_loss': best_val_loss})

        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), save_dir / f'checkpoint_epoch_{epoch}.pt')

        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered after epoch {epoch}")
            break

    wandb.finish()


if __name__ == "__main__":
    main()
