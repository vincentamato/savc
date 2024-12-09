import os
import torch
from torch.utils.data import Dataset
import random
from pathlib import Path
import pretty_midi
import logging
from PIL import Image
from typing import Dict, List, Optional
from torchvision import transforms

class MIDITokenizer:
    def __init__(
        self,
        max_notes: int = 128,
        max_velocity: int = 32,
        time_step: float = 0.125,
        max_time_shift: int = 100,
        special_tokens: Dict[str, int] = None
    ):
        self.max_notes = max_notes
        self.max_velocity = max_velocity
        self.time_step = time_step
        self.max_time_shift = max_time_shift
        
        self.NOTE_ON_OFFSET = 0
        self.NOTE_OFF_OFFSET = max_notes
        self.VELOCITY_OFFSET = 2 * max_notes
        self.TIME_SHIFT_OFFSET = 2 * max_notes + max_velocity
        
        self.special_tokens = special_tokens or {
            'PAD': 0,
            'BOS': 1,
            'EOS': 2,
            'MASK': 3
        }
        
        self.vocab_size = (
            2 * max_notes +
            max_velocity +
            max_time_shift +
            len(self.special_tokens)
        )
    
    def encode_midi(self, midi_path: str) -> List[int]:
        try:
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            
            tokens = [self.special_tokens['BOS']]
            current_time = 0.0
            
            notes = []
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    notes.extend(instrument.notes)
            notes.sort(key=lambda x: (x.start, x.pitch))
            
            for note in notes:
                # Time shift
                time_diff = note.start - current_time
                if time_diff > 0:
                    time_steps = self.quantize_time(time_diff)
                    if time_steps > 0:
                        tokens.append(self.TIME_SHIFT_OFFSET + time_steps)
                
                # Velocity
                velocity_token = self.quantize_velocity(note.velocity)
                tokens.append(self.VELOCITY_OFFSET + velocity_token)
                
                # Note on
                tokens.append(self.NOTE_ON_OFFSET + note.pitch)
                
                # Duration and note off
                time_diff = note.end - note.start
                if time_diff > 0:
                    time_steps = self.quantize_time(time_diff)
                    if time_steps > 0:
                        tokens.append(self.TIME_SHIFT_OFFSET + time_steps)
                tokens.append(self.NOTE_OFF_OFFSET + note.pitch)
                current_time = note.end
            
            tokens.append(self.special_tokens['EOS'])
            return tokens
            
        except Exception as e:
            logging.error(f"Error processing MIDI file {midi_path}: {str(e)}")
            return None
    
    def quantize_velocity(self, velocity: int) -> int:
        return min(self.max_velocity - 1, velocity * self.max_velocity // 128)
    
    def quantize_time(self, time: float) -> int:
        steps = int(round(time / self.time_step))
        return min(self.max_time_shift - 1, steps)

class ImageMIDIDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        max_seq_length: int = 1024,
        image_size: int = 384,
        split: Optional[str] = None,
        val_size: float = 0.1,
        test_size: float = 0.1,
        seed: int = 42
    ):
        self.base_dir = Path(base_dir)
        self.max_seq_length = max_seq_length
        self.tokenizer = MIDITokenizer()
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Build dataset only once
        all_pairs = self._find_pairs()
        
        # Handle split if specified
        if split is not None:
            random.seed(seed)
            indices = list(range(len(all_pairs)))
            random.shuffle(indices)
            
            total_size = len(indices)
            test_idx = int(total_size * (1 - test_size))
            val_idx = int(test_idx * (1 - val_size))
            
            if split == 'train':
                selected_indices = indices[:val_idx]
            elif split == 'val':
                selected_indices = indices[val_idx:test_idx]
            else:  # test
                selected_indices = indices[test_idx:]
                
            self.valid_pairs = [all_pairs[i] for i in selected_indices]
        else:
            self.valid_pairs = all_pairs
            
        print(f"Loaded {len(self.valid_pairs)} valid image-MIDI pairs for {split if split else 'full'} set")
        
        if len(self.valid_pairs) == 0:
            raise ValueError("No valid image-MIDI pairs found!")

    def _find_pairs(self) -> List[Dict]:
        """Find and create all valid image-MIDI pairs only once."""
        valid_pairs = []
        midi_dir = self.base_dir / 'maestro'
        art_dir = self.base_dir / 'artbench-10'
        
        # Cache MIDI files and their tokens
        midi_data = []
        for root, _, files in os.walk(midi_dir):
            for file in files:
                if file.endswith(('.midi', '.mid')):
                    midi_path = os.path.join(root, file)
                    tokens = self.tokenizer.encode_midi(midi_path)
                    if tokens is not None and len(tokens) <= self.max_seq_length:
                        midi_data.append((midi_path, tokens))
        
        # Cache image files
        image_data = []
        for style in os.listdir(art_dir):
            style_dir = art_dir / style
            if not style_dir.is_dir():
                continue
            
            for file in os.listdir(style_dir):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    image_data.append((os.path.join(style_dir, file), style))
        
        # Create pairs
        for midi_path, tokens in midi_data:
            for image_path, style in image_data:
                valid_pairs.append({
                    'image_path': image_path,
                    'midi_path': midi_path,
                    'tokens': tokens,
                    'style': style
                })
        
        return valid_pairs
    
    def __len__(self) -> int:
        return len(self.valid_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.valid_pairs[idx]
        
        # Load and transform image
        image = Image.open(pair['image_path']).convert('RGB')
        image_tensor = self.transform(image)
        
        # Pad tokens if necessary
        tokens = pair['tokens']
        if len(tokens) < self.max_seq_length:
            tokens = tokens + [self.tokenizer.special_tokens['PAD']] * (self.max_seq_length - len(tokens))
        
        return {
            'image': image_tensor,
            'tokens': torch.LongTensor(tokens),
            'length': len(pair['tokens']),
            'style': pair['style']
        }

def create_dataloader(
    dataset: ImageMIDIDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

def get_dataset(
    base_dir: str,
    split: str = 'train',
    **dataset_kwargs
) -> ImageMIDIDataset:
    """Get dataset split directly without creating multiple instances."""
    return ImageMIDIDataset(base_dir=base_dir, split=split, **dataset_kwargs)

if __name__ == "__main__":
    dataset = ImageMIDIDataset(base_dir="data")
    loader = create_dataloader(dataset, batch_size=32)
    
    for batch in loader:
        print("Image shape:", batch['image'].shape)
        print("Tokens shape:", batch['tokens'].shape)
        print("Sequence lengths:", batch['length'])
        print("Styles:", batch['style'])
        break