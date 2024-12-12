import os
import json
import torch
from torch.utils.data import Dataset
import random
from pathlib import Path
import pretty_midi
import logging
from PIL import Image
from typing import Dict, List, Optional, Tuple
from torchvision import transforms
from tqdm import tqdm
import pickle
from multiprocessing import Pool
from functools import partial
import numpy as np

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

def tokenize_midi_file(args: Tuple[str, str, Dict]) -> Optional[Dict]:
    midi_path, cache_path, tokenizer_params = args
    cache_file = Path(cache_path) / f"{Path(midi_path).stem}.pkl"
    
    # Check cache first
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return {'path': midi_path, 'tokens': pickle.load(f)}
        except Exception:
            pass
    
    # If not in cache, tokenize
    tokenizer = MIDITokenizer(**tokenizer_params)
    tokens = tokenizer.encode_midi(midi_path)
    
    if tokens is not None:
        # Save to cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(tokens, f)
        return {'path': midi_path, 'tokens': tokens}
    
    return None

class ImageMIDIDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        max_seq_length: int = 1024,
        image_size: int = 384,
        split: Optional[str] = None,
        val_size: float = 0.1,
        test_size: float = 0.1,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        num_workers: int = 32
    ):
        self.base_dir = Path(base_dir)
        self.max_seq_length = max_seq_length
        self.cache_dir = Path(cache_dir) if cache_dir else self.base_dir / 'token_cache'
        
        print("Initializing dataset transforms...")
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Load matches data
        print("Loading matches.json...")
        matches_path = self.base_dir / 'matches.json'
        with open(matches_path) as f:
            self.matches_data = json.load(f)
        
        # Build initial pairs without tokenization
        print("Creating initial pairs...")
        self.all_pairs = []
        unique_midis = set()
        
        for artwork_path, match_data in self.matches_data['matches'].items():
            # Fix artwork path by removing data/ prefix if it exists
            artwork_path = artwork_path.replace('data/', '')
            artwork_path = self.base_dir / artwork_path
            
            style = artwork_path.parent.name
            
            for midi_match in match_data['midi_matches']:
                # Fix MIDI path
                midi_path = midi_match['midi_path'].replace('data/unique_midis/', 'midis/')
                midi_path = self.base_dir / midi_path
                unique_midis.add(str(midi_path))
                
                self.all_pairs.append({
                    'image_path': artwork_path,
                    'midi_path': midi_path,
                    'style': style,
                    'similarity_score': midi_match['similarity_score'],
                    'emotional_match': midi_match['emotional_match'],
                    'artwork_info': match_data['artwork_info']
                })
        
        print(f"Found {len(self.all_pairs)} total pairs using {len(unique_midis)} unique MIDI files")
        
        
        # Tokenize all unique MIDI files in parallel
        print("Tokenizing MIDI files in parallel...")
        tokenizer_params = {
            'max_notes': 128,
            'max_velocity': 32,
            'time_step': 0.125,
            'max_time_shift': 100
        }
        
        with Pool(num_workers) as pool:
            midi_args = [(str(path), str(self.cache_dir), tokenizer_params) for path in unique_midis]
            tokens_data = list(tqdm(
                pool.imap(tokenize_midi_file, midi_args),
                total=len(midi_args),
                desc="Tokenizing MIDIs"
            ))
        
        # Create tokens lookup dictionary
        self.tokens_lookup = {
            item['path']: item['tokens'] 
            for item in tokens_data 
            if item is not None
        }
        
        # Add tokens to pairs
        self.valid_pairs = [
            {**pair, 'tokens': self.tokens_lookup[str(pair['midi_path'])]}
            for pair in self.all_pairs
            if str(pair['midi_path']) in self.tokens_lookup
        ]
        
        print(f"Successfully processed {len(self.valid_pairs)} valid pairs")
        
        # Handle split if specified
        if split is not None:
            random.seed(seed)
            indices = list(range(len(self.valid_pairs)))
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
                
            self.valid_pairs = [self.valid_pairs[i] for i in selected_indices]
            print(f"Final {split} set size: {len(self.valid_pairs)} pairs")
    
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
            tokens = tokens + [0] * (self.max_seq_length - len(tokens))
        else:
            tokens = tokens[:self.max_seq_length]
        
        return {
            'image': image_tensor,
            'tokens': torch.LongTensor(tokens),
            'length': min(len(pair['tokens']), self.max_seq_length),
            'style': pair['style'],
            'similarity_score': pair['similarity_score'],
            'emotional_match': pair['emotional_match'],
            'artwork_info': pair['artwork_info']
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
    num_workers: int = 32,
    **dataset_kwargs
) -> ImageMIDIDataset:
    """Get dataset split directly without creating multiple instances."""
    return ImageMIDIDataset(base_dir=base_dir, split=split, num_workers=num_workers, **dataset_kwargs)