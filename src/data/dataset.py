import os
import json
import gc
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from pathlib import Path
import pretty_midi
import logging
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
from torchvision import transforms
from tqdm import tqdm
import pickle
from multiprocessing import Pool
import numpy as np
from functools import partial
from dataclasses import dataclass
from transformers import ViTImageProcessor
import hashlib

@dataclass
class MIDIStats:
    """Statistics for MIDI processing"""
    duration: float
    num_events: int
    num_tokens: int
    was_truncated: bool = False
    original_duration: Optional[float] = None
    original_events: Optional[int] = None

class MIDITokenizer:
    def __init__(
        self,
        max_notes: int = 128,
        max_velocity: int = 32,
        time_step: float = 0.125,
        max_time_shift: int = 100,
        special_tokens: Dict[str, int] = None,
        max_duration: float = 30.0
    ):
        self.max_notes = max_notes
        self.max_velocity = max_velocity
        self.time_step = time_step
        self.max_time_shift = max_time_shift
        self.max_duration = max_duration
        
        # Set up special tokens
        if special_tokens is None:
            special_tokens = {
                'PAD': 0,
                'BOS': 1,
                'EOS': 2
            }
        self.special_tokens = special_tokens

        # Number of special tokens
        num_special_tokens = len(self.special_tokens)  # Should be 3 in this case

        # Adjust offsets so special tokens do not overlap with musical tokens
        self.NOTE_ON_OFFSET = num_special_tokens
        self.NOTE_OFF_OFFSET = self.NOTE_ON_OFFSET + self.max_notes
        self.VELOCITY_OFFSET = self.NOTE_OFF_OFFSET + self.max_notes
        self.TIME_SHIFT_OFFSET = self.VELOCITY_OFFSET + self.max_velocity

        self.vocab_size = (
            num_special_tokens +
            2 * self.max_notes +  # note on + note off
            self.max_velocity +
            self.max_time_shift
        )
    
    def quantize_time(self, seconds: float) -> int:
        """Convert time in seconds to number of time steps."""
        steps = int(round(seconds / self.time_step))
        return min(steps, self.max_time_shift - 1)
    
    def quantize_velocity(self, velocity: int) -> int:
        """Convert MIDI velocity (0-127) to quantized velocity token."""
        return min(velocity * self.max_velocity // 128, self.max_velocity - 1)

    def encode_midi(self, midi_path: str) -> Optional[List[int]]:
        """Encode MIDI file to token sequence."""
        try:
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            tokens = [self.special_tokens['BOS']]
            current_time = 0.0
            
            # Collect and sort all notes
            notes = []
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    notes.extend(instrument.notes)
            notes.sort(key=lambda x: (x.start, x.pitch))
            
            for note in notes:
                if note.start >= self.max_duration:
                    break
                
                # Time shift
                time_diff = note.start - current_time
                if time_diff > 0:
                    time_steps = self.quantize_time(time_diff)
                    if time_steps > 0:
                        tokens.append(self.TIME_SHIFT_OFFSET + time_steps)
                
                # Velocity and note on
                velocity_token = self.quantize_velocity(note.velocity)
                tokens.append(self.VELOCITY_OFFSET + velocity_token)
                tokens.append(self.NOTE_ON_OFFSET + note.pitch)
                
                # Duration and note off
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
            
            tokens.append(self.special_tokens['EOS'])
            return tokens
                
        except Exception as e:
            logging.error(f"Error processing MIDI file {midi_path}: {str(e)}")
            return None

def tokenize_midi_file_sequential(midi_path: str, tokenizer_params: Dict, cache_dir: str) -> Optional[Dict]:
    """
    Tokenize a single MIDI file with caching for sequential processing.

    Args:
        midi_path (str): Path to the MIDI file.
        tokenizer_params (Dict): Parameters for the MIDITokenizer.
        cache_dir (str): Directory to store cached tokens.

    Returns:
        Optional[Dict]: Dictionary containing MIDI path and tokens, or None if processing fails.
    """
    cache_file = Path(cache_dir) / f"{Path(midi_path).stem}.pt"

    try:
        # Check cache first
        if cache_file.exists():
            tokens = torch.load(cache_file)
            return {'path': midi_path, 'tokens': tokens}

        # Tokenize
        tokenizer = MIDITokenizer(**tokenizer_params)
        tokens = tokenizer.encode_midi(midi_path)

        if tokens is not None:
            tokens = torch.tensor(tokens, dtype=torch.long)
            # Save to cache using numpy first
            npy_cache = str(cache_file).replace('.pt', '.npy')
            np.save(npy_cache, tokens.numpy())
            torch.save(tokens, cache_file)
            os.remove(npy_cache)
            return {'path': midi_path, 'tokens': tokens}

    except Exception as e:
        error_message = f"Error processing {midi_path}: {str(e)}"
        print(error_message)
        logging.error(error_message)
        return None
    
class ImageMIDIDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        max_seq_length: int = 1024,
        vit_model_name: str = "google/vit-base-patch16-384",
        split: Optional[str] = None,
        val_size: float = 0.1,
        test_size: float = 0.1,
        seed: int = 42,
        num_workers: int = 1,
        max_duration: float = 30.0,
        use_cache: bool = True,
        tokenizer_params: Optional[Dict] = None
    ):
        self.base_dir = Path(base_dir)
        self.max_seq_length = max_seq_length
        self.use_cache = use_cache
        self.vit_model_name = vit_model_name  # Store the model name
        
        # Initialize image processor
        self.image_processor = ViTImageProcessor.from_pretrained(vit_model_name)

        # Use existing cache directory
        self.cache_dir = Path("~/savc/cache").expanduser()
        self.token_cache_dir = self.cache_dir / 'tokens'
        self.image_cache_dir = self.cache_dir / 'images'
        
        if use_cache:
            print(f"\nUsing existing cache directories:")
            print(f"Token cache: {self.token_cache_dir}")
            print(f"Image cache: {self.image_cache_dir}")
            
            if not self.token_cache_dir.exists() or not self.image_cache_dir.exists():
                print("Warning: Cache directories not found. Creating new cache directories.")
                self.token_cache_dir.mkdir(parents=True, exist_ok=True)
                self.image_cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tokenizer
        default_tokenizer_params = {
            'max_notes': 128,
            'max_velocity': 32,
            'time_step': 0.125,
            'max_time_shift': 100,
            'max_duration': max_duration,
            'special_tokens': {
                'PAD': 0,
                'BOS': 1,
                'EOS': 2
            }
        }
        if tokenizer_params:
            default_tokenizer_params.update(tokenizer_params)
        
        self.tokenizer = MIDITokenizer(**default_tokenizer_params)

        # Load matches and create pairs
        self._load_matches_and_create_pairs()
        
        # Process only uncached MIDI files
        if use_cache:
            self._load_or_process_midi_files(default_tokenizer_params)

        # Handle dataset split
        if split is not None:
            self._split_dataset(split, val_size, test_size, seed)

        self._compute_statistics()

    def _process_image(self, image_path: str) -> torch.Tensor:
        """Process an image file into a tensor suitable for ViT."""
        try:
            # Check cache first
            cache_path = self._get_image_cache_path(str(image_path))
            if self.use_cache and cache_path.exists():
                return torch.load(cache_path)

            # Load and convert image
            image = Image.open(image_path).convert('RGB')
            
            # Process image using ViT processor
            inputs = self.image_processor(images=image, return_tensors="pt")
            image_tensor = inputs['pixel_values'].squeeze(0)

            # Cache the processed tensor
            if self.use_cache:
                torch.save(image_tensor, cache_path)

            return image_tensor

        except Exception as e:
            logging.error(f"Error processing image {image_path}: {str(e)}")
            # Return a zero tensor with the correct shape for ViT-base
            return torch.zeros((3, 384, 384))

    def _get_image_cache_path(self, image_path: str) -> Path:
        """Generate a cache file path for processed images."""
        # Create a unique identifier based on the image path and vit model
        identifier = f"{image_path}_{self.vit_model_name}"
        cache_name = hashlib.md5(identifier.encode()).hexdigest()
        return self.image_cache_dir / f"{cache_name}.pt"

    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        pair = self.valid_pairs[idx]
        
        # Process image
        image_tensor = self._process_image(pair['image_path'])
        
        # Get tokens from lookup
        tokens = self.tokens_lookup.get(pair['midi_path'], torch.zeros(self.max_seq_length, dtype=torch.long))
        tokens = tokens[:self.max_seq_length].clone()
        
        # Create token types
        token_types = torch.zeros_like(tokens)  # Simplified token types
        
        # Pad sequences if necessary
        if len(tokens) < self.max_seq_length:
            padding_length = self.max_seq_length - len(tokens)
            tokens = torch.cat([tokens, torch.zeros(padding_length, dtype=torch.long)])
            token_types = torch.cat([token_types, torch.zeros(padding_length, dtype=torch.long)])
            attention_mask = torch.cat([
                torch.ones(len(self.tokens_lookup.get(pair['midi_path'], [])), dtype=torch.bool),
                torch.zeros(padding_length, dtype=torch.bool)
            ])
        else:
            attention_mask = torch.ones(self.max_seq_length, dtype=torch.bool)
        
        return {
            'image': image_tensor,
            'tokens': tokens,
            'token_types': token_types,
            'attention_mask': attention_mask,
            'length': min(len(self.tokens_lookup.get(pair['midi_path'], [])), self.max_seq_length),
            'style': pair['style'],
            'similarity_score': pair['similarity_score'],
            'emotional_match': pair['emotional_match']
        }

    def _process_midi_files_sequential(self, unique_midis, tokenizer_params):
        """Process MIDI files sequentially using simpler cache paths."""
        self.tokens_lookup = {}
        cache_hits = 0
        to_process = set()
        
        # First try to load all from cache
        print("\nChecking existing cache...")
        for midi_path in tqdm(unique_midis, desc="Loading from cache"):
            cache_file = self._get_cache_path(midi_path)
            if cache_file.exists():
                try:
                    tokens = torch.load(cache_file)
                    if tokens is not None and len(tokens) > 0:
                        self.tokens_lookup[midi_path] = tokens
                        cache_hits += 1
                    else:
                        to_process.add(midi_path)
                except Exception as e:
                    print(f"Error loading cache for {midi_path}: {e}")
                    to_process.add(midi_path)
            else:
                to_process.add(midi_path)
        
        print(f"\nCache status:")
        print(f"Found {cache_hits} cached files")
        print(f"Need to process {len(to_process)} files")
        
        # Process only uncached files
        if to_process:
            print("\nProcessing uncached files...")
            processed_count = 0
            error_count = 0
            
            for midi_path in tqdm(to_process, desc="Processing new files"):
                try:
                    tokens = self.tokenizer.encode_midi(midi_path)
                    if tokens is not None and len(tokens) > 0:
                        tokens = torch.tensor(tokens, dtype=torch.long)
                        cache_file = self._get_cache_path(midi_path)
                        torch.save(tokens, cache_file)
                        self.tokens_lookup[midi_path] = tokens
                        processed_count += 1
                except Exception as e:
                    error_count += 1
                    print(f"Error processing {midi_path}: {e}")
                    
            print(f"\nProcessing results:")
            print(f"Successfully processed: {processed_count}")
            print(f"Errors: {error_count}")
        
        if len(self.tokens_lookup) > 0:
            lengths = [len(t) for t in self.tokens_lookup.values()]
            print(f"\nToken statistics:")
            print(f"Min length: {min(lengths)}")
            print(f"Max length: {max(lengths)}")
            print(f"Mean length: {sum(lengths)/len(lengths):.2f}")

    def _load_matches_and_create_pairs(self):
        """Load matches data and create initial pairs"""
        print("\nLoading matches.json...")
        matches_path = self.base_dir / 'matches.json'
        with open(matches_path) as f:
            self.matches_data = json.load(f)

        print("Creating initial pairs...")
        self.all_pairs = []
        self.unique_midis = set()
        midi_not_found = set()

        for artwork_path, match_data in tqdm(self.matches_data['matches'].items(), desc="Processing artworks"):
            artwork_path = artwork_path.replace('data/', '')
            artwork_path = self.base_dir / artwork_path
            style = artwork_path.parent.name

            for midi_match in match_data['midi_matches']:
                midi_path = midi_match['midi_path'].replace('data/unique_midis/', 'midis/')
                midi_path = self.base_dir / midi_path
                
                if not midi_path.exists():
                    midi_not_found.add(str(midi_path))
                    continue
                    
                self.unique_midis.add(str(midi_path))
                self.all_pairs.append({
                    'image_path': artwork_path,
                    'midi_path': str(midi_path),
                    'style': style,
                    'similarity_score': midi_match['similarity_score'],
                    'emotional_match': midi_match['emotional_match'],
                    'artwork_info': match_data['artwork_info']
                })

        print(f"\nPath verification:")
        print(f"Found {len(self.unique_midis)} unique MIDI files")
        if midi_not_found:
            print(f"Warning: {len(midi_not_found)} MIDI files not found")

    def _load_or_process_midi_files(self, tokenizer_params):
        """Load from cache or process only uncached MIDI files"""
        self.tokens_lookup = {}
        cache_hits = 0
        to_process = set()
        
        # First try to load all from cache
        print("\nChecking existing cache...")
        for midi_path in tqdm(self.unique_midis, desc="Loading from cache"):
            cache_file = self._get_cache_path(midi_path)
            if cache_file.exists():
                try:
                    tokens = torch.load(cache_file)
                    if tokens is not None and len(tokens) > 0:
                        self.tokens_lookup[midi_path] = tokens
                        cache_hits += 1
                    else:
                        to_process.add(midi_path)
                except Exception as e:
                    print(f"Error loading cache for {midi_path}: {e}")
                    to_process.add(midi_path)
            else:
                to_process.add(midi_path)
        
        print(f"\nCache status:")
        print(f"Found {cache_hits} cached files")
        print(f"Need to process {len(to_process)} files")
        
        # Process only uncached files
        if to_process:
            print("\nProcessing uncached files...")
            processed_count = 0
            error_count = 0
            
            for midi_path in tqdm(to_process, desc="Processing new files"):
                try:
                    tokens = self.tokenizer.encode_midi(midi_path)
                    if tokens is not None and len(tokens) > 0:
                        tokens = torch.tensor(tokens, dtype=torch.long)
                        cache_file = self._get_cache_path(midi_path)
                        torch.save(tokens, cache_file)
                        self.tokens_lookup[midi_path] = tokens
                        processed_count += 1
                except Exception as e:
                    error_count += 1
                    print(f"Error processing {midi_path}: {e}")
            
            print(f"\nProcessing results:")
            print(f"Successfully processed: {processed_count}")
            print(f"Errors: {error_count}")
        
        print(f"Total tokens in lookup: {len(self.tokens_lookup)}")

    def _split_dataset(self, split: str, val_size: float, test_size: float, seed: int):
        """Split dataset into train/val/test sets."""
        random.seed(seed)
        
        # Create list of indices
        indices = list(range(len(self.all_pairs)))
        random.shuffle(indices)
        
        # Calculate split points
        total_size = len(indices)
        test_idx = int(total_size * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))
        
        # Select appropriate indices based on split
        if split == 'train':
            selected_indices = indices[:val_idx]
        elif split == 'val':
            selected_indices = indices[val_idx:test_idx]
        else:  # test
            selected_indices = indices[test_idx:]
        
        # Filter pairs to only include selected indices
        self.valid_pairs = [self.all_pairs[i] for i in selected_indices]
        
        print(f"\nDataset split ({split}):")
        print(f"Total pairs: {total_size}")
        print(f"Selected pairs: {len(self.valid_pairs)}")
        
        # Additional information about the split
        styles = {}
        for pair in self.valid_pairs:
            style = pair['style']
            styles[style] = styles.get(style, 0) + 1
        
        print("\nStyle distribution:")
        for style, count in sorted(styles.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"{style}: {count} ({count/len(self.valid_pairs)*100:.1f}%)")

    def _compute_statistics(self):
        """Compute dataset statistics."""
        if not hasattr(self, 'valid_pairs'):
            print("\nWarning: No valid pairs available for statistics!")
            return
            
        if not self.tokens_lookup:
            print("\nWarning: No tokens in lookup dictionary!")
            return
        
        token_lengths = []
        empty_tokens = 0
        
        for pair in self.valid_pairs:
            tokens = self.tokens_lookup.get(str(pair['midi_path']), [])
            if len(tokens) == 0:
                empty_tokens += 1
            else:
                token_lengths.append(len(tokens))
        
        if token_lengths:
            print(f"\nSequence length statistics:")
            print(f"Mean length: {np.mean(token_lengths):.2f}")
            print(f"Max length: {max(token_lengths)}")
            print(f"Min length: {min(token_lengths)}")
            print(f"Sequences > {self.max_seq_length}: {sum(1 for l in token_lengths if l > self.max_seq_length)}")
            if empty_tokens > 0:
                print(f"Warning: {empty_tokens} sequences have no tokens")
        else:
            print("\nWarning: No valid token sequences found!")

    def __len__(self):
        """Return the number of valid pairs."""
        return len(self.valid_pairs)
    
    def _get_cache_path(self, midi_path: str) -> Path:
        """Generate a cache file path from MIDI path to match existing cache."""
        # Just use the stem of the MIDI path for consistency with existing cache
        safe_name = Path(midi_path).stem
        return self.token_cache_dir / f"{safe_name}.pt"

    def _compute_token_types(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute token type IDs for the sequence."""
        token_types = torch.zeros_like(tokens)
        
        # Identify different token types
        for i, token in enumerate(tokens):
            if token >= self.tokenizer.TIME_SHIFT_OFFSET:
                token_types[i] = 3  # Time shift
            elif token >= self.tokenizer.VELOCITY_OFFSET:
                token_types[i] = 2  # Velocity
            elif token >= self.tokenizer.NOTE_OFF_OFFSET:
                token_types[i] = 1  # Note off
            # Note on and special tokens remain 0
            
        return token_types

    
def get_dataset(
    base_dir: str,
    split: str = 'train',
    num_workers: int = 32,
    tokenizer_params: Optional[Dict] = None,
    max_seq_length: int = 1024,
    vit_model_name: str = "google/vit-base-patch16-384",
    max_duration: float = 30.0,
    **kwargs
) -> ImageMIDIDataset:
    """Get dataset split with optional parameters."""
    return ImageMIDIDataset(
        base_dir=base_dir,
        split=split,
        num_workers=num_workers,
        tokenizer_params=tokenizer_params,
        max_seq_length=max_seq_length,
        vit_model_name=vit_model_name,
        max_duration=max_duration,
        **kwargs
    )