import os
import json
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
from functools import partial
import numpy as np
from dataclasses import dataclass

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
        
        # Token type ranges
        self.NOTE_ON_OFFSET = 0
        self.NOTE_OFF_OFFSET = max_notes
        self.VELOCITY_OFFSET = 2 * max_notes
        self.TIME_SHIFT_OFFSET = 2 * max_notes + max_velocity
        
        # Special tokens
        self.special_tokens = special_tokens or {
            'PAD': 0,
            'BOS': 1,
            'EOS': 2,
            'MASK': 3
        }
        self.special_tokens_inv = {v: k for k, v in self.special_tokens.items()}
        
        self.vocab_size = (
            2 * max_notes +  # Note on/off
            max_velocity +   # Velocity levels
            max_time_shift + # Time shifts
            len(self.special_tokens)
        )
        
        # Statistics tracking
        self.stats = {
            'sequence_lengths': [],
            'unique_tokens': set(),
            'token_frequencies': {},
        }
    
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
            self._update_stats(tokens)
            return tokens
            
        except Exception as e:
            logging.error(f"Error processing MIDI file {midi_path}: {str(e)}")
            return None
    
    def decode_to_midi(self, tokens: Union[List[int], torch.Tensor], output_path: Optional[str] = None) -> Union[pretty_midi.PrettyMIDI, Tuple[pretty_midi.PrettyMIDI, str]]:
        """Decode token sequence back to MIDI file.
        
        Args:
            tokens: List or tensor of tokens
            output_path: Optional path to save the MIDI file
            
        Returns:
            If output_path is None: Returns PrettyMIDI object
            If output_path is provided: Returns tuple of (PrettyMIDI object, output_path)
        """
        try:
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.cpu().tolist()
            
            midi_data = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)  # Piano by default
            
            current_time = 0.0
            current_velocity = 64  # Default velocity
            active_notes = {}  # pitch -> start_time
            
            for token in tokens:
                # Skip special tokens
                if token in self.special_tokens.values():
                    continue
                
                # Process token based on type
                if self.NOTE_ON_OFFSET <= token < self.NOTE_OFF_OFFSET:
                    # Note on event
                    pitch = token - self.NOTE_ON_OFFSET
                    active_notes[pitch] = current_time
                
                elif self.NOTE_OFF_OFFSET <= token < self.VELOCITY_OFFSET:
                    # Note off event
                    pitch = token - self.NOTE_OFF_OFFSET
                    if pitch in active_notes:
                        start_time = active_notes[pitch]
                        note = pretty_midi.Note(
                            velocity=current_velocity,
                            pitch=pitch,
                            start=start_time,
                            end=current_time
                        )
                        instrument.notes.append(note)
                        del active_notes[pitch]
                
                elif self.VELOCITY_OFFSET <= token < self.TIME_SHIFT_OFFSET:
                    # Velocity change
                    current_velocity = (token - self.VELOCITY_OFFSET) * (128 // self.max_velocity)
                
                elif self.TIME_SHIFT_OFFSET <= token < self.vocab_size:
                    # Time shift
                    time_steps = token - self.TIME_SHIFT_OFFSET
                    current_time += time_steps * self.time_step
            
            # Close any still-active notes
            for pitch, start_time in active_notes.items():
                note = pretty_midi.Note(
                    velocity=current_velocity,
                    pitch=pitch,
                    start=start_time,
                    end=current_time
                )
                instrument.notes.append(note)
            
            midi_data.instruments.append(instrument)
            
            if output_path:
                midi_data.write(output_path)
                return midi_data, output_path
            
            return midi_data
            
        except Exception as e:
            logging.error(f"Error decoding tokens to MIDI: {str(e)}")
            return None
        
def tokenize_midi_file(args: Tuple[str, str, Dict]) -> Optional[Dict]:
    """Tokenize a single MIDI file with caching."""
    midi_path, cache_path, tokenizer_params = args
    cache_file = Path(cache_path) / f"{Path(midi_path).stem}.pkl"
    
    # Check cache first
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return {'path': midi_path, 'tokens': pickle.load(f)}
        except Exception:
            pass
    
    # Tokenize
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
        num_workers: int = 32,
        max_duration: float = 30.0
    ):
        self.base_dir = Path(base_dir)
        self.max_seq_length = max_seq_length
        print(f"Initializing dataset with max_seq_length: {max_seq_length}")
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
        
        # Build initial pairs
        print("Creating initial pairs...")
        self.all_pairs = []
        unique_midis = set()
        
        for artwork_path, match_data in tqdm(self.matches_data['matches'].items(), 
                                           desc="Processing artworks"):
            artwork_path = artwork_path.replace('data/', '')
            artwork_path = self.base_dir / artwork_path
            
            style = artwork_path.parent.name
            
            for midi_match in match_data['midi_matches']:
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
        print("\nTokenizing MIDI files in parallel...")
        tokenizer_params = {
            'max_notes': 128,
            'max_velocity': 32,
            'time_step': 0.125,
            'max_time_shift': 100,
            'max_duration': max_duration
        }
        
        with Pool(num_workers) as pool:
            midi_args = [(str(path), str(self.cache_dir), tokenizer_params) 
                        for path in unique_midis]
            tokens_data = list(tqdm(
                pool.imap(tokenize_midi_file, midi_args),
                total=len(midi_args),
                desc="Tokenizing MIDIs"
            ))
        
        # Create tokens lookup dictionary
        self.tokens_lookup = {
            item['path']: torch.LongTensor(item['tokens'][:self.max_seq_length]).pin_memory()
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

        token_lengths = [len(pair['tokens']) for pair in self.valid_pairs]
        print("\nToken sequence statistics:")
        print(f"Mean length: {np.mean(token_lengths):.2f}")
        print(f"Max length: {max(token_lengths)}")
        print(f"Min length: {min(token_lengths)}")
        print(f"Number of sequences > {max_seq_length}: {sum(1 for l in token_lengths if l > max_seq_length)}")
    
    def __len__(self) -> int:
        return len(self.valid_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.valid_pairs[idx]
        
        # Load and transform image
        image = Image.open(pair['image_path']).convert('RGB')
        image_tensor = self.transform(image)
        
        # Get tokens and convert to tensor first
        tokens = torch.tensor(pair['tokens'][:self.max_seq_length])
        
        # Pad if necessary using torch operations
        if len(tokens) < self.max_seq_length:
            padding = torch.zeros(self.max_seq_length - len(tokens), dtype=torch.long)
            tokens = torch.cat([tokens, padding])

        return {
            'image': image_tensor,
            'tokens': tokens,
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
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2
    )

def get_dataset(
    base_dir: str,
    split: str = 'train',
    num_workers: int = 32,
    **dataset_kwargs
) -> ImageMIDIDataset:
    """Get dataset split directly without creating multiple instances.
    
    Args:
        base_dir (str): Base directory containing the dataset
        split (str): Dataset split ('train', 'val', or 'test')
        num_workers (int): Number of workers for parallel processing
        **dataset_kwargs: Additional arguments for ImageMIDIDataset
    
    Returns:
        ImageMIDIDataset: The dataset instance for the specified split
    """
    return ImageMIDIDataset(
        base_dir=base_dir,
        split=split,
        num_workers=num_workers,
        **dataset_kwargs
    )

if __name__ == "__main__":
    # Example usage and dataset analysis
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze dataset and create samples')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--max_duration', type=float, default=30.0, help='Maximum MIDI duration in seconds')
    parser.add_argument('--test_decode', action='store_true', help='Test MIDI decoding')
    args = parser.parse_args()
    
    # Create dataset instance
    dataset = get_dataset(
        base_dir=args.data_dir,
        split='train',
        max_duration=args.max_duration
    )
    
    # Print memory usage
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Create and test dataloader
    dataloader = create_dataloader(dataset, batch_size=4)
    
    print("\nTesting dataloader...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == 0:
            print(f"Batch size: {batch['image'].size()}")
            print(f"Token sequence length: {batch['tokens'].size()}")
            print(f"Number of unique tokens: {batch['tokens'].unique().size(0)}")
            
            # Test decoding if requested
            if args.test_decode:
                print("\nTesting MIDI decoding...")
                tokenizer = MIDITokenizer(max_duration=args.max_duration)
                midi_data, output_path = tokenizer.decode_to_midi(
                    batch['tokens'][0],
                    output_path="test_decoded.mid"
                )
                if midi_data:
                    print("Successfully decoded tokens to MIDI!")
                    print(f"Duration: {midi_data.get_end_time():.2f} seconds")
                    print(f"Number of notes: {sum(len(inst.notes) for inst in midi_data.instruments)}")
                    print(f"Saved to: {output_path}")
            break