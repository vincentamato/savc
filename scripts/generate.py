#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import torch
from torch import nn
from PIL import Image
import pretty_midi
import logging
from accelerate import Accelerator

from src.models.music_transformer import MusicTransformer
from src.data.dataset import MIDITokenizer
from transformers import ViTImageProcessor

def parse_args():
    parser = argparse.ArgumentParser(description='Generate MIDI from Art using MusicTransformer')
    
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image file')
    parser.add_argument('--output_midi', type=str, required=True,
                        help='Path to save the generated MIDI file')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/good-deluge-33/',
                        help='Path to the model checkpoint directory')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature for generation')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-K sampling for generation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model on (cuda or cpu)')
    
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def tokens_to_midi(tokens, tokenizer, logger=None):
    """Convert token sequence to MIDI with proper note handling"""
    if logger:
        logger.info(f"Starting MIDI conversion of {len(tokens)} tokens")
    
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    piano = pretty_midi.Instrument(program=0)
    
    current_time = 0.0
    current_velocity = 64
    active_notes = {}  # Dictionary to track active notes: {note_num: (start_time, velocity)}
    completed_notes = []  # List to track completed notes before adding to MIDI
    note_count = 0
    time_shift_count = 0
    
    if logger:
        logger.info(f"Initial state - Time: {current_time:.2f}, Velocity: {current_velocity}")
    
    for i, token in enumerate(tokens):
        if i % 50 == 0 and logger:
            logger.info(f"Processing token {i}/{len(tokens)} at time {current_time:.2f}s")
        
        token = token.item()
        
        # Skip special tokens
        if token in tokenizer.special_tokens.values():
            if logger:
                logger.info(f"Special token encountered: {token}")
            continue
        
        # Note on event
        if tokenizer.NOTE_ON_OFFSET <= token < tokenizer.NOTE_OFF_OFFSET:
            note_num = token - tokenizer.NOTE_ON_OFFSET
            if note_num not in active_notes:  # Prevent duplicate note-on events
                active_notes[note_num] = (current_time, current_velocity)
                note_count += 1
                if logger:
                    logger.info(f"Note ON: {note_num} at time {current_time:.2f}s with velocity {current_velocity}")
        
        # Note off event
        elif tokenizer.NOTE_OFF_OFFSET <= token < tokenizer.VELOCITY_OFFSET:
            note_num = token - tokenizer.NOTE_OFF_OFFSET
            if note_num in active_notes:
                start_time, velocity = active_notes[note_num]
                if current_time > start_time:  # Ensure positive duration
                    completed_notes.append((note_num, start_time, current_time, velocity))
                    if logger:
                        logger.info(f"Note OFF: {note_num} at time {current_time:.2f}s, duration: {current_time - start_time:.2f}s")
                del active_notes[note_num]
        
        # Velocity event
        elif tokenizer.VELOCITY_OFFSET <= token < tokenizer.TIME_SHIFT_OFFSET:
            velocity_idx = token - tokenizer.VELOCITY_OFFSET
            current_velocity = int((velocity_idx / tokenizer.max_velocity) * 127)
            if logger:
                logger.info(f"Velocity change: {current_velocity}")
        
        # Time shift event
        elif token >= tokenizer.TIME_SHIFT_OFFSET:
            time_steps = token - tokenizer.TIME_SHIFT_OFFSET
            time_shift = time_steps * tokenizer.time_step
            current_time += time_shift
            time_shift_count += 1
            if logger:
                logger.info(f"Time shift: +{time_shift:.3f}s, new time: {current_time:.3f}s")
    
    # Close any remaining active notes at the final time
    for note_num, (start_time, velocity) in active_notes.items():
        if current_time > start_time:  # Ensure positive duration
            completed_notes.append((note_num, start_time, current_time, velocity))
            if logger:
                logger.info(f"Closing remaining note: {note_num} at time {current_time:.2f}s")
    
    # Add all completed notes to the MIDI track
    for note_num, start_time, end_time, velocity in completed_notes:
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=note_num,
            start=start_time,
            end=end_time
        )
        piano.notes.append(note)
    
    midi.instruments.append(piano)
    
    if logger:
        logger.info("MIDI conversion complete:")
        logger.info(f"- Total duration: {current_time:.2f} seconds")
        logger.info(f"- Total notes created: {note_count}")
        logger.info(f"- Time shift events: {time_shift_count}")
        logger.info(f"- Final note count in MIDI: {len(piano.notes)}")
        logger.info("- Velocity changes recorded and applied")
    
    return midi

def generate_note_off_sequence(self, active_notes, current_time):
    """Helper to generate note-off events for active notes"""
    sequence = []
    for note_num in list(active_notes.keys()):
        if current_time - active_notes[note_num] >= 0.125:  # Minimum note duration
            sequence.append(self.tokenizer.NOTE_OFF_OFFSET + note_num)
    return sequence

def enforce_note_closure(self, logits, active_notes, current_time, last_event_type):
    """Enforce note closure for long-held notes"""
    for note_num, start_time in active_notes.items():
        if current_time - start_time > 4.0:  # Force closure for notes held too long
            note_off_token = self.tokenizer.NOTE_OFF_OFFSET + note_num
            logits[0, 0, note_off_token] *= 2.0  # Increase probability of closing
    return logits

def main():
    args = parse_args()
    logger = setup_logging()

    # Initialize accelerator
    accelerator = Accelerator()

    # Validate input image path
    image_path = Path(args.image_path)
    if not image_path.exists():
        logger.error(f"Input image not found: {image_path}")
        return

    # Initialize tokenizer with the same parameters used during training
    tokenizer = MIDITokenizer(
        max_notes=128,
        max_velocity=32,
        time_step=0.125,
        max_time_shift=100,
        max_duration=30.0,
        special_tokens={
            'PAD': 0,
            'BOS': 1,
            'EOS': 2
        }
    )

    # Calculate vocab size
    vocab_size = (
        len(tokenizer.special_tokens) +  # Special tokens
        2 * tokenizer.max_notes +        # Note on/off events
        tokenizer.max_velocity +         # Velocity levels
        tokenizer.max_time_shift         # Time shift tokens
    )

    # Initialize model
    model = MusicTransformer(
        vocab_size=vocab_size,          # Add this back
        d_model=1024,
        nhead=16,
        num_decoder_layers=16,
        dim_feedforward=4096,
        dropout=0.15,
        max_seq_length=1024,
        tokenizer=tokenizer,
        vit_model="google/vit-base-patch16-384",
        freeze_vit=True
    )

    # Load checkpoint using Accelerator
    checkpoint_dir = Path(args.checkpoint_dir) / 'best_model.pt'
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return

    try:
        model = accelerator.prepare(model)
        accelerator.load_state(checkpoint_dir)
        logger.info(f"Loaded model weights from {checkpoint_dir}")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        return

    model.eval()

    # Initialize image processor
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-384")

    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return

    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(args.device)

    # Generate tokens
    try:
        logger.info("Starting token generation...")
        logger.info(f"Model parameters: temperature={args.temperature}, top_k={args.top_k}")
        
        with torch.no_grad():
            logger.info("Processing image through ViT...")
            generated_tokens = model.generate(
                visual_features=pixel_values,
                max_length=1024,
                temperature=args.temperature,
                top_k=args.top_k,
                logger=logger  # Pass logger to the generate method
            )
        
        logger.info(f"Generation complete. Generated {len(generated_tokens[0])} tokens")
    except Exception as e:
        logger.error(f"Failed during generation: {e}")
        return
    
    # Convert generated tokens to MIDI
    output_midi_path = Path(args.output_midi)
    try:
        logger.info("Converting tokens to MIDI...")
        midi_data = tokens_to_midi(generated_tokens[0], tokenizer, logger)
        output_midi_path.parent.mkdir(parents=True, exist_ok=True)
        midi_data.write(str(output_midi_path))
        logger.info(f"MIDI file saved to {output_midi_path}")
    except Exception as e:
        logger.error(f"Failed to decode tokens to MIDI: {e}")
        return

if __name__ == "__main__":
    main()