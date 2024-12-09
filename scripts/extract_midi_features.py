import pretty_midi
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import warnings
import logging
import shutil
import os
import csv
from queue import Queue
from threading import Lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_piano_only(midi_data: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    """Create a new MIDI file containing only piano tracks from the original"""
    new_midi = pretty_midi.PrettyMIDI(initial_tempo=midi_data.estimate_tempo())
    new_midi._time_signature_changes = midi_data.time_signature_changes
    new_midi._key_signature_changes = midi_data.key_signature_changes
    
    for instrument in midi_data.instruments:
        if not instrument.is_drum and 0 <= instrument.program <= 7:
            piano = pretty_midi.Instrument(program=instrument.program)
            piano.notes = instrument.notes
            new_midi.instruments.append(piano)
    
    return new_midi

def extract_features_from_midi_pretty(midi_data: pretty_midi.PrettyMIDI) -> dict:
    """Extract musical features using pretty_midi"""
    try:
        # Get all piano notes
        all_notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum and 0 <= instrument.program <= 7:
                all_notes.extend(instrument.notes)
        
        if not all_notes:
            return None
            
        # Calculate features
        pitches = [note.pitch for note in all_notes]
        velocities = [note.velocity for note in all_notes]
        durations = [note.end - note.start for note in all_notes]
        
        # Time signature - improved extraction
        time_sigs = midi_data.time_signature_changes
        if time_sigs:
            time_sig = f"{time_sigs[0].numerator}/{time_sigs[0].denominator}"
        else:
            # Default to 4/4 if no time signature is specified (common in MIDI files)
            time_sig = "4/4"
        
        # Key signature
        key_sigs = midi_data.key_signature_changes
        if key_sigs:
            key_number = key_sigs[0].key_number
            # Convert key number to human-readable format
            keys = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb']
            if -7 <= key_number <= 7:
                key_name = keys[key_number + 7]
            else:
                key_name = 'Unknown'
        else:
            key_number = 0
            key_name = 'C'  # Default to C if no key signature found
        
        return {
            "total_notes": len(all_notes),
            "avg_pitch": float(np.mean(pitches)),
            "min_pitch": min(pitches),
            "max_pitch": max(pitches),
            "pitch_range": max(pitches) - min(pitches),
            "avg_velocity": float(np.mean(velocities)),
            "avg_duration": float(np.mean(durations)),
            "total_duration": float(midi_data.get_end_time()),
            "tempo": float(midi_data.estimate_tempo()),
            "time_signature": time_sig,
            "key": key_name,
            "key_number": key_number,
            "notes_per_second": len(all_notes) / float(midi_data.get_end_time()) if midi_data.get_end_time() > 0 else 0
        }
    except Exception as e:
        logger.debug(f"Error extracting features: {str(e)}")
        return None

def process_midi(args):
    """Process a single MIDI file - extract piano and features"""
    midi_path, output_dir = args
    try:
        # First extract piano
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Check for piano presence
        has_piano = False
        for inst in midi_data.instruments:
            if not inst.is_drum and 0 <= inst.program <= 7:
                has_piano = True
                break
        
        if not has_piano:
            return None
            
        # Extract piano and save
        piano_midi = extract_piano_only(midi_data)
        rel_path = Path(midi_path).relative_to(Path(midi_path).parts[0])
        new_midi_path = output_dir / rel_path
        new_midi_path.parent.mkdir(parents=True, exist_ok=True)
        piano_midi.write(str(new_midi_path))
        
        # Extract features using pretty_midi
        features = extract_features_from_midi_pretty(piano_midi)
        if features:
            features['original_path'] = str(midi_path)
            features['extracted_path'] = str(new_midi_path)
        
        return features
        
    except Exception as e:
        logger.debug(f"Error processing {midi_path}: {str(e)}")
        return None

def save_batch_to_csv(results, output_csv, fieldnames=None):
    """Save a batch of results to CSV"""
    if not results:
        return
    
    # Get fieldnames from first result if not provided
    if fieldnames is None:
        fieldnames = list(results[0].keys())
    
    # Check if file exists to determine if we need to write header
    file_exists = output_csv.exists()
    
    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for result in results:
            writer.writerow(result)

def main():
    logger.info("Starting MIDI processing...")
    
    # Define directories
    datasets = {
        'maestro': 'data/maestro',
        'giant_piano': 'data/giant_piano',
        'lmd': 'data/lmd'
    }
    
    # Set up output directory
    output_dir = Path('data/piano_extracted')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    # Set up CSV output
    output_csv = Path('data/midi_features.csv')
    if output_csv.exists():
        output_csv.unlink()
    
    # Find all MIDI files
    midi_files = [(str(f), output_dir) for d in datasets.values() 
                  for f in Path(d).rglob("*.mid*")]
    total_files = len(midi_files)
    
    logger.info(f"Found {total_files} total MIDI files")
    
    # Process files with multiprocessing
    results_buffer = []
    total_processed = 0
    fieldnames = None
    
    with tqdm(total=total_files, desc="Processing MIDI files") as pbar:
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            for result in executor.map(process_midi, midi_files):
                if result is not None:
                    results_buffer.append(result)
                    
                    # Save every 100 successful results
                    if len(results_buffer) >= 100:
                        if fieldnames is None and results_buffer:
                            fieldnames = list(results_buffer[0].keys())
                        save_batch_to_csv(results_buffer, output_csv, fieldnames)
                        total_processed += len(results_buffer)
                        results_buffer = []
                        
                pbar.update(1)
    
    # Save any remaining results
    if results_buffer:
        save_batch_to_csv(results_buffer, output_csv, fieldnames)
        total_processed += len(results_buffer)
    
    # Print summary
    print(f"\nProcessing complete:")
    print(f"Total files processed: {total_files}")
    print(f"Successfully extracted features: {total_processed}")
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()