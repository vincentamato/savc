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

def key_number_to_string(key_number):
    """Convert key number to human-readable key signature.
    
    Args:
        key_number (int): Key number from MIDI (-7 to +7 for major, 
                         -7-8 to +7+8 for minor where +8 indicates minor)
    
    Returns:
        str: Human-readable key signature (e.g., "C major", "A minor")
    """
    major_keys = ['Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F', 
                  'C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#']
    minor_keys = ['Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D',
                  'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#']
    
    if key_number >= 0:
        if key_number < 8:  # Major
            return f"{major_keys[key_number + 7]} major"
        else:  # Minor
            return f"{minor_keys[key_number - 8 + 7]} minor"
    else:
        if key_number > -8:  # Major
            return f"{major_keys[key_number + 7]} major"
        else:  # Minor
            return f"{minor_keys[key_number + 8 + 7]} minor"

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

def analyze_key_signature(notes):
    """Analyze notes to determine likely key signature when MIDI key signature is missing"""
    # Count occurrence of each pitch class
    pitch_classes = np.zeros(12)
    for note in notes:
        pitch_class = note.pitch % 12
        pitch_classes[pitch_class] += note.end - note.start
    
    # Correlation coefficients for major and minor scales
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Calculate correlations
    max_corr = -1
    best_key = 0
    is_major = True
    
    for i in range(12):
        # Shift pitch class distribution
        shifted = np.roll(pitch_classes, -i)
        
        # Correlation with major profile
        major_corr = np.corrcoef(shifted, major_profile)[0,1]
        
        # Correlation with minor profile
        minor_corr = np.corrcoef(shifted, minor_profile)[0,1]
        
        if major_corr > max_corr:
            max_corr = major_corr
            best_key = i
            is_major = True
            
        if minor_corr > max_corr:
            max_corr = minor_corr
            best_key = i
            is_major = False
    
    # Convert to key number format
    if is_major:
        key_num = best_key - 7 if best_key > 6 else best_key
    else:
        key_num = (best_key + 8) if best_key < 4 else (best_key - 4)
        
    return key_num

def extract_features_from_midi_pretty(midi_data: pretty_midi.PrettyMIDI) -> dict:
    """Extract musical features using pretty_midi with improved key detection"""
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
            time_sig = "4/4"  # Default to 4/4 if no time signature is specified
        
        # Key signature detection
        key_sigs = midi_data.key_signature_changes
        if key_sigs and len(key_sigs) > 0:
            key_number = key_sigs[0].key_number
        else:
            # If no key signature in MIDI, analyze notes
            key_number = analyze_key_signature(all_notes)
        
        # Convert key number to string
        key_name = key_number_to_string(key_number)
        
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

# Rest of the code remains the same...

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