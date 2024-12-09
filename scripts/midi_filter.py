import pretty_midi
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import warnings
import numpy as np
import logging
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_piano_only(midi_data: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    """Create a new MIDI file containing only piano tracks from the original"""
    new_midi = pretty_midi.PrettyMIDI(initial_tempo=midi_data.estimate_tempo())
    
    # Copy time signature and key signature changes
    new_midi._time_signature_changes = midi_data.time_signature_changes
    new_midi._key_signature_changes = midi_data.key_signature_changes
    
    for instrument in midi_data.instruments:
        # Check if it's a piano (program numbers 0-7)
        if not instrument.is_drum and 0 <= instrument.program <= 7:
            # Create a new piano instrument
            piano = pretty_midi.Instrument(program=instrument.program)
            piano.notes = instrument.notes
            new_midi.instruments.append(piano)
    
    return new_midi

def analyze_and_extract_midi(midi_path: str, output_dir: Path) -> dict:
    """Analyze a MIDI file and extract piano tracks if present"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Get piano notes and info
        piano_notes = []
        original_instruments = 0
        has_piano = False
        
        for inst in midi_data.instruments:
            if not inst.is_drum:
                original_instruments += 1
                if 0 <= inst.program <= 7:
                    has_piano = True
                    piano_notes.extend(inst.notes)
        
        if not piano_notes:  # Skip if no piano notes found
            return {'path': str(midi_path), 'filter_reason': 'no_piano_notes'}
            
        # Extract piano tracks and save new MIDI
        piano_midi = extract_piano_only(midi_data)
        duration = piano_midi.get_end_time()
        
        # Create output directory structure
        rel_path = Path(midi_path).relative_to(Path(midi_path).parts[0])
        new_midi_path = output_dir / rel_path
        new_midi_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the piano-only MIDI
        piano_midi.write(str(new_midi_path))
            
        return {
            'original_path': str(midi_path),
            'extracted_path': str(new_midi_path),
            'dataset': Path(midi_path).parts[-3] if 'data' in Path(midi_path).parts else 'unknown',
            'filename': Path(midi_path).name,
            'duration': duration,
            'total_notes': len(piano_notes),
            'notes_per_second': len(piano_notes) / duration if duration > 0 else 0,
            'original_instrument_count': original_instruments,
            'had_non_piano': original_instruments > 1,
            'min_pitch': min(note.pitch for note in piano_notes),
            'max_pitch': max(note.pitch for note in piano_notes),
            'mean_velocity': float(np.mean([note.velocity for note in piano_notes]))
        }
        
    except Exception as e:
        return {'path': str(midi_path), 'filter_reason': f'error: {str(e)}'}

def process_batch(args):
    """Process a batch of MIDI files"""
    file_batch, output_dir = args
    return [analyze_and_extract_midi(midi_path, output_dir) for midi_path in file_batch]

def main():
    logger.info("Starting piano track extraction...")
    
    # Define dataset directories
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
    
    midi_files = [str(f) for d in datasets.values() for f in Path(d).rglob("*.mid*")]
    total_files = len(midi_files)
    
    logger.info(f"Found {total_files} total MIDI files")
    
    # Process files
    batch_size = 20
    num_workers = mp.cpu_count()
    batches = [(midi_files[i:i + batch_size], output_dir) 
              for i in range(0, len(midi_files), batch_size)]
    
    all_results = []
    logger.info(f"Processing with {num_workers} workers and batch size of {batch_size}")
    
    with tqdm(total=total_files, desc="Extracting piano tracks") as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for batch_results in executor.map(process_batch, batches):
                all_results.extend(batch_results)
                pbar.update(batch_size)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Print diagnostics
    print("\nProcessing Results:")
    
    if 'filter_reason' in df.columns:
        print("\nFiles filtered out:")
        print(df['filter_reason'].value_counts())
        # Remove filtered files for further analysis
        df = df[~df['filter_reason'].notna()]
    
    print(f"\nSuccessfully processed files: {len(df)}")
    print(f"Files that had non-piano instruments: {df['had_non_piano'].sum()}")
    
    # Save metadata
    df.to_csv(output_dir / 'metadata.csv', index=False)
    
    # Print duration distribution
    print("\nDuration distribution (seconds):")
    print(df['duration'].describe())
    
    # Print notes distribution
    print("\nNotes per second distribution:")
    print(df['notes_per_second'].describe())
    
    logger.info("Extraction complete")

if __name__ == "__main__":
    main()