import pretty_midi
from pathlib import Path
from tqdm import tqdm
import shutil
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

def is_piano_only(midi_path):
    """Check if MIDI file contains only piano tracks"""
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Skip files with no instruments
        if not midi_data.instruments:
            return False
        
        # Track if we found any piano tracks
        found_piano = False
        
        # Check all instruments
        for instrument in midi_data.instruments:
            # Skip drum tracks
            if instrument.is_drum:
                continue
            
            # If it's a piano, mark it
            if instrument.program in range(8):
                if len(instrument.notes) > 0:  # Ensure track has notes
                    found_piano = True
            else:
                # If we find any non-piano instrument, return False
                return False
        
        return found_piano
    
    except Exception:
        return False

def process_batch(file_batch, input_dir, output_dir):
    """Process a batch of MIDI files"""
    results = []
    for midi_path in file_batch:
        try:
            if is_piano_only(midi_path):
                relative_path = midi_path.relative_to(input_dir)
                output_path = output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(midi_path), str(output_path))
                results.append(True)
            else:
                results.append(False)
        except Exception as e:
            print(f"Error processing {midi_path}: {str(e)}")
            results.append(False)
    return results

def filter_midi_files(input_dir, output_dir, num_workers=None):
    """Filter MIDI files using multiprocessing"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of all MIDI files
    midi_files = list(input_dir.rglob("*.mid"))
    total_files = len(midi_files)
    
    if not total_files:
        print("No MIDI files found!")
        return
    
    # Determine number of workers
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # Calculate batch size
    batch_size = max(1, total_files // (num_workers * 10))  # 10 batches per worker
    
    # Create batches
    batches = [midi_files[i:i + batch_size] for i in range(0, len(midi_files), batch_size)]
    
    print(f"Processing {total_files} files using {num_workers} workers...")
    print(f"Batch size: {batch_size}, Number of batches: {len(batches)}")
    
    # Process batches in parallel
    piano_count = 0
    process_fn = partial(process_batch, input_dir=input_dir, output_dir=output_dir)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_fn, batch) for batch in batches]
        
        with tqdm(total=total_files, desc="Processing MIDI files") as pbar:
            for future in as_completed(futures):
                results = future.result()
                piano_count += sum(results)
                pbar.update(len(results))
    
    print(f"\nFound {piano_count} piano MIDI files out of {total_files} total files")
    print(f"Piano files percentage: {(piano_count/total_files * 100):.2f}%")
    print(f"Processed files saved to {output_dir}")

def main():
    # Adjust these paths to match your setup
    input_dir = "downloads/clean_midi/clean_midi"  # Path to your MIDI files
    output_dir = "data/lmd"  # Where to save piano-only files
    
    # Optional: specify number of workers (default: number of CPU cores)
    num_workers = None  
    
    filter_midi_files(input_dir, output_dir, num_workers)

if __name__ == "__main__":
    main()