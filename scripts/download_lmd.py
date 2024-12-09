import requests
import pretty_midi
import os
from pathlib import Path
from tqdm import tqdm
import tarfile
import shutil

def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=str(Path(destination).name),
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def is_piano_only(midi_path):
    """Check if MIDI file contains only piano tracks"""
    try:
        # Convert Path to string for compatibility
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Track if we found any piano tracks
        found_piano = False
        
        # Check all instruments
        for instrument in midi_data.instruments:
            # Skip drum tracks
            if instrument.is_drum:
                continue
            
            # If it's a piano, mark it
            if instrument.program in range(8):
                found_piano = True
            else:
                # If we find any non-piano instrument, return False
                return False
        
        # Return True only if we found at least one piano track
        return found_piano and len(midi_data.instruments) > 0
    
    except Exception as e:
        print(f"Error processing {midi_path}: {str(e)}")
        return False

def process_midi_dataset(download_dir, output_dir):
    """Download and process the Clean MIDI subset"""
    # Create directories
    download_dir = Path(download_dir)
    output_dir = Path(output_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download Clean MIDI subset
    clean_midi_url = "http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz"
    tar_path = download_dir / "clean_midi.tar.gz"
    
    if not tar_path.exists():
        print("Downloading Clean MIDI subset...")
        download_file(clean_midi_url, str(tar_path))
    
    # Extract tar.gz file
    extract_path = download_dir / "clean_midi"
    if not extract_path.exists():
        print("Extracting TAR file...")
        with tarfile.open(str(tar_path), 'r:gz') as tar_ref:
            # Extract to the target directory
            tar_ref.extractall(str(extract_path))
    
    # # Process MIDI files
    # print("Filtering for piano-only MIDI files...")
    # midi_files = list(extract_path.rglob("*.mid"))
    
    # piano_count = 0
    # total_files = len(midi_files)
    
    # for midi_path in tqdm(midi_files, desc="Processing MIDI files"):
    #     try:
    #         if is_piano_only(midi_path):
    #             piano_count += 1
    #             # Preserve directory structure
    #             relative_path = midi_path.relative_to(extract_path)
    #             output_path = output_dir / relative_path
    #             output_path.parent.mkdir(parents=True, exist_ok=True)
    #             shutil.copy2(str(midi_path), str(output_path))
                
    #             if piano_count % 100 == 0:
    #                 print(f"\nFound {piano_count} piano files so far...")
    #     except Exception as e:
    #         print(f"Error copying {midi_path}: {str(e)}")
    #         continue
    
    # print(f"\nFound {piano_count} piano MIDI files out of {total_files} total files")
    # print(f"Piano files percentage: {(piano_count/total_files * 100):.2f}%")
    
    # # Cleanup
    # if tar_path.exists():
    #     tar_path.unlink()
    # if extract_path.exists():
    #     shutil.rmtree(extract_path)
    
    # print(f"Processed files saved to {output_dir}")

def main():
    download_dir = "downloads"
    output_dir = "data/lmd"
    process_midi_dataset(download_dir, output_dir)

if __name__ == "__main__":
    main()