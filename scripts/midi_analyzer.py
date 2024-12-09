import pretty_midi
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import multiprocessing as mp
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

@dataclass
class MIDIFileInfo:
    path: Path
    year: Optional[str] = None
    piece_id: Optional[str] = None
    track: Optional[str] = None
    raw_name: Optional[str] = None

class MIDIDatasetAnalyzer:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
    
    def parse_maestro_filename(self, path: Path) -> MIDIFileInfo:
        """Parse MAESTRO dataset filename structure"""
        filename = path.name
        info = MIDIFileInfo(path=path, raw_name=filename)
        
        # Parse MIDI-Unprocessed_SMF_XX_RX_YYYY... format
        pattern = r"MIDI-Unprocessed_SMF_(\d+).*?_(\d{4}).*?Track(\d+)"
        match = re.search(pattern, filename)
        
        if match:
            info.piece_id = match.group(1)
            info.year = match.group(2)
            info.track = match.group(3)
            
        return info

    def analyze_midi_file(self, file_info: MIDIFileInfo) -> dict:
        """Analyze a single MIDI file with enhanced metadata"""
        try:
            midi_data = pretty_midi.PrettyMIDI(str(file_info.path))
            
            # Collect all notes from all instruments
            all_notes = []
            instruments_info = []
            
            for inst in midi_data.instruments:
                if not inst.is_drum:
                    all_notes.extend(inst.notes)
                    instruments_info.append({
                        'program': inst.program,
                        'name': pretty_midi.program_to_instrument_name(inst.program),
                        'is_drum': inst.is_drum,
                        'note_count': len(inst.notes)
                    })
            
            if not all_notes:
                return {
                    'valid': False, 
                    'error': 'No notes found',
                    'metadata': vars(file_info)
                }
            
            stats = {
                'valid': True,
                'metadata': vars(file_info),
                'duration': midi_data.get_end_time(),
                'num_notes': len(all_notes),
                'tempo': float(np.mean(midi_data.get_tempo_changes()[1])),
                'num_instruments': len(midi_data.instruments),
                'notes_per_second': len(all_notes) / midi_data.get_end_time(),
                'pitch_range': (
                    min(note.pitch for note in all_notes),
                    max(note.pitch for note in all_notes)
                ),
                'velocity_range': (
                    min(note.velocity for note in all_notes),
                    max(note.velocity for note in all_notes)
                ),
                'time_signature': f"{midi_data.time_signature_changes[0].numerator}/{midi_data.time_signature_changes[0].denominator}" if midi_data.time_signature_changes else "None",
                'instruments': instruments_info,
                'avg_velocity': np.mean([note.velocity for note in all_notes]),
                'avg_note_duration': np.mean([note.end - note.start for note in all_notes])
            }
            return stats
            
        except Exception as e:
            return {
                'valid': False, 
                'error': str(e),
                'metadata': vars(file_info)
            }

    def process_batch(self, files: List[MIDIFileInfo]) -> List[dict]:
        """Process a batch of MIDI files"""
        return [self.analyze_midi_file(file_info) for file_info in files]

    def analyze_dataset(self, dataset_path: str, dataset_type: str = 'generic', num_workers: int = None):
        """Analyze an entire dataset with type-specific processing"""
        print(f"\nAnalyzing {dataset_type} dataset at {dataset_path}...")
        
        # Find all MIDI files
        midi_files = list(Path(dataset_path).rglob("*.mid*"))
        total_files = len(midi_files)
        
        if not total_files:
            print(f"No MIDI files found in {dataset_path}")
            return None
            
        print(f"Found {total_files} MIDI files")
        
        # Parse filenames based on dataset type
        file_infos = []
        for path in midi_files:
            if dataset_type.lower() == 'maestro':
                file_info = self.parse_maestro_filename(path)
            else:
                file_info = MIDIFileInfo(path=path, raw_name=path.name)
            file_infos.append(file_info)
        
        # Create batches
        batches = [file_infos[i:i + self.batch_size] 
                  for i in range(0, len(file_infos), self.batch_size)]
        
        # Process files in parallel
        num_workers = num_workers or mp.cpu_count()
        all_results = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.process_batch, batch) 
                      for batch in batches]
            
            with tqdm(total=total_files, desc="Processing files") as pbar:
                for future in futures:
                    results = future.result()
                    all_results.extend(results)
                    pbar.update(len(results))
        
        return self.compute_statistics(all_results, dataset_type)

    def compute_statistics(self, results: List[dict], dataset_type: str) -> Tuple[dict, pd.DataFrame]:
        """Compute comprehensive statistics from analysis results"""
        valid_results = [r for r in results if r['valid']]
        
        if not valid_results:
            print("No valid MIDI files found!")
            return None
        
        stats = {
            'dataset_type': dataset_type,
            'total_files': len(results),
            'valid_files': len(valid_results),
            'invalid_files': len(results) - len(valid_results),
            'duration_stats': {
                'mean': np.mean([r['duration'] for r in valid_results]),
                'std': np.std([r['duration'] for r in valid_results]),
                'min': np.min([r['duration'] for r in valid_results]),
                'max': np.max([r['duration'] for r in valid_results])
            },
            'note_stats': {
                'mean': np.mean([r['num_notes'] for r in valid_results]),
                'std': np.std([r['num_notes'] for r in valid_results]),
                'min': np.min([r['num_notes'] for r in valid_results]),
                'max': np.max([r['num_notes'] for r in valid_results])
            }
        }
        
        # Create detailed DataFrame
        df = pd.DataFrame(valid_results)
        
        # Add year-based statistics for MAESTRO
        if dataset_type.lower() == 'maestro':
            years = [r['metadata']['year'] for r in valid_results if r['metadata']['year']]
            if years:
                stats['year_distribution'] = pd.Series(years).value_counts().to_dict()
        
        return stats, df

def main():
    analyzer = MIDIDatasetAnalyzer(batch_size=50)
    
    datasets = {
        'MAESTRO': ('data/maestro', 'maestro'),
        'Giant_Piano': ('data/giant_piano', 'generic'),
        'LMD': ('data/lmd', 'generic')
    }
    
    for name, (path, dtype) in datasets.items():
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist, skipping {name}")
            continue
        
        results = analyzer.analyze_dataset(path, dtype)
        if results:
            stats, df = results
            
            # Save detailed results
            output_dir = Path('analysis_results')
            output_dir.mkdir(exist_ok=True)
            
            df.to_csv(output_dir / f"{name.lower()}_analysis.csv", index=False)
            
            # Print summary
            print(f"\n=== {name} Dataset Summary ===")
            print(f"Total files: {stats['total_files']}")
            print(f"Valid files: {stats['valid_files']} ({stats['valid_files']/stats['total_files']*100:.1f}%)")
            print(f"\nDuration (seconds):")
            print(f"  Mean ± std: {stats['duration_stats']['mean']:.1f} ± {stats['duration_stats']['std']:.1f}")
            print(f"  Range: {stats['duration_stats']['min']:.1f} - {stats['duration_stats']['max']:.1f}")
            
            if 'year_distribution' in stats:
                print("\nYear distribution:")
                for year, count in sorted(stats['year_distribution'].items()):
                    print(f"  {year}: {count}")

if __name__ == "__main__":
    main()