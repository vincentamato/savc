import os
import time
import pandas as pd
from pathlib import Path

def monitor_progress(csv_path='data/midi_features.csv', total_files=9469, interval=1):
    """
    Monitor the progress of MIDI processing by checking the CSV file growth.
    
    Args:
        csv_path: Path to the features CSV file
        total_files: Total number of files being processed
        interval: How often to check progress (in seconds)
    """
    csv_path = Path(csv_path)
    last_count = 0
    start_time = time.time()
    last_check_time = start_time
    
    print(f"\nMonitoring progress of {csv_path}")
    print(f"Total files to process: {total_files}")
    print(f"Checking every {interval} seconds...")
    print("\nTime Elapsed  |  Files Processed  |  Progress  |  Files/Minute  |  Est. Time Remaining")
    print("-" * 85)
    
    while True:
        try:
            if csv_path.exists():
                # Count lines in CSV (subtract 1 for header)
                with open(csv_path, 'r') as f:
                    current_count = sum(1 for _ in f) - 1
                
                current_time = time.time()
                elapsed_time = current_time - start_time
                elapsed_minutes = elapsed_time / 60
                
                # Calculate processing rate
                if elapsed_minutes > 0:
                    files_per_minute = current_count / elapsed_minutes
                else:
                    files_per_minute = 0
                
                # Calculate estimated time remaining
                if files_per_minute > 0:
                    remaining_files = total_files - current_count
                    est_minutes_remaining = remaining_files / files_per_minute
                else:
                    est_minutes_remaining = float('inf')
                
                # Calculate progress percentage
                progress = (current_count / total_files) * 100
                
                # Format times
                elapsed_str = f"{int(elapsed_minutes//60):02d}:{int(elapsed_minutes%60):02d}:00"
                if est_minutes_remaining == float('inf'):
                    remaining_str = "Unknown"
                else:
                    remaining_str = f"{int(est_minutes_remaining//60):02d}:{int(est_minutes_remaining%60):02d}:00"
                
                # Print progress
                print(f"\r{elapsed_str}  |  {current_count:7d}/{total_files}  |  {progress:6.1f}%  |  {files_per_minute:9.1f}  |  {remaining_str}", 
                      end='', flush=True)
                
                # If processing is complete, break
                if current_count >= total_files:
                    print("\n\nProcessing complete!")
                    break
                
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"\nError reading CSV: {e}")
            time.sleep(interval)

if __name__ == "__main__":
    monitor_progress()