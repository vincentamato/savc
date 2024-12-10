import json
import argparse

def count_unique_midis(matches_file):
    """Counts the number of unique MIDI matches in the results file."""
    try:
        # Load the JSON file
        with open(matches_file, 'r') as f:
            data = json.load(f)
        
        # Extract unique keys from best matches
        midi_keys = {match['best_match']['key'] for match in data if match['best_match']}
        
        # Count unique keys
        unique_count = len(midi_keys)
        print(f"Total unique MIDI keys used: {unique_count}")
        return unique_count
    except Exception as e:
        print(f"Error: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Count unique MIDI keys")
    parser.add_argument('--matches_file', required=True, help="Path to the JSON file with matches")
    args = parser.parse_args()
    count_unique_midis(args.matches_file)

if __name__ == "__main__":
    main()
