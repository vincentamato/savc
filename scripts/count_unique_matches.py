import json
import argparse

def count_unique_midis(matches_file):
    """Counts the number of unique MIDI files used in the results file."""
    try:
        # Load the JSON file
        with open(matches_file, 'r') as f:
            data = json.load(f)
        
        # Extract unique midi indices from best matches
        midi_indices = {match['best_match']['midi_idx'] for match in data if match['best_match']}
        
        # Count unique MIDI files
        unique_count = len(midi_indices)
        print(f"Total unique MIDI files used: {unique_count}")
        print(f"Total matches: {len(data)}")
        print(f"Unique MIDI usage percentage: {(unique_count / len(data)) * 100:.2f}%")

        # Additional analysis
        usage_counts = {}
        for match in data:
            if match['best_match']:
                midi_idx = match['best_match']['midi_idx']
                usage_counts[midi_idx] = usage_counts.get(midi_idx, 0) + 1

        if usage_counts:
            max_usage = max(usage_counts.values())
            min_usage = min(usage_counts.values())
            avg_usage = sum(usage_counts.values()) / len(usage_counts)
            print(f"\nUsage Statistics:")
            print(f"Most used MIDI: {max_usage} times")
            print(f"Least used MIDI: {min_usage} times")
            print(f"Average usage per MIDI: {avg_usage:.1f} times")

        return unique_count
    except Exception as e:
        print(f"Error: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Count unique MIDI files used")
    parser.add_argument('--matches_file', required=True, help="Path to the JSON file with matches")
    args = parser.parse_args()
    count_unique_midis(args.matches_file)

if __name__ == "__main__":
    main()