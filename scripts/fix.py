import json
from pathlib import Path

def fix_json_paths(json_path: str) -> None:
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Fix paths in the matches
    for artwork_path, match_data in data['matches'].items():
        for midi_match in match_data['midi_matches']:
            # Fix MIDI path by removing data/ prefix and changing unique_midis to midis
            old_path = midi_match['midi_path']
            new_path = old_path.replace('data/unique_midis/', 'midis/')
            midi_match['midi_path'] = new_path
    
    # Save the modified JSON
    output_path = Path(json_path).parent / 'matches_fixed.json'
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Fixed JSON saved to {output_path}")

if __name__ == "__main__":
    fix_json_paths('data/matches.json')