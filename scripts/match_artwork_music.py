import numpy as np
import pandas as pd
from PIL import Image
import cv2
import json
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import MinMaxScaler
import colorsys

class ArtworkMusicMatcher:
    def __init__(self, midi_features_path, image_base_dir, n_workers=None):
        self.midi_features = pd.read_csv(midi_features_path)
        self.image_base_dir = image_base_dir
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        
        # Initialize scalers for MIDI features
        self.init_scalers()
        print(f"Initialized with {self.n_workers} worker processes")
        
    def init_scalers(self):
        """Initialize scalers for MIDI features normalization"""
        self.scalers = {
            'tempo': MinMaxScaler(),
            'pitch': MinMaxScaler(),
            'velocity': MinMaxScaler(),
            'density': MinMaxScaler()
        }
        
        # Fit scalers
        self.scalers['tempo'].fit(self.midi_features[['tempo']].values)
        self.scalers['pitch'].fit(self.midi_features[['avg_pitch']].values)
        self.scalers['velocity'].fit(self.midi_features[['avg_velocity']].values)
        self.scalers['density'].fit(self.midi_features[['notes_per_second']].values.reshape(-1, 1))

    def analyze_image(self, img_array):
        """Extract emotional and musical characteristics from image"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Calculate basic color features
        avg_saturation = np.mean(hsv[:,:,1]) / 255.0
        avg_brightness = np.mean(hsv[:,:,2]) / 255.0
        
        # Calculate image complexity using edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        complexity = np.count_nonzero(edges) / edges.size
        
        # Calculate energy (based on saturation, brightness, and complexity)
        energy = (avg_saturation * 0.4 + avg_brightness * 0.3 + complexity * 0.3)
        
        # Musical characteristics
        tempo_score = energy * 0.7 + complexity * 0.3
        pitch_score = avg_brightness * 0.6 + avg_saturation * 0.4
        velocity_score = avg_saturation * 0.6 + complexity * 0.4
        density_score = complexity
        
        return {
            'tempo_score': tempo_score,
            'pitch_score': pitch_score,
            'velocity_score': velocity_score,
            'density_score': density_score,
            'complexity': complexity,
            'energy': energy
        }
    
    def calculate_midi_features(self, midi_row):
        """Extract normalized MIDI features"""
        return {
            'tempo_score': self.scalers['tempo'].transform([[midi_row['tempo']]])[0][0],
            'pitch_score': self.scalers['pitch'].transform([[midi_row['avg_pitch']]])[0][0],
            'velocity_score': self.scalers['velocity'].transform([[midi_row['avg_velocity']]])[0][0],
            'density_score': self.scalers['density'].transform([[midi_row['notes_per_second']]])[0][0],
        }
    
    def calculate_compatibility_score(self, image_features, midi_features):
        """Calculate weighted compatibility score between image and MIDI"""
        weights = {
            'tempo': 0.3,
            'pitch': 0.3,
            'velocity': 0.2,
            'density': 0.2
        }
        
        differences = {
            'tempo': abs(image_features['tempo_score'] - midi_features['tempo_score']),
            'pitch': abs(image_features['pitch_score'] - midi_features['pitch_score']),
            'velocity': abs(image_features['velocity_score'] - midi_features['velocity_score']),
            'density': abs(image_features['density_score'] - midi_features['density_score'])
        }
        
        score = sum(weights[k] * (1 - differences[k]) for k in weights)
        return max(0, min(1, score))

    def find_best_match(self, img_path):
        """Find the best matching MIDI file for a single image"""
        try:
            # Construct full image path
            full_img_path = os.path.join(self.image_base_dir, img_path)
            
            # Check if file exists
            if not os.path.exists(full_img_path):
                print(f"Image file not found: {full_img_path}")
                return None
            
            # Load and process image
            img = Image.open(full_img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
            
            # Get image features
            image_features = self.analyze_image(img_array)
            
            # Calculate compatibility with each MIDI and find the best match
            best_score = -1
            best_midi_idx = -1
            
            for idx, midi_row in self.midi_features.iterrows():
                midi_features = self.calculate_midi_features(midi_row)
                score = self.calculate_compatibility_score(image_features, midi_features)
                
                if score > best_score:
                    best_score = score
                    best_midi_idx = idx
            
            if best_midi_idx >= 0:
                return {
                    'image_path': img_path,
                    'midi_path': str(self.midi_features.iloc[best_midi_idx]['extracted_path']),
                    'compatibility_score': float(best_score),
                    'image_features': {k: float(v) for k, v in image_features.items() 
                                     if isinstance(v, (int, float))}
                }
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
        
        return None

    def process_images(self, image_paths):
        """Process images in parallel to find best matches"""
        print(f"Processing {len(image_paths)} images using {self.n_workers} workers...")
        
        # Filter out invalid paths first
        valid_paths = []
        for path in image_paths:
            full_path = os.path.join(self.image_base_dir, path)
            if os.path.exists(full_path):
                valid_paths.append(path)
            else:
                print(f"Skipping non-existent file: {full_path}")

        print(f"Found {len(valid_paths)} valid image paths")
        
        matches = []
        with Pool(self.n_workers) as pool:
            for match in tqdm(
                pool.imap_unordered(self.find_best_match, valid_paths),
                total=len(valid_paths),
                desc="Finding best matches"
            ):
                if match is not None:
                    matches.append(match)
        
        return matches
    
    def save_matches(self, matches, output_path):
        """Save matches to JSON file"""
        with open(output_path, 'w') as f:
            json.dump({
                'matches': matches,
                'total_matches': len(matches)
            }, f, indent=2)
        print(f"Matches saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Match artwork with best MIDI file')
    parser.add_argument('--midi_features', required=True, help='Path to MIDI features CSV file')
    parser.add_argument('--image_dir', required=True, help='Base directory containing images')
    parser.add_argument('--image_list', required=True, help='File containing list of image paths')
    parser.add_argument('--output', default='matches.json', help='Output JSON file path')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Read and validate image paths
    with open(args.image_list, 'r') as f:
        image_paths = [line.strip() for line in f]
    
    print(f"Read {len(image_paths)} paths from image list")
    
    matcher = ArtworkMusicMatcher(args.midi_features, args.image_dir, n_workers=args.workers)
    matches = matcher.process_images(image_paths)
    matcher.save_matches(matches, args.output)