import numpy as np
import pandas as pd
from PIL import Image
import cv2
import json
from tqdm import tqdm
import os
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool, cpu_count
from functools import partial

class ArtworkMusicMatcher:
    def __init__(self, midi_features_path, image_base_dir, n_workers=None):
        self.midi_features = pd.read_csv(midi_features_path)
        self.image_base_dir = image_base_dir
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        print(f"Initialized with {self.n_workers} worker processes")
        
    def analyze_image(self, img_array):
        """Extract emotional and musical characteristics from image"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Calculate color features
        avg_saturation = np.mean(hsv[:,:,1]) / 255.0
        avg_brightness = np.mean(hsv[:,:,2]) / 255.0
        
        # Calculate emotional characteristics
        major_mode_score = avg_brightness * 0.7 + avg_saturation * 0.3
        tempo_score = avg_saturation * 0.8 + avg_brightness * 0.2
        pitch_score = avg_brightness * 0.6 + avg_saturation * 0.4
        
        return {
            'major_mode_score': major_mode_score,
            'tempo_score': tempo_score,
            'pitch_score': pitch_score
        }
    
    def calculate_midi_features(self, midi_row):
        """Normalize MIDI features to 0-1 range"""
        # Convert mode to numeric (major=1, minor=0)
        mode_score = 1.0 if midi_row['key_mode'].lower() == 'major' else 0.0
        
        # Normalize tempo (assuming range 40-208 bpm)
        tempo_score = (midi_row['avg_tempo'] - 40) / (208 - 40)
        
        # Normalize pitch (MIDI pitch range 0-127)
        pitch_score = midi_row['avg_pitch'] / 127
        
        return {
            'mode_score': mode_score,
            'tempo_score': tempo_score,
            'pitch_score': pitch_score
        }
    
    def calculate_compatibility_score(self, image_features, midi_features):
        """Calculate compatibility score between image and MIDI"""
        # Weight the different components
        weights = {
            'mode': 0.4,
            'tempo': 0.3,
            'pitch': 0.3
        }
        
        # Calculate weighted differences
        mode_diff = abs(image_features['major_mode_score'] - midi_features['mode_score'])
        tempo_diff = abs(image_features['tempo_score'] - midi_features['tempo_score'])
        pitch_diff = abs(image_features['pitch_score'] - midi_features['pitch_score'])
        
        # Convert differences to similarity scores (1 - diff)
        total_score = (
            weights['mode'] * (1 - mode_diff) +
            weights['tempo'] * (1 - tempo_diff) +
            weights['pitch'] * (1 - pitch_diff)
        )
        
        return total_score

    def process_image_chunk(self, args):
        """Process a chunk of images and calculate compatibility scores"""
        img_paths, start_idx = args
        scores = np.zeros((len(img_paths), len(self.midi_features)))
        
        for i, img_path in enumerate(img_paths):
            try:
                # Load and process image
                img = Image.open(os.path.join(self.image_base_dir, img_path))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_array = np.array(img)
                
                # Get image features
                image_features = self.analyze_image(img_array)
                
                # Calculate compatibility with each MIDI
                for j, (_, midi_row) in enumerate(self.midi_features.iterrows()):
                    midi_features = self.calculate_midi_features(midi_row)
                    score = self.calculate_compatibility_score(image_features, midi_features)
                    scores[i, j] = score
                    
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                scores[i, :] = 0  # Set all scores to 0 for failed image
                
        return scores, start_idx

    def find_optimal_matches(self, image_paths, chunk_size=10):
        """Find optimal one-to-one matches using parallel processing"""
        n_images = len(image_paths)
        n_midis = len(self.midi_features)
        
        # Split images into chunks for parallel processing
        chunks = []
        for i in range(0, n_images, chunk_size):
            chunk = image_paths[i:i + chunk_size]
            chunks.append((chunk, i))
        
        # Process chunks in parallel
        print(f"Processing {len(chunks)} chunks using {self.n_workers} workers...")
        cost_matrix = np.zeros((n_images, n_midis))
        
        with Pool(self.n_workers) as pool:
            for chunk_scores, start_idx in tqdm(
                pool.imap(self.process_image_chunk, chunks),
                total=len(chunks),
                desc="Processing image chunks"
            ):
                # Update cost matrix with chunk results
                end_idx = start_idx + chunk_scores.shape[0]
                cost_matrix[start_idx:end_idx, :] = 1 - chunk_scores  # Convert to costs
        
        # Find optimal assignment using Hungarian algorithm
        print("Finding optimal matches...")
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create matches dictionary
        matches = []
        for i, j in zip(row_ind, col_ind):
            matches.append({
                'image_path': image_paths[i],
                'midi_path': self.midi_features.iloc[j]['path'],
                'compatibility_score': 1 - cost_matrix[i, j]
            })
        
        return matches
    
    def save_matches(self, matches, output_path):
        """Save matches to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(matches, f, indent=2)
        print(f"Matches saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Match artwork with MIDI files (one-to-one)')
    parser.add_argument('--midi_features', required=True, help='Path to MIDI features CSV file')
    parser.add_argument('--image_dir', required=True, help='Base directory containing images')
    parser.add_argument('--image_list', required=True, help='File containing list of image paths')
    parser.add_argument('--output', default='optimal_matches.json', help='Output JSON file path')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    parser.add_argument('--chunk_size', type=int, default=10, help='Number of images to process per chunk')
    
    args = parser.parse_args()
    
    # Load image paths
    with open(args.image_list, 'r') as f:
        image_paths = [line.strip() for line in f]
    
    # Create matcher and find optimal matches
    matcher = ArtworkMusicMatcher(args.midi_features, args.image_dir, n_workers=args.workers)
    matches = matcher.find_optimal_matches(image_paths, chunk_size=args.chunk_size)
    matcher.save_matches(matches, args.output)