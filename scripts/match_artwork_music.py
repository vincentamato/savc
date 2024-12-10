import multiprocessing
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import json
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count, Manager
from sklearn.preprocessing import MinMaxScaler
import logging
from scipy import stats
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionalAnalyzer:
    """Analyzes emotional content of artwork based on color and composition"""
    KEY_EMOTION_MAPPINGS = {
        'C Maj': {'emotions': ['pure', 'innocent', 'simple'], 'weight': 0.8},
        'G Maj': {'emotions': ['serene', 'gentle', 'rustic'], 'weight': 0.7},
        'D Maj': {'emotions': ['triumphant', 'victorious'], 'weight': 0.9},
        'A Maj': {'emotions': ['confident', 'warm', 'optimistic'], 'weight': 0.8},
        'E Maj': {'emotions': ['radiant', 'joyful', 'bright'], 'weight': 0.7},
        'B Maj': {'emotions': ['passionate', 'bold'], 'weight': 0.6},
        'F# Maj': {'emotions': ['triumphant', 'brilliant'], 'weight': 0.5},
        'F Maj': {'emotions': ['pastoral', 'calm'], 'weight': 0.8},
        'Bb Maj': {'emotions': ['noble', 'elegant'], 'weight': 0.7},
        'Eb Maj': {'emotions': ['heroic', 'bold'], 'weight': 0.6},
        'Ab Maj': {'emotions': ['graceful', 'dreamy'], 'weight': 0.7},
        'C Min': {'emotions': ['tragic', 'dark'], 'weight': 0.9},
        'G Min': {'emotions': ['serious', 'dramatic'], 'weight': 0.8},
        'D Min': {'emotions': ['melancholic', 'serious'], 'weight': 0.9},
        'A Min': {'emotions': ['tender', 'lyrical'], 'weight': 0.8},
        'E Min': {'emotions': ['restless', 'haunting'], 'weight': 0.7},
        'B Min': {'emotions': ['solitary', 'dark'], 'weight': 0.8},
        'F# Min': {'emotions': ['tragic', 'mysterious'], 'weight': 0.7},
        'F Min': {'emotions': ['melancholic', 'obscure'], 'weight': 0.8}
    }

    @staticmethod
    def analyze_color_emotion(img_array):
        """Analyze emotional content based on color properties with improved metrics"""
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Calculate weighted color statistics
        weights = np.ones_like(img_hsv[:,:,2]) / (img_hsv[:,:,2].size)
        hue_mean = np.average(img_hsv[:,:,0], weights=weights)
        saturation_mean = np.average(img_hsv[:,:,1], weights=weights)
        value_mean = np.average(img_hsv[:,:,2], weights=weights)
        
        # Calculate color variance with improved weighting
        hue_var = np.average((img_hsv[:,:,0] - hue_mean)**2, weights=weights)
        sat_var = np.average((img_hsv[:,:,1] - saturation_mean)**2, weights=weights)
        val_var = np.average((img_hsv[:,:,2] - value_mean)**2, weights=weights)
        
        # Calculate color distributions
        hue_hist = cv2.calcHist([img_hsv], [0], None, [18], [0, 180])
        sat_hist = cv2.calcHist([img_hsv], [1], None, [25], [0, 256])
        val_hist = cv2.calcHist([img_hsv], [2], None, [25], [0, 256])
        
        # Normalize histograms
        hue_hist = hue_hist.ravel() / np.sum(hue_hist)
        sat_hist = sat_hist.ravel() / np.sum(sat_hist)
        val_hist = val_hist.ravel() / np.sum(val_hist)
        
        # Calculate entropy for complexity
        hue_entropy = stats.entropy(hue_hist)
        sat_entropy = stats.entropy(sat_hist)
        val_entropy = stats.entropy(val_hist)
        
        emotions = {
            'valence': 0.0,
            'arousal': 0.0,
            'dominance': 0.0,
            'complexity': 0.0
        }
        
        # Enhanced valence calculation
        emotions['valence'] = (
            (saturation_mean / 255.0) * 0.3 +
            (value_mean / 255.0) * 0.4 +
            (1 - hue_var / (180 * 180)) * 0.2 +
            (1 - val_entropy / np.log(25)) * 0.1  # Lower entropy -> more positive
        )
        
        # Enhanced arousal calculation
        emotions['arousal'] = (
            (saturation_mean / 255.0) * 0.3 +
            (value_mean / 255.0) * 0.2 +
            (sat_var / (255 * 255)) * 0.2 +
            (hue_entropy / np.log(18)) * 0.3  # Higher entropy -> more exciting
        )
        
        # Enhanced dominance calculation
        emotions['dominance'] = (
            (1 - value_mean / 255.0) * 0.3 +
            (saturation_mean / 255.0) * 0.3 +
            (val_var / (255 * 255)) * 0.2 +
            (sat_entropy / np.log(25)) * 0.2
        )
        
        # Enhanced complexity calculation
        emotions['complexity'] = (
            (hue_entropy / np.log(18)) * 0.4 +
            (sat_entropy / np.log(25)) * 0.3 +
            (val_entropy / np.log(25)) * 0.3
        )
        
        return emotions

    @staticmethod
    def match_key_to_emotion(emotions, composition_analysis):
        """Match emotional content to musical keys with improved weighting"""
        matches = []
        darkness = 1.0 - emotions['valence']
        intensity = emotions['arousal'] * emotions['dominance']
        complexity = emotions['complexity']
        
        # Calculate movement intensity
        movement_factor = composition_analysis['movement']['movement_intensity']
        
        for key, properties in EmotionalAnalyzer.KEY_EMOTION_MAPPINGS.items():
            score = 0.0
            
            # Enhanced dark/dramatic scoring
            if darkness > 0.5:
                if 'minor' in key.lower():
                    score += darkness * (0.4 + intensity * 0.2)
                    if intensity > 0.6:
                        if key in ['C minor', 'G minor', 'D minor']:
                            score += intensity * 0.3
                    else:
                        if key in ['F minor', 'B minor', 'E minor']:
                            score += (1 - intensity) * 0.2
            else:
                if 'major' in key.lower():
                    score += (1 - darkness) * (0.3 + (1 - intensity) * 0.2)
                    if intensity > 0.5:
                        if key in ['D major', 'A major', 'E major']:
                            score += intensity * 0.25
                    else:
                        if key in ['F major', 'G major', 'C major']:
                            score += (1 - intensity) * 0.25
            
            # Movement and complexity considerations
            if movement_factor > 0.6:
                if key in ['D major', 'E major', 'G minor', 'D minor']:
                    score += movement_factor * 0.2
            else:
                if key in ['F major', 'Bb major', 'F minor', 'C minor']:
                    score += (1 - movement_factor) * 0.2
            
            if complexity > 0.6:
                if key in ['B minor', 'F# minor', 'E minor', 'C# minor']:
                    score += complexity * 0.2
            
            # Apply emotional weight
            final_score = score * properties['weight']
            
            # Normalize score
            final_score = max(0.0, min(1.0, final_score))
            
            matches.append({
                'key': key,
                'score': final_score,
                'emotions': properties['emotions']
            })
        
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:5]

class EnhancedCompositionAnalyzer:
    """Analyzes composition with improved metrics"""
    
    @staticmethod
    def analyze_composition(img_array):
        """Analyze compositional characteristics with enhanced features"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Multi-scale edge detection
        edges_fine = cv2.Canny(gray, 50, 150)
        edges_coarse = cv2.Canny(gray, 100, 200)
        
        complexity_fine = np.count_nonzero(edges_fine) / edges_fine.size
        complexity_coarse = np.count_nonzero(edges_coarse) / edges_coarse.size
        
        # Enhanced directional analysis
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx)
        
        # Calculate flow coherence
        direction_hist = np.histogram(direction.flatten(), bins=16, range=(-np.pi, np.pi))[0]
        direction_hist = direction_hist / np.sum(direction_hist)
        flow_coherence = 1.0 - stats.entropy(direction_hist) / np.log(16)
        
        movement = {
            'horizontal_flow': float(np.mean(np.abs(sobelx))) / 255.0,
            'vertical_flow': float(np.mean(np.abs(sobely))) / 255.0,
            'flow_coherence': float(flow_coherence),
            'movement_intensity': float(np.mean(magnitude)) / 255.0
        }
        
        # Enhanced symmetry calculation
        h, w = magnitude.shape
        left_half = magnitude[:, :w//2]
        right_half = np.fliplr(magnitude[:, w//2:])
        vertical_symmetry = 1.0 - np.mean(np.abs(left_half - right_half)) / 255.0
        
        top_half = magnitude[:h//2, :]
        bottom_half = np.flipud(magnitude[h//2:, :])
        horizontal_symmetry = 1.0 - np.mean(np.abs(top_half - bottom_half)) / 255.0
        
        # Calculate suggested tempo based on multiple factors
        base_tempo = 80
        tempo_factors = [
            movement['movement_intensity'] * 60,
            movement['flow_coherence'] * 40,
            complexity_fine * 30,
            (1 - vertical_symmetry) * 20
        ]
        suggested_tempo = base_tempo + sum(tempo_factors)
        
        # Enhanced rhythm complexity
        rhythm_complexity = (
            complexity_fine * 0.3 +
            complexity_coarse * 0.2 +
            movement['flow_coherence'] * 0.2 +
            movement['movement_intensity'] * 0.3
        )
        
        return {
            'complexity': float((complexity_fine + complexity_coarse) / 2),
            'symmetry': {
                'vertical': float(vertical_symmetry),
                'horizontal': float(horizontal_symmetry),
                'overall': float((vertical_symmetry + horizontal_symmetry) / 2)
            },
            'movement': movement,
            'suggested_tempo': float(suggested_tempo),
            'rhythm_complexity': float(rhythm_complexity),
            'balance': float(1.0 - abs(np.mean(magnitude[:, :w//2]) - 
                                    np.mean(magnitude[:, w//2:])) / 255.0)
        }

class ImprovedArtworkMusicMatcher:
    def __init__(self, midi_features_path, image_base_dir, n_workers=None):
        self.midi_features = pd.read_csv(midi_features_path)
        self.image_base_dir = os.path.abspath(image_base_dir)
        self.n_workers = n_workers or cpu_count()
        self.manager = Manager()
        self.shared_midi_usage = self.manager.dict()
        self.lock = self.manager.Lock()

        # Initialize MIDI usage counter
        self.midi_usage = {key: 0 for key in self.midi_features['key'].unique()}
        
        self._preprocess_midi_features()
        logger.info(f"Initialized matcher with {len(self.midi_features)} MIDI files")

    def _preprocess_midi_features(self):
        """Preprocess MIDI features with normalization and feature creation"""
        numerical_columns = [
            'tempo', 'avg_velocity', 'notes_per_second',
            'avg_pitch', 'pitch_range', 'total_notes',
            'avg_duration', 'total_duration'
        ]
        scaler = MinMaxScaler()
        for col in numerical_columns:
            if col not in self.midi_features.columns:
                self.midi_features[col] = 0.0

        self.midi_features[numerical_columns] = scaler.fit_transform(
            self.midi_features[numerical_columns]
        )
        
        # Normalize key format
        self.midi_features['key'] = self.midi_features['key'].str.strip()
        
        # Create derived features
        self.midi_features['emotional_intensity'] = (
            self.midi_features['avg_velocity'] * 0.4 +
            self.midi_features['notes_per_second'] * 0.3 +
            self.midi_features['pitch_range'] / self.midi_features['pitch_range'].max() * 0.3
        )
        self.midi_features['complexity'] = (
            self.midi_features['notes_per_second'] * 0.4 +
            self.midi_features['pitch_range'] / self.midi_features['pitch_range'].max() * 0.3 +
            self.midi_features['avg_duration'] * 0.3
        )

    def process_images_batch(self, image_paths, batch_size=100):
        """Process images in batches using worker pool"""
        valid_paths = [
            path for path in image_paths 
            if os.path.exists(os.path.join(self.image_base_dir, path))
        ]
        if not valid_paths:
            logger.error("No valid image paths found!")
            return []

        all_matches = []
        worker_batch_size = max(1, batch_size // self.n_workers)
        total_images = len(valid_paths)
        
        try:
            with Pool(
                self.n_workers,
                initializer=self._init_worker,
                initargs=(self.shared_midi_usage, self.lock)
            ) as pool:
                batches = [
                    valid_paths[i:i + worker_batch_size]
                    for i in range(0, len(valid_paths), worker_batch_size)
                ]
                args = [
                    (batch, self.midi_features, self.image_base_dir, self.midi_usage)
                    for batch in batches
                ]

                for result in tqdm(
                    pool.imap_unordered(self._process_batch, args),
                    total=len(batches),
                    desc="Processing Images"
                ):
                    all_matches.extend(result)

            return all_matches
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return []

    @staticmethod
    def _process_batch(args):
        """Process a batch of images with shared MIDI usage tracking"""
        image_paths, midi_features, image_base_dir, midi_usage = args
        matches = []

        for image_path in image_paths:
            try:
                img_array = ImprovedArtworkMusicMatcher._load_and_process_image(image_path, image_base_dir)
                if img_array is None:
                    continue

                composition = EnhancedCompositionAnalyzer.analyze_composition(img_array)
                emotions = EmotionalAnalyzer.analyze_color_emotion(img_array)
                key_matches = EmotionalAnalyzer.match_key_to_emotion(emotions, composition)

                if key_matches:
                    # Sort matches by score but also consider MIDI usage
                    valid_matches = [
                        match for match in key_matches 
                        if match['key'] in midi_usage
                    ]
                    
                    if not valid_matches:
                        continue

                    # Calculate coverage
                    total_keys = len(midi_usage)
                    used_keys = sum(1 for k, v in midi_usage.items() if v > 0)
                    coverage = used_keys / total_keys

                    # Only require 90% coverage before allowing reuse
                    if coverage >= 0.90:
                        # Select based on emotional score when coverage is met
                        best_match = max(valid_matches, key=lambda x: x['score'])
                    else:
                        # Prioritize unused keys
                        unused_matches = [m for m in valid_matches if midi_usage[m['key']] == 0]
                        best_match = (
                            unused_matches[0] if unused_matches 
                            else min(valid_matches, key=lambda x: midi_usage[x['key']])
                        )

                    # Update MIDI usage count
                    midi_usage[best_match['key']] += 1

                    match = {
                        'image_path': image_path,
                        'composition': composition,
                        'emotions': emotions,
                        'best_match': best_match,
                    }
                    matches.append(match)

            except Exception as e:
                logger.error(f"Error processing image {image_path}: {str(e)}")

        return matches

    @staticmethod
    def _init_worker(shared_midi_usage, lock):
        """Initialize worker process with shared state"""
        global _shared_midi_usage, _lock
        _shared_midi_usage = shared_midi_usage
        _lock = lock

    @staticmethod
    def _load_and_process_image(img_path, base_dir):
        """Static method for image loading"""
        try:
            full_path = os.path.join(base_dir, img_path)
            if not os.path.exists(full_path):
                return None

            img = Image.open(full_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            return np.array(img)
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            return None

def main():
    """Main function with improved argument handling"""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Artwork-Music Matcher')
    parser.add_argument('--midi_features', required=True, help='Path to MIDI features CSV')
    parser.add_argument('--image_dir', required=True, help='Base directory containing images')
    parser.add_argument('--image_list', required=True, help='File containing image paths')
    parser.add_argument('--output', default='matches.json', help='Output JSON path')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing')

    args = parser.parse_args()

    with open(args.image_list, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]

    matcher = ImprovedArtworkMusicMatcher(
        args.midi_features,
        args.image_dir,
        args.workers
    )

    matches = matcher.process_images_batch(image_paths, args.batch_size)
    if matches:
        with open(args.output, 'w') as output_file:
            json.dump(matches, output_file, indent=2)
        logger.info(f"Processing complete! Found {len(matches)} matches.")
    else:
        logger.warning("No matches found. Please check the image paths and data directory structure.")

if __name__ == "__main__":
    main()
