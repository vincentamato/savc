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
from collections import Counter, defaultdict
import logging
from scipy import stats
import sys
from datetime import datetime
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionalAnalyzer:
    """Analyzes emotional content of artwork based on color and composition"""
    
    KEY_EMOTION_MAPPINGS = {
        'C major': {'emotions': ['pure', 'innocent', 'simple'], 'weight': 0.8},
        'G major': {'emotions': ['serene', 'gentle', 'rustic'], 'weight': 0.7},
        'D major': {'emotions': ['triumphant', 'victorious'], 'weight': 0.9},
        'A major': {'emotions': ['confident', 'warm', 'optimistic'], 'weight': 0.8},
        'E major': {'emotions': ['radiant', 'joyful', 'bright'], 'weight': 0.7},
        'B major': {'emotions': ['passionate', 'bold'], 'weight': 0.6},
        'F# major': {'emotions': ['triumphant', 'brilliant'], 'weight': 0.5},
        'F major': {'emotions': ['pastoral', 'calm'], 'weight': 0.8},
        'Bb major': {'emotions': ['noble', 'elegant'], 'weight': 0.7},
        'Eb major': {'emotions': ['heroic', 'bold'], 'weight': 0.6},
        'Ab major': {'emotions': ['graceful', 'dreamy'], 'weight': 0.7},
        'C minor': {'emotions': ['tragic', 'dark'], 'weight': 0.9},
        'G minor': {'emotions': ['serious', 'dramatic'], 'weight': 0.8},
        'D minor': {'emotions': ['melancholic', 'serious'], 'weight': 0.9},
        'A minor': {'emotions': ['tender', 'lyrical'], 'weight': 0.8},
        'E minor': {'emotions': ['restless', 'haunting'], 'weight': 0.7},
        'B minor': {'emotions': ['solitary', 'dark'], 'weight': 0.8},
        'F# minor': {'emotions': ['tragic', 'mysterious'], 'weight': 0.7},
        'F minor': {'emotions': ['melancholic', 'obscure'], 'weight': 0.8}
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
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        
        self.emotional_analyzer = EmotionalAnalyzer()
        self.composition_analyzer = EnhancedCompositionAnalyzer()
        
        self.match_history = defaultdict(list)
        self.midi_usage = Counter()
        self.key_usage = Counter()
        
        # Enhanced preprocessing
        self._preprocess_midi_features()
        
        logger.info(f"Initialized matcher with {len(self.midi_features)} MIDI files")

    def _preprocess_midi_features(self):
        """Preprocess MIDI features with enhanced normalization"""
        numerical_columns = [
            'tempo', 'avg_velocity', 'notes_per_second', 
            'avg_pitch', 'pitch_range', 'pitch_std',
            'velocity_std', 'duration_mean'
        ]
        
        # Ensure all numerical columns exist
        for col in numerical_columns:
            if col not in self.midi_features.columns:
                self.midi_features[col] = 0.0
        
        # Robust scaling with outlier handling
        scaler = MinMaxScaler()
        self.midi_features[numerical_columns] = scaler.fit_transform(
            np.clip(self.midi_features[numerical_columns], 
                   *np.percentile(self.midi_features[numerical_columns], [1, 99]))
        )
        
        # Calculate emotional features
        self.midi_features['emotional_intensity'] = (
            self.midi_features['avg_velocity'] * 0.4 +
            self.midi_features['notes_per_second'] * 0.3 +
            self.midi_features['velocity_std'] * 0.3
        )
        
        self.midi_features['complexity'] = (
            self.midi_features['notes_per_second'] * 0.4 +
            self.midi_features['pitch_std'] * 0.3 +
            self.midi_features['duration_mean'] * 0.3
        )
        
        # Extract and normalize key information
        self.midi_features['mode'] = self.midi_features['key'].apply(
            lambda x: 'major' if 'major' in str(x).lower() else 'minor'
        )
        
        # Ensure original_path exists
        # Ensure original_path exists
        if 'original_path' not in self.midi_features.columns:
            self.midi_features['original_path'] = self.midi_features.index.astype(str)
            logger.warning("Created default original_path from index")
            
        # Create unique identifier for each MIDI file
        self.midi_features['midi_identifier'] = self.midi_features['original_path'].apply(
            lambda x: os.path.basename(str(x))
        )

    def _save_progress(self, matches, output_path, processed_count=None):
        """Save matching progress with detailed statistics"""
        try:
            if not matches:
                logger.warning("No matches to save!")
                return
                
            save_data = {
                'matches': matches,
                'statistics': {
                    'total_matches': len(matches),
                    'unique_midis_used': len(set(m['midi_identifier'] for m in matches)),
                    'key_distribution': dict(self.key_usage),
                    'average_score': float(np.mean([m['compatibility_score'] for m in matches])),
                    'score_distribution': {
                        'min': float(min(m['compatibility_score'] for m in matches)),
                        'max': float(max(m['compatibility_score'] for m in matches)),
                        'std': float(np.std([m['compatibility_score'] for m in matches]))
                    },
                    'processed_count': processed_count,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Atomic save with backup
            temp_path = output_path + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            os.replace(temp_path, output_path)
            
        except Exception as e:
            logger.error(f"Error saving progress: {str(e)}")

    def _calculate_compatibility(self, midi_row, composition, emotions, key_matches):
        """Calculate compatibility score with stronger diversity emphasis"""
        base_score = 0.0
        weights = {
            'key_emotional': 0.20,  # Reduced from 0.25
            'tempo': 0.20,
            'complexity': 0.20,
            'movement': 0.20,
            'mode': 0.10,  # Reduced from 0.15
            'diversity': 0.10  # New weight for diversity
        }
        
        # Calculate base components as before
        key_match = next((m for m in key_matches if m['key'] == midi_row['key']), None)
        if key_match:
            base_score += key_match['score'] * weights['key_emotional']
        
        tempo_diff = abs(composition['suggested_tempo'] - (midi_row['tempo'] * 200))
        tempo_score = max(0, 1 - (tempo_diff / 100))
        base_score += tempo_score * weights['tempo']
        
        complexity_diff = abs(composition['complexity'] - midi_row['complexity'])
        complexity_score = max(0, 1 - complexity_diff)
        base_score += complexity_score * weights['complexity']
        
        movement_score = max(0, 1 - abs(
            composition['movement']['movement_intensity'] - midi_row['emotional_intensity']
        ))
        base_score += movement_score * weights['movement']
        
        # Mode scoring with less weight
        if emotions['valence'] > 0.6 and midi_row['mode'] == 'major':
            base_score += weights['mode']
        elif emotions['valence'] < 0.4 and midi_row['mode'] == 'minor':
            base_score += weights['mode']
        
        # Enhanced diversity scoring
        usage_count = self.midi_usage[midi_row['midi_identifier']]
        diversity_score = 1.0 / (1.0 + usage_count)  # Decreases with usage
        base_score += diversity_score * weights['diversity']
        
        # Apply stronger diversity penalty
        diversity_penalty = min(0.5, usage_count * 0.15)  # Increased from 0.3/0.1
        final_score = max(0, base_score - diversity_penalty)
        
        return float(final_score)

    def _optimize_batch_diversity(self, matches, min_distance=0.15):  # Reduced from 0.2
        """Optimize batch of matches with stronger diversity emphasis"""
        if not matches:
            return matches
            
        optimized = [matches[0]]
        used_midis = {matches[0]['midi_identifier']}
        used_keys = {matches[0]['key']}
        
        for match in matches[1:]:
            # More lenient MIDI reuse conditions
            if match['midi_identifier'] in used_midis and len(matches) > len(used_midis) * 2:  # Reduced from 3
                continue
                
            # Calculate diversity metrics
            key_diversity = match['key'] not in used_keys
            tempo_diversity = all(abs(match['tempo'] - m['tempo']) > 0.15 for m in optimized[-2:])  # Reduced window
            
            # Calculate feature distance with more features
            recent_features = np.array([[
                m['analysis']['suggested_tempo'] / 200.0,
                m['analysis']['complexity_match'],
                m['analysis']['movement_intensity'],
                float(m['key'] == match['key']),  # Add key comparison
                self.midi_usage[m['midi_identifier']] / 10.0  # Add usage history
            ] for m in optimized[-2:]])  # Reduced window
            
            current_features = np.array([
                match['analysis']['suggested_tempo'] / 200.0,
                match['analysis']['complexity_match'],
                match['analysis']['movement_intensity'],
                0.0,  # New key comparison
                self.midi_usage[match['midi_identifier']] / 10.0
            ])
            
            # Calculate minimum distance to recent matches
            if len(recent_features) > 0:
                distances = np.linalg.norm(recent_features - current_features, axis=1)
                min_dist = np.min(distances)
            else:
                min_dist = float('inf')
            
            # More lenient acceptance criteria
            if min_dist > min_distance or key_diversity or tempo_diversity:
                # Add extra check for global diversity
                if len(used_midis) < len(self.midi_features) * 0.8:  # Ensure we use at least 80% of available MIDIs
                    optimized.append(match)
                    used_midis.add(match['midi_identifier'])
                    used_keys.add(match['key'])
            
            # Force diversity if using too few MIDIs
            if len(optimized) > 10 and len(used_midis) < len(optimized) * 0.4:  # Ensure at least 40% unique MIDIs
                for potential_match in matches:
                    if potential_match['midi_identifier'] not in used_midis:
                        optimized.append(potential_match)
                        used_midis.add(potential_match['midi_identifier'])
                        used_keys.add(potential_match['key'])
                        break
        
        return optimized

    def find_best_match(self, image_path):
        """Find best matching MIDI file with forced diversity"""
        try:
            img_array = self._load_and_process_image(image_path)
            if img_array is None:
                return None
            
            composition = self.composition_analyzer.analyze_composition(img_array)
            emotions = self.emotional_analyzer.analyze_color_emotion(img_array)
            key_matches = self.emotional_analyzer.match_key_to_emotion(emotions, composition)
            
            # Calculate usage statistics
            total_matches = sum(self.midi_usage.values())
            if total_matches > 0:
                avg_usage = total_matches / len(self.midi_usage)
                unused_ratio = len([m for m in self.midi_features['midi_identifier'] if m not in self.midi_usage]) / len(self.midi_features)
            else:
                avg_usage = 0
                unused_ratio = 1.0
            
            # Adjust scoring based on usage statistics
            matches = []
            for _, midi_row in self.midi_features.iterrows():
                base_score = self._calculate_compatibility(midi_row, composition, emotions, key_matches)
                
                # Calculate diversity bonus
                usage_count = self.midi_usage.get(midi_row['midi_identifier'], 0)
                if usage_count == 0:
                    diversity_bonus = 0.3  # Big bonus for unused files
                elif usage_count < avg_usage:
                    diversity_bonus = 0.2  # Moderate bonus for below-average usage
                else:
                    diversity_bonus = max(0, 0.1 - (usage_count - avg_usage) * 0.02)  # Decreasing bonus for heavily used files
                
                # Force selection of unused files periodically
                if unused_ratio < 0.4 and usage_count == 0:  # If less than 40% of files used
                    diversity_bonus += 0.5  # Very large bonus for unused files
                
                final_score = base_score + diversity_bonus
                
                matches.append({
                    'image_path': image_path,
                    'midi_identifier': midi_row['midi_identifier'],
                    'compatibility_score': final_score,
                    'key': midi_row['key'],
                    'tempo': float(midi_row['tempo']),
                    'analysis': {
                        'emotional_matches': [m['key'] for m in key_matches[:3]],
                        'suggested_tempo': float(composition['suggested_tempo']),
                        'complexity_match': float(composition['complexity']),
                        'movement_intensity': float(composition['movement']['movement_intensity'])
                    }
                })
            
            # Sort by score
            matches.sort(key=lambda x: x['compatibility_score'], reverse=True)
            
            # Force selection of unused MIDI files periodically
            if unused_ratio < 0.4:
                for match in matches:
                    if self.midi_usage.get(match['midi_identifier'], 0) == 0:
                        selected_match = match
                        break
                else:
                    selected_match = matches[0]
            else:
                # Select from top matches with preference for less used files
                top_matches = matches[:10]
                top_matches.sort(key=lambda x: self.midi_usage.get(x['midi_identifier'], 0))
                selected_match = top_matches[0]
            
            # Update usage tracking
            self.midi_usage[selected_match['midi_identifier']] += 1
            self.key_usage[selected_match['key']] += 1
            
            folder_path = os.path.dirname(image_path)
            if folder_path not in self.match_history:
                self.match_history[folder_path] = []
            self.match_history[folder_path].append(selected_match)
            
            return selected_match
            
        except Exception as e:
            logger.error(f"Error matching {image_path}: {str(e)}")
            return None

    def _load_and_process_image(self, img_path):
        """Load and preprocess image with enhanced error handling"""
        try:
            possible_paths = [
                os.path.join(self.image_base_dir, img_path),
                os.path.join(self.image_base_dir, os.path.basename(img_path)),
                img_path
            ]
            
            valid_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    valid_path = path
                    break
            
            if not valid_path:
                logger.error(f"Image not found at any of: {possible_paths}")
                return None
            
            img = Image.open(valid_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize maintaining aspect ratio
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            return np.array(img)
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            return None

    def process_images_batch(self, image_paths, batch_size=100):
        """Process images in batches with improved progress tracking"""
        print(f"Processing {len(image_paths)} images...")
        
        # Validate paths upfront
        valid_paths = []
        for path in image_paths:
            if self._load_and_process_image(path) is not None:
                valid_paths.append(path)
        
        if not valid_paths:
            logger.error("No valid image paths found!")
            return []
        
        all_matches = []
        temp_save_path = 'temp_matches.json'
        
        try:
            with Pool(self.n_workers) as pool:
                for batch_start in range(0, len(valid_paths), batch_size):
                    batch_paths = valid_paths[batch_start:min(batch_start + batch_size, len(valid_paths))]
                    
                    batch_results = []
                    for result in tqdm(
                        pool.imap_unordered(self.find_best_match, batch_paths),
                        total=len(batch_paths),
                        desc=f"Batch {batch_start//batch_size + 1}"
                    ):
                        if result is not None:
                            batch_results.append(result)
                    
                    # Optimize batch diversity
                    optimized_results = self._optimize_batch_diversity(batch_results)
                    all_matches.extend(optimized_results)
                    
                    # Save progress
                    if len(all_matches) % 100 == 0:
                        self._save_progress(all_matches, temp_save_path)
            
            return all_matches
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            if all_matches:
                self._save_progress(all_matches, temp_save_path)
            return all_matches
        
        finally:
            if os.path.exists(temp_save_path):
                try:
                    os.remove(temp_save_path)
                except Exception as e:
                    logger.warning(f"Could not remove temporary file: {str(e)}")

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
    
    try:
        with open(args.image_list, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        matcher = ImprovedArtworkMusicMatcher(
            args.midi_features,
            args.image_dir,
            args.workers
        )
        
        matches = matcher.process_images_batch(image_paths, args.batch_size)
        
        if matches:
            matcher._save_progress(matches, args.output)
            print(f"\nProcessing complete! Found {len(matches)} matches.")
            print(f"Used {len(set(m['midi_identifier'] for m in matches))} unique MIDI files.")
        else:
            print("\nNo matches found. Please check the image paths and data directory structure.")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()