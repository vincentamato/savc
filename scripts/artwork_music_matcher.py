import numpy as np
import pandas as pd
from PIL import Image
import pretty_midi
import json
import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pickle
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp
from collections import defaultdict
import cv2
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EmotionalFeatures:
    """Emotional characteristics for both music and images."""
    valence: float  # -1 (negative) to 1 (positive)
    arousal: float  # 0 (calm) to 1 (excited)
    emotions: Dict[str, float]  # Scores for each emotion category
    narrative_quality: float  # 0 (abstract) to 1 (narrative)
    emotional_intensity: float  # 0 to 1
    tension_progression: List[float]  # Changes in tension over time
    
    @property
    def dominant_emotion(self) -> str:
        """Return the strongest emotion category."""
        return max(self.emotions.items(), key=lambda x: x[1])[0]

@dataclass
class MusicFeatures:
    """Musical characteristics extracted from MIDI."""
    # Existing features
    tempo: float
    note_density: float
    pitch_range: float
    avg_pitch: float
    avg_velocity: float
    key: str
    duration: float
    energy: float
    complexity: float
    brightness: float
    warmth: float
    tension: float
    midi_path: str
    
    # New emotional features
    emotional: EmotionalFeatures

@dataclass
class ImageFeatures:
    """Visual characteristics extracted from image."""
    # Existing features
    brightness: float
    warmth: float
    complexity: float
    energy: float
    color_palette: List[Tuple[float, float, float]]
    dominant_colors: List[Tuple[float, float, float]]
    image_path: str
    
    # New emotional features
    emotional: EmotionalFeatures

class EmotionalAnalyzer:
    """Analyze emotional content of music and images."""
    
    def __init__(self):
        # Define emotion categories
        self.emotion_categories = [
            'joy', 'sadness', 'triumph', 'melancholy',
            'tension', 'relief', 'longing', 'serenity',
            'excitement', 'contemplation', 'aggression', 'tenderness'
        ]
        
        # Musical key emotional associations
        self.key_emotions = {
            'C major':  {'joy': 0.8, 'triumph': 0.7, 'serenity': 0.6},
            'G major':  {'joy': 0.7, 'excitement': 0.6, 'triumph': 0.5},
            'D major':  {'triumph': 0.8, 'excitement': 0.7, 'joy': 0.6},
            'A major':  {'excitement': 0.8, 'joy': 0.6, 'triumph': 0.5},
            'E major':  {'excitement': 0.7, 'triumph': 0.6, 'tension': 0.5},
            'B major':  {'tension': 0.7, 'excitement': 0.6, 'triumph': 0.5},
            'F# major': {'tension': 0.8, 'excitement': 0.7, 'longing': 0.6},
            'C# major': {'tension': 0.7, 'longing': 0.6, 'excitement': 0.5},
            'F major':  {'serenity': 0.8, 'joy': 0.6, 'tenderness': 0.5},
            'Bb major': {'serenity': 0.7, 'contemplation': 0.6, 'tenderness': 0.5},
            'Eb major': {'contemplation': 0.8, 'serenity': 0.6, 'melancholy': 0.5},
            'Ab major': {'melancholy': 0.7, 'contemplation': 0.6, 'longing': 0.5},
            
            'A minor':  {'melancholy': 0.8, 'sadness': 0.7, 'longing': 0.6},
            'E minor':  {'sadness': 0.7, 'melancholy': 0.6, 'tension': 0.5},
            'B minor':  {'tension': 0.8, 'sadness': 0.7, 'aggression': 0.6},
            'F# minor': {'aggression': 0.7, 'tension': 0.6, 'sadness': 0.5},
            'C# minor': {'tension': 0.8, 'aggression': 0.7, 'sadness': 0.6},
            'G# minor': {'aggression': 0.8, 'tension': 0.7, 'sadness': 0.6},
            'D# minor': {'tension': 0.7, 'sadness': 0.6, 'melancholy': 0.5},
            'D minor':  {'sadness': 0.8, 'melancholy': 0.7, 'tension': 0.5},
            'G minor':  {'melancholy': 0.7, 'sadness': 0.6, 'longing': 0.5},
            'C minor':  {'sadness': 0.7, 'tension': 0.6, 'melancholy': 0.5},
            'F minor':  {'melancholy': 0.8, 'sadness': 0.7, 'longing': 0.6},
            'Bb minor': {'sadness': 0.8, 'melancholy': 0.7, 'tension': 0.6}
        }
        
        # Color emotional associations
        self.color_emotions = {
            'red': {'excitement': 0.8, 'aggression': 0.7, 'passion': 0.9},
            'blue': {'serenity': 0.8, 'melancholy': 0.6, 'contemplation': 0.7},
            'yellow': {'joy': 0.9, 'excitement': 0.7, 'triumph': 0.6},
            'green': {'serenity': 0.7, 'contemplation': 0.6, 'relief': 0.8},
            'purple': {'contemplation': 0.7, 'tension': 0.6, 'longing': 0.8},
            'orange': {'excitement': 0.7, 'joy': 0.8, 'triumph': 0.6},
            'brown': {'melancholy': 0.6, 'contemplation': 0.7, 'serenity': 0.5},
            'black': {'tension': 0.8, 'aggression': 0.7, 'melancholy': 0.6},
            'white': {'serenity': 0.8, 'relief': 0.7, 'contemplation': 0.6}
        }
    
    def extract_midi_features(self, midi_data: Dict[str, Any]) -> Optional[MusicFeatures]:
        """Extract musical features from MIDI data."""
        try:
            # Extract basic features
            tempo = float(midi_data['tempo'])
            notes_per_second = float(midi_data['notes_per_second'])
            pitch_range = float(midi_data['pitch_range'])
            avg_pitch = float(midi_data['avg_pitch'])
            avg_velocity = float(midi_data['avg_velocity'])
            key = midi_data['key']
            duration = float(midi_data['total_duration'])
            
            # Get key characteristics
            key_chars = self.key_characteristics.get(key, self.key_characteristics['unknown'])
            
            # Normalize values
            tempo_normalized = np.clip((tempo - 40) / (208 - 40), 0, 1)
            density_normalized = min(notes_per_second / 15, 1)
            range_normalized = pitch_range / 88
            velocity_normalized = avg_velocity / 127
            
            # Calculate composite features
            energy = (
                tempo_normalized * 0.4 +
                density_normalized * 0.3 +
                velocity_normalized * 0.3
            )
            
            complexity = (
                range_normalized * 0.3 +
                density_normalized * 0.3 +
                key_chars['complexity'] * 0.4
            )
            
            # Extract emotional features
            emotional = EmotionalFeatures(
                valence=0.5 * ('major' in key.lower()) + 0.3 * tempo_normalized + 0.2 * velocity_normalized,
                arousal=np.mean([tempo_normalized, density_normalized, velocity_normalized]),
                emotions=self._get_key_emotions(key),
                narrative_quality=complexity,
                emotional_intensity=energy,
                tension_progression=[energy]
            )
            
            return MusicFeatures(
                tempo=tempo,
                note_density=notes_per_second,
                pitch_range=pitch_range,
                avg_pitch=avg_pitch,
                avg_velocity=avg_velocity,
                key=key,
                duration=duration,
                energy=float(energy),
                complexity=float(complexity),
                brightness=key_chars['brightness'],
                warmth=key_chars['warmth'],
                tension=float(complexity * energy),
                midi_path=midi_data['extracted_path'],
                emotional=emotional
            )
            
        except Exception as e:
            logger.error(f"Error extracting MIDI features for {midi_data.get('extracted_path', 'unknown')}: {e}")
            return None

    def _get_key_emotions(self, key: str) -> Dict[str, float]:
        """Get emotional characteristics for a musical key."""
        base_emotions = {
            'joy': 0.0,
            'sadness': 0.0,
            'triumph': 0.0,
            'melancholy': 0.0,
            'tension': 0.0,
            'relief': 0.0,
            'longing': 0.0,
            'serenity': 0.0
        }
        
        # Define emotional characteristics for keys
        if 'major' in key.lower():
            base_emotions.update({
                'joy': 0.7,
                'triumph': 0.6,
                'relief': 0.5
            })
        else:
            base_emotions.update({
                'melancholy': 0.7,
                'sadness': 0.6,
                'longing': 0.5
            })
        
        return base_emotions
    
    def extract_image_features(self, image_path: str) -> Optional[ImageFeatures]:
        """Extract visual features from image."""
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # Calculate average color and characteristics
            avg_color = img_array.mean(axis=(0,1)) / 255.0
            
            # Extract dominant colors using k-means clustering
            resized = cv2.resize(img_array, (50, 50))
            pixels = resized.reshape(-1, 3).astype(np.float32) / 255.0
            
            n_colors = 5
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
            flags = cv2.KMEANS_RANDOM_CENTERS
            _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
            
            # Sort colors by frequency
            unique_labels, counts = np.unique(labels, return_counts=True)
            sorted_indices = np.argsort(-counts)
            dominant_colors = [(float(r), float(g), float(b)) for r, g, b in palette[sorted_indices]]
            
            # Calculate image characteristics
            brightness = float(np.mean(avg_color))
            warmth = float((avg_color[0] + avg_color[1] - avg_color[2]) / 2)
            complexity = float(np.std(palette.reshape(-1)))
            
            # Energy from color intensity and contrast
            color_energy = np.mean(np.abs(np.diff(img_array.mean(axis=2))))
            energy = float(np.clip((brightness * 0.5 + color_energy * 0.5), 0, 1))
            
            # Extract emotional features
            hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv_img[:, :, 1]) / 255
            
            emotional = EmotionalFeatures(
                valence=brightness,
                arousal=saturation,
                emotions=self._get_color_emotions(dominant_colors),
                narrative_quality=complexity,
                emotional_intensity=energy,
                tension_progression=[energy]
            )
            
            return ImageFeatures(
                brightness=brightness,
                warmth=warmth,
                complexity=complexity,
                energy=energy,
                color_palette=[(float(r), float(g), float(b)) for r, g, b in palette],
                dominant_colors=dominant_colors,
                image_path=image_path,
                emotional=emotional
            )
            
        except Exception as e:
            logger.error(f"Error extracting image features for {image_path}: {e}")
            return None

    def _get_color_emotions(self, colors: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """Get emotional characteristics from colors."""
        base_emotions = {
            'joy': 0.0,
            'sadness': 0.0,
            'triumph': 0.0,
            'melancholy': 0.0,
            'tension': 0.0,
            'relief': 0.0,
            'longing': 0.0,
            'serenity': 0.0
        }
        
        for r, g, b in colors:
            # Warm colors
            if r > 0.6 and g < 0.6:
                base_emotions['joy'] += 0.2
                base_emotions['triumph'] += 0.2
            # Cool colors
            elif b > 0.6 and r < 0.6:
                base_emotions['melancholy'] += 0.2
                base_emotions['serenity'] += 0.2
            # Dark colors
            if max(r, g, b) < 0.3:
                base_emotions['tension'] += 0.2
                base_emotions['sadness'] += 0.2
        
        # Normalize emotions
        total = sum(base_emotions.values())
        if total > 0:
            base_emotions = {k: v/total for k, v in base_emotions.items()}
        
        return base_emotions

class FeatureExtractor:
    """Extract and cache features from MIDI files and images."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.emotional_analyzer = EmotionalAnalyzer()
        
        # Complete key characteristics mapping
        self.key_characteristics = {
            # Major keys - generally brighter and warmer
            'C major':  {'brightness': 0.80, 'warmth': 0.60, 'complexity': 0.40},
            'G major':  {'brightness': 0.85, 'warmth': 0.65, 'complexity': 0.45},
            'D major':  {'brightness': 0.90, 'warmth': 0.70, 'complexity': 0.50},
            'A major':  {'brightness': 0.85, 'warmth': 0.75, 'complexity': 0.55},
            'E major':  {'brightness': 0.80, 'warmth': 0.80, 'complexity': 0.60},
            'B major':  {'brightness': 0.75, 'warmth': 0.75, 'complexity': 0.65},
            'F# major': {'brightness': 0.70, 'warmth': 0.70, 'complexity': 0.70},
            'C# major': {'brightness': 0.65, 'warmth': 0.65, 'complexity': 0.75},
            'F major':  {'brightness': 0.75, 'warmth': 0.55, 'complexity': 0.45},
            'Bb major': {'brightness': 0.70, 'warmth': 0.50, 'complexity': 0.50},
            'Eb major': {'brightness': 0.65, 'warmth': 0.45, 'complexity': 0.55},
            'Ab major': {'brightness': 0.60, 'warmth': 0.40, 'complexity': 0.60},
            
            # Minor keys - generally darker and cooler
            'A minor':  {'brightness': 0.45, 'warmth': 0.35, 'complexity': 0.60},
            'E minor':  {'brightness': 0.50, 'warmth': 0.40, 'complexity': 0.65},
            'B minor':  {'brightness': 0.45, 'warmth': 0.45, 'complexity': 0.70},
            'F# minor': {'brightness': 0.40, 'warmth': 0.40, 'complexity': 0.75},
            'C# minor': {'brightness': 0.35, 'warmth': 0.35, 'complexity': 0.80},
            'G# minor': {'brightness': 0.30, 'warmth': 0.30, 'complexity': 0.85},
            'D# minor': {'brightness': 0.35, 'warmth': 0.25, 'complexity': 0.80},
            'D minor':  {'brightness': 0.50, 'warmth': 0.30, 'complexity': 0.65},
            'G minor':  {'brightness': 0.45, 'warmth': 0.35, 'complexity': 0.70},
            'C minor':  {'brightness': 0.40, 'warmth': 0.30, 'complexity': 0.75},
            'F minor':  {'brightness': 0.35, 'warmth': 0.25, 'complexity': 0.70},
            'Bb minor': {'brightness': 0.30, 'warmth': 0.20, 'complexity': 0.75},
            
            # Default for unknown keys
            'unknown':  {'brightness': 0.50, 'warmth': 0.50, 'complexity': 0.50}
        }
    
    def extract_midi_features(self, midi_data: Dict[str, Any]) -> Optional[MusicFeatures]:
        """Extract musical features from MIDI data."""
        try:
            # Extract basic features
            tempo = float(midi_data['tempo'])
            notes_per_second = float(midi_data['notes_per_second'])
            pitch_range = float(midi_data['pitch_range'])
            avg_pitch = float(midi_data['avg_pitch'])
            avg_velocity = float(midi_data['avg_velocity'])
            key = midi_data['key']
            duration = float(midi_data['total_duration'])
            
            # Get key characteristics
            key_chars = self.key_characteristics.get(key, self.key_characteristics['unknown'])
            
            # Normalize values
            tempo_normalized = np.clip((tempo - 40) / (208 - 40), 0, 1)
            density_normalized = min(notes_per_second / 15, 1)
            range_normalized = pitch_range / 88
            velocity_normalized = avg_velocity / 127
            
            # Calculate composite features
            energy = (
                tempo_normalized * 0.4 +
                density_normalized * 0.3 +
                velocity_normalized * 0.3
            )
            
            complexity = (
                range_normalized * 0.3 +
                density_normalized * 0.3 +
                key_chars['complexity'] * 0.4
            )
            
            # Calculate emotional features
            emotional = EmotionalFeatures(
                valence=0.5 * ('major' in key.lower()) + 0.3 * tempo_normalized + 0.2 * velocity_normalized,
                arousal=np.mean([tempo_normalized, density_normalized, velocity_normalized]),
                emotions=self.emotional_analyzer.key_emotions.get(key, {}),
                narrative_quality=complexity,
                emotional_intensity=energy,
                tension_progression=[energy]
            )
            
            return MusicFeatures(
                tempo=tempo,
                note_density=notes_per_second,
                pitch_range=pitch_range,
                avg_pitch=avg_pitch,
                avg_velocity=avg_velocity,
                key=key,
                duration=duration,
                energy=float(energy),
                complexity=float(complexity),
                brightness=key_chars['brightness'],
                warmth=key_chars['warmth'],
                tension=float(complexity * energy),
                midi_path=midi_data['extracted_path'],
                emotional=emotional
            )
            
        except Exception as e:
            logger.error(f"Error extracting MIDI features: {e}")
            return None
    
    def extract_image_features(self, image_path: str) -> Optional[ImageFeatures]:
        """Extract visual features from image."""
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # Calculate average color and characteristics
            avg_color = img_array.mean(axis=(0,1)) / 255.0
            
            # Extract dominant colors using k-means clustering
            resized = cv2.resize(img_array, (50, 50))
            pixels = resized.reshape(-1, 3).astype(np.float32) / 255.0
            
            n_colors = 5
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
            flags = cv2.KMEANS_RANDOM_CENTERS
            _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
            
            # Sort colors by frequency
            unique_labels, counts = np.unique(labels, return_counts=True)
            sorted_indices = np.argsort(-counts)
            dominant_colors = [(float(r), float(g), float(b)) for r, g, b in palette[sorted_indices]]
            
            # Calculate image characteristics
            brightness = float(np.mean(avg_color))
            warmth = float((avg_color[0] + avg_color[1] - avg_color[2]) / 2)
            complexity = float(np.std(palette.reshape(-1)))
            
            # Energy from color intensity and contrast
            color_energy = np.mean(np.abs(np.diff(img_array.mean(axis=2))))
            energy = float(np.clip((brightness * 0.5 + color_energy * 0.5), 0, 1))
            
            # Extract emotional features
            hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv_img[:, :, 1]) / 255
            
            # Get nearest named colors and their emotions
            color_emotions = {}
            for color in dominant_colors:
                nearest_color = self._get_nearest_named_color(color)
                if nearest_color in self.emotional_analyzer.color_emotions:
                    emotions = self.emotional_analyzer.color_emotions[nearest_color]
                    for emotion, score in emotions.items():
                        color_emotions[emotion] = max(color_emotions.get(emotion, 0), score)
            
            emotional = EmotionalFeatures(
                valence=brightness,
                arousal=saturation,
                emotions=color_emotions,
                narrative_quality=complexity,
                emotional_intensity=energy,
                tension_progression=[energy]
            )
            
            return ImageFeatures(
                brightness=brightness,
                warmth=warmth,
                complexity=complexity,
                energy=energy,
                color_palette=[(float(r), float(g), float(b)) for r, g, b in palette],
                dominant_colors=dominant_colors,
                image_path=image_path,
                emotional=emotional
            )
            
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            return None

    def _get_nearest_named_color(self, rgb: Tuple[float, float, float]) -> str:
        """Find the nearest named color for an RGB value."""
        named_colors = {
            'red': (1, 0, 0),
            'blue': (0, 0, 1),
            'yellow': (1, 1, 0),
            'green': (0, 1, 0),
            'purple': (0.5, 0, 0.5),
            'orange': (1, 0.5, 0),
            'brown': (0.6, 0.4, 0.2),
            'black': (0, 0, 0),
            'white': (1, 1, 1)
        }
        
        min_dist = float('inf')
        nearest_color = 'black'
        
        for name, value in named_colors.items():
            dist = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(rgb, value)))
            if dist < min_dist:
                min_dist = dist
                nearest_color = name
        
        return nearest_color

class FeaturePreprocessor:
    """Handle feature extraction and caching for both MIDI and images."""
    
    def __init__(self, cache_dir: str, num_workers: int = 32):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.feature_extractor = FeatureExtractor(cache_dir)
        self.num_workers = num_workers
        
    def process_midi_file(self, midi_data: Dict[str, Any]) -> Optional[MusicFeatures]:
        """Process a single MIDI file."""
        return self.feature_extractor.extract_midi_features(midi_data)
    
    def process_image_file(self, image_path: str) -> Optional[ImageFeatures]:
        """Process a single image file."""
        return self.feature_extractor.extract_image_features(image_path)
    
    def preprocess_all_midis(self, midi_df: pd.DataFrame) -> Dict[str, MusicFeatures]:
        """Preprocess all MIDI files with caching."""
        cache_path = self.cache_dir / 'midi_features.pkl'
        
        if cache_path.exists():
            logger.info("Loading cached MIDI features...")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        logger.info("Extracting MIDI features...")
        midi_features = {}
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_midi = {
                executor.submit(self.process_midi_file, row): row['extracted_path']
                for _, row in midi_df.iterrows()
            }
            
            for future in tqdm(future_to_midi, desc="Processing MIDIs", total=len(midi_df)):
                midi_path = future_to_midi[future]
                try:
                    features = future.result()
                    if features:
                        midi_features[midi_path] = features
                except Exception as e:
                    logger.error(f"Error processing MIDI {midi_path}: {e}")
        
        # Save cache
        with open(cache_path, 'wb') as f:
            pickle.dump(midi_features, f)
        
        logger.info(f"Processed {len(midi_features)} MIDI files successfully")
        return midi_features
    
    def preprocess_all_images(self, image_dir: str) -> Dict[str, ImageFeatures]:
        """Preprocess all images with caching."""
        cache_path = self.cache_dir / 'image_features.pkl'
        
        if cache_path.exists():
            logger.info("Loading cached image features...")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        logger.info("Extracting image features...")
        image_features = {}
        
        # Get all image paths
        image_paths = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        for root, _, files in os.walk(image_dir):
            for file in files:
                if Path(file).suffix.lower() in valid_extensions:
                    image_paths.append(os.path.join(root, file))
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_image = {
                executor.submit(self.process_image_file, path): path
                for path in image_paths
            }
            
            for future in tqdm(future_to_image, desc="Processing images", total=len(image_paths)):
                image_path = future_to_image[future]
                try:
                    features = future.result()
                    if features:
                        image_features[image_path] = features
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {e}")
        
        # Save cache
        with open(cache_path, 'wb') as f:
            pickle.dump(image_features, f)
        
        logger.info(f"Processed {len(image_features)} images successfully")
        return image_features

class MatchMaker:
    """Match MIDI files with images based on technical and emotional features."""
    
    def __init__(self, midi_features: Dict[str, MusicFeatures], 
                 image_features: Dict[str, ImageFeatures],
                 output_dir: str):
        self.midi_features = midi_features
        self.image_features = image_features
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Updated matching weights to include emotional aspects
        self.match_weights = {
            # Technical features (40% total)
            'technical': {
                'energy': 0.1,
                'complexity': 0.1,
                'brightness': 0.1,
                'warmth': 0.1
            },
            # Emotional features (60% total)
            'emotional': {
                'valence': 0.15,
                'arousal': 0.15,
                'emotion_category': 0.15,
                'narrative': 0.075,
                'intensity': 0.075
            }
        }
        
        # Keep existing period mappings
        self.period_mappings = {
            'baroque': {
                'brightness_range': (0.6, 0.8),
                'complexity_range': (0.7, 0.9),
                'preferred_styles': ['art_nouveau', 'baroque_art'],
                'emotional_bias': {
                    'contemplation': 0.2,
                    'triumph': 0.2
                }
            },
            'classical': {
                'brightness_range': (0.5, 0.7),
                'complexity_range': (0.5, 0.7),
                'preferred_styles': ['neoclassicism', 'romanticism'],
                'emotional_bias': {
                    'serenity': 0.2,
                    'joy': 0.2
                }
            },
            'romantic': {
                'brightness_range': (0.4, 0.7),
                'complexity_range': (0.6, 0.8),
                'preferred_styles': ['romanticism', 'symbolism'],
                'emotional_bias': {
                    'longing': 0.2,
                    'melancholy': 0.2
                }
            },
            'modern': {
                'brightness_range': (0.3, 0.8),
                'complexity_range': (0.5, 0.9),
                'preferred_styles': ['expressionism', 'surrealism'],
                'emotional_bias': {
                    'tension': 0.2,
                    'excitement': 0.2
                }
            }
        }
        
        # Initialize usage tracking
        self.image_usage = defaultdict(int)
        self.style_usage = defaultdict(int)
        self.max_image_uses = 2
        self.total_processed = 0
        self.save_interval = 100

    def detect_music_period(self, midi_path: str) -> str:
        """Detect the musical period based on the MIDI filename and characteristics."""
        filename = midi_path.lower()
        
        # Look for composer names and characteristics
        if any(name in filename for name in ['bach', 'handel', 'scarlatti', 'telemann']):
            return 'baroque'
        elif any(name in filename for name in ['mozart', 'haydn', 'clementi']):
            return 'classical'
        elif any(name in filename for name in ['chopin', 'liszt', 'schumann', 'brahms']):
            return 'romantic'
        elif any(name in filename for name in ['debussy', 'ravel', 'prokofiev']):
            return 'modern'
            
        # If no composer match, use musical characteristics
        music = self.midi_features[midi_path]
        if music.complexity > 0.7 and music.energy > 0.6:
            return 'baroque'
        elif 0.4 <= music.complexity <= 0.6:
            return 'classical'
        elif music.warmth > 0.6 and music.complexity > 0.6:
            return 'romantic'
        else:
            return 'modern'
    
    def detect_art_style(self, image_path: str) -> str:
        """Detect art style from image path and characteristics."""
        # Extract from path
        path_parts = Path(image_path).parts
        for part in path_parts:
            if part in ['art_nouveau', 'baroque_art', 'romanticism', 'symbolism', 
                       'expressionism', 'surrealism', 'neoclassicism']:
                return part
        return 'unknown'

    def compute_emotion_score(self, music: MusicFeatures, image: ImageFeatures) -> float:
        """Compute emotional matching score between music and image."""
        try:
            # Basic emotional feature matching
            valence_score = 1 - abs(music.emotional.valence - image.emotional.valence)
            arousal_score = 1 - abs(music.emotional.arousal - image.emotional.arousal)
            narrative_score = 1 - abs(music.emotional.narrative_quality - image.emotional.narrative_quality)
            intensity_score = 1 - abs(music.emotional.emotional_intensity - image.emotional.emotional_intensity)
            
            # Emotion category matching
            shared_emotions = set(music.emotional.emotions.keys()) & set(image.emotional.emotions.keys())
            if not shared_emotions:
                emotion_category_score = 0
            else:
                emotion_scores = []
                for emotion in shared_emotions:
                    music_score = music.emotional.emotions[emotion]
                    image_score = image.emotional.emotions[emotion]
                    emotion_scores.append(1 - abs(music_score - image_score))
                emotion_category_score = np.mean(emotion_scores)
            
            # Weight and combine scores
            w = self.match_weights['emotional']
            emotion_score = (
                w['valence'] * valence_score +
                w['arousal'] * arousal_score +
                w['emotion_category'] * emotion_category_score +
                w['narrative'] * narrative_score +
                w['intensity'] * intensity_score
            )
            
            return float(emotion_score)
            
        except Exception as e:
            logger.error(f"Error computing emotion score: {e}")
            return 0.0

    def compute_match_score(self, music: MusicFeatures, image: ImageFeatures, 
                          music_period: str, art_style: str) -> float:
        """Compute enhanced matching score between a MIDI file and an image."""
        try:
            # Technical feature matching
            energy_score = 1 - abs(music.energy - image.energy)
            complexity_score = 1 - abs(music.complexity - image.complexity)
            brightness_score = 1 - abs(music.brightness - image.brightness)
            warmth_score = 1 - abs(music.warmth - image.warmth)
            
            # Calculate technical score
            w = self.match_weights['technical']
            technical_score = (
                w['energy'] * energy_score +
                w['complexity'] * complexity_score +
                w['brightness'] * brightness_score +
                w['warmth'] * warmth_score
            )
            
            # Calculate emotional score
            emotional_score = self.compute_emotion_score(music, image)
            
            # Period and style compatibility bonus
            period_info = self.period_mappings.get(music_period, {})
            style_bonus = 0.1 if art_style in period_info.get('preferred_styles', []) else 0
            
            # Emotional bias bonus based on period
            emotional_bias = period_info.get('emotional_bias', {})
            bias_score = 0
            if emotional_bias:
                for emotion, weight in emotional_bias.items():
                    if emotion in music.emotional.emotions and emotion in image.emotional.emotions:
                        bias_score += weight * min(music.emotional.emotions[emotion],
                                                image.emotional.emotions[emotion])
            
            # Apply usage penalties
            image_penalty = self.image_usage[image.image_path] * 0.15
            style_penalty = self.style_usage[art_style] * 0.05
            
            # Combine all scores
            final_score = (
                technical_score +
                emotional_score +
                style_bonus +
                bias_score -
                image_penalty -
                style_penalty
            )
            
            return float(np.clip(final_score, 0, 1))
            
        except Exception as e:
            logger.error(f"Error computing match score: {e}")
            return 0.0

    def save_interim_matches(self, matches: List[Dict], force: bool = False):
        """Save interim matching results with enhanced statistics."""
        if not (self.total_processed % self.save_interval == 0 or force):
            return
            
        # Calculate current statistics
        art_style_counts = defaultdict(int)
        period_counts = defaultdict(int)
        emotion_matches = defaultdict(float)
        
        for match in matches:
            art_style_counts[match['art_style']] += 1
            period_counts[match['music_period']] += 1
            
            # Track emotional matching statistics
            if 'emotional_matches' in match:
                for emotion, strength in match['emotional_matches'].items():
                    emotion_matches[emotion] += strength
        
        # Average the emotion matches
        avg_emotion_matches = {
            emotion: score/len(matches) 
            for emotion, score in emotion_matches.items()
        }
        
        stats = {
            'total_matches': len(matches),
            'unique_midis': len(set(m['midi_path'] for m in matches)),
            'unique_images': len(set(m['image_path'] for m in matches)),
            'average_score': np.mean([m['score'] for m in matches]),
            'min_score': np.min([m['score'] for m in matches]),
            'max_score': np.max([m['score'] for m in matches]),
            'art_style_distribution': dict(art_style_counts),
            'music_period_distribution': dict(period_counts),
            'emotional_match_distribution': avg_emotion_matches,
            'timestamp': datetime.now().isoformat(),
            'processed_count': self.total_processed,
            'total_midi_files': len(self.midi_features)
        }
        
        results = {
            'matches': matches,
            'statistics': stats
        }
        
        # Save both interim and current files
        interim_path = self.output_dir / f'matches_interim_{self.total_processed}.json'
        current_path = self.output_dir / 'matches_current.json'
        
        for path in [interim_path, current_path]:
            with open(path, 'w') as f:
                json.dump(results, f, indent=2)
        
        logger.info(f"Saved interim results at count {self.total_processed}")
        logger.info(f"Current stats - Files processed: {self.total_processed}, "
                   f"Matches found: {stats['total_matches']}, "
                   f"Average score: {stats['average_score']:.3f}")

    def find_matches(self, min_score: float = 0.6) -> Dict[str, Any]:
        """Find optimal matches between MIDI files and images."""
        logger.info("Finding optimal matches with emotional matching...")
        
        matches = []
        unmatched_midis = []
        self.total_processed = 0
        
        # First pass: Try to match each MIDI optimally
        for midi_path, midi_feat in tqdm(self.midi_features.items(), desc="Matching MIDIs"):
            self.total_processed += 1
            
            music_period = self.detect_music_period(midi_path)
            best_score = min_score
            best_match = None
            best_emotions = None
            
            for img_path, img_feat in self.image_features.items():
                if self.image_usage[img_path] >= self.max_image_uses:
                    continue
                    
                art_style = self.detect_art_style(img_path)
                score = self.compute_match_score(midi_feat, img_feat, music_period, art_style)
                
                # Track matched emotions
                shared_emotions = {
                    emotion: min(midi_feat.emotional.emotions.get(emotion, 0),
                               img_feat.emotional.emotions.get(emotion, 0))
                    for emotion in midi_feat.emotional.emotions
                    if emotion in img_feat.emotional.emotions
                }
                
                if score > best_score:
                    best_score = score
                    best_match = (img_path, art_style, score)
                    best_emotions = shared_emotions
            
            if best_match:
                img_path, art_style, score = best_match
                matches.append({
                    'midi_path': midi_path,
                    'image_path': img_path,
                    'music_period': music_period,
                    'art_style': art_style,
                    'score': score,
                    'emotional_matches': best_emotions
                })
                self.image_usage[img_path] += 1
                self.style_usage[art_style] += 1
            else:
                unmatched_midis.append(midi_path)
            
            # Save interim results
            self.save_interim_matches(matches)
        
        # Second pass with relaxed constraints
        if unmatched_midis:
            logger.info(f"Attempting to match {len(unmatched_midis)} remaining MIDIs...")
            min_score *= 0.9
            
            for midi_path in tqdm(unmatched_midis, desc="Matching remaining MIDIs"):
                self.total_processed += 1
                
                midi_feat = self.midi_features[midi_path]
                music_period = self.detect_music_period(midi_path)
                best_score = min_score
                best_match = None
                best_emotions = None
                
                for img_path, img_feat in self.image_features.items():
                    if self.image_usage[img_path] >= self.max_image_uses + 1:
                        continue
                        
                    art_style = self.detect_art_style(img_path)
                    score = self.compute_match_score(midi_feat, img_feat, music_period, art_style)
                    
                    shared_emotions = {
                        emotion: min(midi_feat.emotional.emotions.get(emotion, 0),
                                   img_feat.emotional.emotions.get(emotion, 0))
                        for emotion in midi_feat.emotional.emotions
                        if emotion in img_feat.emotional.emotions
                    }
                    
                    if score > best_score:
                        best_score = score
                        best_match = (img_path, art_style, score)
                        best_emotions = shared_emotions
                
                if best_match:
                    img_path, art_style, score = best_match
                    matches.append({
                        'midi_path': midi_path,
                        'image_path': img_path,
                        'music_period': music_period,
                        'art_style': art_style,
                        'score': score,
                        'emotional_matches': best_emotions
                    })
                    self.image_usage[img_path] += 1
                    self.style_usage[art_style] += 1
                
                # Save interim results
                self.save_interim_matches(matches)
        
        # Force final save
        self.save_interim_matches(matches, force=True)
        
        logger.info("Matching complete!")
        return matches

def main():
    parser = argparse.ArgumentParser(description='Art-Music Matcher')
    parser.add_argument('--midi-csv', required=True, help='Path to MIDI features CSV')
    parser.add_argument('--image-dir', required=True, help='Directory containing artwork images')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--cache-dir', default='cache', help='Directory for feature caches')
    parser.add_argument('--min-score', type=float, default=0.5, help='Minimum matching score')
    parser.add_argument('--workers', type=int, default=32, help='Number of worker processes')
    parser.add_argument('--force-reprocess', action='store_true', help='Force reprocessing of features')
    args = parser.parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging to file
    log_path = Path(args.output_dir) / 'matcher.log'
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    try:
        # Load MIDI data
        logger.info(f"Loading MIDI features from {args.midi_csv}")
        midi_df = pd.read_csv(args.midi_csv)
        logger.info(f"Loaded {len(midi_df)} MIDI entries")
        
        # Initialize feature preprocessor
        preprocessor = FeaturePreprocessor(
            cache_dir=args.cache_dir,
            num_workers=args.workers
        )
        
        # Process features
        if args.force_reprocess:
            logger.info("Force reprocessing enabled - clearing cache")
            for cache_file in Path(args.cache_dir).glob('*.pkl'):
                cache_file.unlink()
        
        # Extract all features
        midi_features = preprocessor.preprocess_all_midis(midi_df)
        image_features = preprocessor.preprocess_all_images(args.image_dir)
        
        # Initialize and run matcher
        matcher = MatchMaker(
            midi_features=midi_features,
            image_features=image_features,
            output_dir=args.output_dir
        )
        
        # Find matches
        matches = matcher.find_matches(min_score=args.min_score)
        
        # Print summary
        print("\nMatching Summary:")
        print(f"Total matches: {matches['statistics']['total_matches']}")
        print(f"Unique MIDIs matched: {matches['statistics']['unique_midis']}")
        print(f"Unique images matched: {matches['statistics']['unique_images']}")
        print(f"Average match score: {matches['statistics']['average_score']:.3f}")
        print(f"Score range: {matches['statistics']['min_score']:.3f} - {matches['statistics']['max_score']:.3f}")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    exit(main())