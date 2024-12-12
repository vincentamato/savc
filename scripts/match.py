import os
import random
import torch
from tqdm import tqdm
from transformers import MllamaForConditionalGeneration, AutoProcessor
import cv2
import numpy as np
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
import google.generativeai as genai
from dataclasses import dataclass
from typing import Dict, List, Optional
import random
import pandas as pd
import datetime

#AIzaSyB4nyeHrOVyh7C0fIhbaajhVx5Wx3bNqgg
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

class ArtworkAnalyzer:
    def __init__(self, images_base_dir):
        self.images_base_dir = images_base_dir

        # self._setup_llama("meta-llama/Llama-3.2-11B-Vision-Instruct")
    
    def _setup_llama(self, model_id, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        print(f"Using device: {device}")
        self.llama_model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(device)
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        print(f"{model_id} loaded successfully")
    
    def _extract_color_features(self, img_rgb, img_hsv):
        features = {}
        
        # Color palette extraction
        pixels = img_rgb.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, n_init=10)
        kmeans.fit(pixels)
        palette = kmeans.cluster_centers_
        
        # Calculate color proportions
        labels = kmeans.labels_
        proportions = np.bincount(labels) / len(labels)
        
        features['dominant_colors'] = palette.tolist()
        features['color_proportions'] = proportions.tolist()
        
        # Color statistics
        features['color_saturation_mean'] = np.mean(img_hsv[:,:,1])
        features['color_saturation_std'] = np.std(img_hsv[:,:,1])
        features['color_value_mean'] = np.mean(img_hsv[:,:,2])
        features['color_value_std'] = np.std(img_hsv[:,:,2])
        
        # Color temperature (warm vs cool colors)
        hue = img_hsv[:,:,0]
        warm_mask = ((hue >= 0) & (hue <= 60)) | (hue >= 300)
        features['warm_color_ratio'] = np.mean(warm_mask)
        
        return features
    
    def _extract_texture_features(self, img_gray):
        features = {}
        
        # GLCM features
        glcm = graycomatrix(img_gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                           symmetric=True, normed=True)
        
        features['contrast'] = np.mean(graycoprops(glcm, 'contrast'))
        features['dissimilarity'] = np.mean(graycoprops(glcm, 'dissimilarity'))
        features['homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
        features['energy'] = np.mean(graycoprops(glcm, 'energy'))
        features['correlation'] = np.mean(graycoprops(glcm, 'correlation'))
        
        # Edge density
        edges = cv2.Canny(img_gray, 100, 200)
        features['edge_density'] = np.mean(edges > 0)
        
        return features
    
    def _extract_composition_features(self, img_gray):
        features = {}
        
        # Rule of thirds points
        h, w = img_gray.shape
        third_h, third_w = h // 3, w // 3
        
        thirds_points = [
            img_gray[third_h, third_w],
            img_gray[third_h, 2*third_w],
            img_gray[2*third_h, third_w],
            img_gray[2*third_h, 2*third_w]
        ]
        features['thirds_intensity'] = np.mean(thirds_points)
        
        # Balance measures
        left_half = img_gray[:, :w//2]
        right_half = img_gray[:, w//2:]
        top_half = img_gray[:h//2, :]
        bottom_half = img_gray[h//2:, :]
        
        features['horizontal_balance'] = abs(np.mean(left_half) - np.mean(right_half))
        features['vertical_balance'] = abs(np.mean(top_half) - np.mean(bottom_half))
        
        # Visual complexity (entropy)
        features['complexity'] = entropy(img_gray.flatten())
        
        return features
    
    def _extract_movement_features(self, img_gray):
        features = {}
        
        # Gradient-based movement estimation
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx)
        
        features['movement_intensity'] = np.mean(magnitude)
        features['movement_variance'] = np.std(magnitude)
        
        # Directional tendencies
        direction_hist, _ = np.histogram(direction, bins=8, range=(-np.pi, np.pi))
        features['movement_direction_distribution'] = direction_hist.tolist()
        
        return features
    
    def get_random_artbench_images(self, images_base_dir, num_images=5):
        style_paths = []
        
        for style_dir in os.listdir(images_base_dir):
            full_style_path = os.path.join(images_base_dir, style_dir)
            if os.path.isdir(full_style_path):
                style_paths.append(full_style_path)
        
        all_images = []
        for style_path in style_paths:
            style_images = [os.path.join(style_path, f) for f in os.listdir(style_path) 
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
            all_images.extend(style_images)
        
        selected_images = random.sample(all_images, min(num_images, len(all_images)))
        return selected_images
    
    def analyze_art_emotion(self, image_path):
        import google.generativeai as genai
        import base64
        
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
        with open(image_path, 'rb') as img_file:
            image_data = img_file.read()
        
        prompt = """Analyze this artwork's emotional resonance and expressive qualities. 

                Choose ONE primary emotion from this carefully curated set that best captures the artwork's dominant feeling:

                - Transcendent (spiritual, ethereal, sublime)
                - Melancholic (wistful, longing, contemplative)
                - Turbulent (dramatic, intense, passionate)
                - Euphoric (ecstatic, joyful, celebratory)
                - Somber (serious, grave, dignified)
                - Intimate (tender, gentle, nurturing)
                - Dynamic (energetic, forceful, vigorous)
                - Mysterious (enigmatic, dreamlike, uncertain)
                - Serene (peaceful, harmonious, balanced)
                - Unsettling (disquieting, anxious, tense)

                Also provide:
                1. Emotional Intensity (1-5): How strongly does the artwork convey this emotion?
                    1 = Very subtle
                    2 = Mild
                    3 = Moderate
                    4 = Strong
                    5 = Very intense

                2. Psychological Depth (1-5): How complex and layered is the emotional content?
                    1 = Simple, straightforward
                    2 = Somewhat nuanced
                    3 = Moderately complex
                    4 = Complex, multilayered
                    5 = Highly complex, profound

                Respond ONLY with these three values in exactly this format:
                emotion,intensity,depth

                Example responses:
                turbulent,4,3
                intimate,2,5
                mysterious,3,4"""

        try:
            response = model.generate_content([
                {
                    'mime_type': 'image/jpeg',
                    'data': base64.b64encode(image_data).decode('utf-8')
                },
                prompt
            ])
            
            response_text = response.text.strip()
            parts = [p.strip() for p in response_text.split(',')]
            
            if len(parts) >= 3:
                return {
                    'emotion': parts[0].lower(),
                    'intensity': float(parts[1]),
                    'depth': float(parts[2]),
                }
                    
        except Exception as e:
            print(f"Error analyzing image: {e}")
            print(f"Raw response: {response_text if 'response_text' in locals() else 'No response'}")
        
        return None
    
    def extract_visual_features(self, image_path):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = {}
        
        # 1. Color Features
        features.update(self._extract_color_features(img_rgb, img_hsv))
        
        # 2. Texture Features
        features.update(self._extract_texture_features(img_gray))
        
        # 3. Composition Features
        features.update(self._extract_composition_features(img_gray))
        
        # 4. Movement Features
        features.update(self._extract_movement_features(img_gray))
        
        return features

@dataclass
class MidiFeatures:
    emotion: str
    intensity: float
    depth: float
    movement: float
    texture_density: float
    color_warmth: float
    complexity: float
    secondary_emotion: str
    harmonic_complexity: float
    rhythmic_stability: float

class MidiAnalyzer:
    def __init__(self):
        # Enhanced emotional mappings with more nuanced criteria
        self.emotion_mappings = {
            'transcendent': {
                'pitch_range': (60, 75),
                'tempo': (60, 100),
                'velocity_range': (40, 80),
                'key_qualities': ['major'],
                'duration_profile': 'long',
                'rhythmic_stability': 'high',
                'harmonic_complexity': 'medium'
            },
            'melancholic': {
                'pitch_range': (45, 65),
                'tempo': (60, 90),
                'velocity_range': (40, 70),
                'key_qualities': ['minor'],
                'duration_profile': 'medium',
                'rhythmic_stability': 'medium',
                'harmonic_complexity': 'high'
            },
            'turbulent': {
                'pitch_range': (55, 80),
                'tempo': (140, 200),
                'velocity_range': (70, 127),
                'key_qualities': ['minor'],
                'duration_profile': 'short',
                'rhythmic_stability': 'low',
                'harmonic_complexity': 'high'
            },
            'euphoric': {
                'pitch_range': (60, 85),
                'tempo': (120, 180),
                'velocity_range': (80, 127),
                'key_qualities': ['major'],
                'duration_profile': 'medium',
                'rhythmic_stability': 'high',
                'harmonic_complexity': 'medium'
            },
            'somber': {
                'pitch_range': (40, 60),
                'tempo': (60, 100),
                'velocity_range': (40, 70),
                'key_qualities': ['minor'],
                'duration_profile': 'long',
                'rhythmic_stability': 'high',
                'harmonic_complexity': 'medium'
            },
            'intimate': {
                'pitch_range': (50, 70),
                'tempo': (60, 100),
                'velocity_range': (30, 60),
                'key_qualities': ['major', 'minor'],
                'duration_profile': 'medium',
                'rhythmic_stability': 'high',
                'harmonic_complexity': 'low'
            },
            'dynamic': {
                'pitch_range': (50, 90),
                'tempo': (120, 200),
                'velocity_range': (60, 127),
                'key_qualities': ['major', 'minor'],
                'duration_profile': 'short',
                'rhythmic_stability': 'low',
                'harmonic_complexity': 'high'
            },
            'mysterious': {
                'pitch_range': (40, 75),
                'tempo': (60, 120),
                'velocity_range': (30, 70),
                'key_qualities': ['minor'],
                'duration_profile': 'long',
                'rhythmic_stability': 'low',
                'harmonic_complexity': 'high'
            },
            'serene': {
                'pitch_range': (55, 75),
                'tempo': (60, 100),
                'velocity_range': (40, 70),
                'key_qualities': ['major'],
                'duration_profile': 'long',
                'rhythmic_stability': 'high',
                'harmonic_complexity': 'low'
            },
            'unsettling': {
                'pitch_range': (30, 85),
                'tempo': (80, 140),
                'velocity_range': (50, 90),
                'key_qualities': ['minor'],
                'duration_profile': 'short',
                'rhythmic_stability': 'low',
                'harmonic_complexity': 'high'
            }
        }

    def analyze_midi(self, midi_data: Dict) -> MidiFeatures:
        """Enhanced MIDI analysis with additional features"""
        
        # Extract basic features
        avg_pitch = midi_data['avg_pitch']
        pitch_range = midi_data['pitch_range']
        tempo = midi_data['tempo']
        velocity = midi_data['avg_velocity']
        notes_per_second = midi_data['notes_per_second']
        avg_duration = midi_data['avg_duration']
        
        # Calculate additional features
        harmonic_complexity = self._calculate_harmonic_complexity(
            pitch_range,
            notes_per_second,
            midi_data['key']
        )
        
        rhythmic_stability = self._calculate_rhythmic_stability(
            tempo,
            notes_per_second,
            avg_duration
        )
        
        # Get emotional analysis with primary and secondary emotions
        emotions = self._determine_emotions(
            avg_pitch,
            pitch_range,
            tempo,
            velocity,
            avg_duration,
            harmonic_complexity,
            rhythmic_stability,
            midi_data['key']
        )
        
        # Calculate enhanced features
        intensity = self._calculate_intensity(
            velocity,
            notes_per_second,
            pitch_range,
            rhythmic_stability
        )
        
        depth = self._calculate_depth(
            pitch_range,
            avg_duration,
            notes_per_second,
            harmonic_complexity
        )
        
        movement = self._calculate_movement_mapping(
            tempo,
            notes_per_second,
            pitch_range,
            rhythmic_stability
        )
        
        texture_density = self._calculate_texture_mapping(
            notes_per_second,
            avg_duration,
            harmonic_complexity
        )
        
        color_warmth = self._calculate_color_warmth(
            avg_pitch,
            velocity,
            midi_data['key'],
            harmonic_complexity
        )
        
        complexity = self._calculate_complexity(
            pitch_range,
            notes_per_second,
            avg_duration,
            harmonic_complexity,
            rhythmic_stability
        )
        
        return MidiFeatures(
            emotion=emotions['primary'],
            secondary_emotion=emotions['secondary'],
            intensity=intensity,
            depth=depth,
            movement=min(movement, 100),  # Cap movement at 100
            texture_density=texture_density,
            color_warmth=color_warmth,
            complexity=complexity,
            harmonic_complexity=harmonic_complexity,
            rhythmic_stability=rhythmic_stability
        )

    def _calculate_harmonic_complexity(self, pitch_range, notes_per_second, key):
        """Calculate harmonic complexity based on musical features"""
        # Normalize components
        range_factor = pitch_range / 88
        density_factor = min(notes_per_second / 15, 1)
        
        # Key complexity factor (minor keys typically more complex)
        key_factor = 0.8 if 'minor' in key.lower() else 0.6
        
        complexity = (range_factor * 0.4 + 
                     density_factor * 0.3 + 
                     key_factor * 0.3)
        
        return complexity

    def _calculate_rhythmic_stability(self, tempo, notes_per_second, avg_duration):
        """Calculate rhythmic stability (0 = unstable, 1 = very stable)"""
        # Fast tempo and high density suggest less stability
        tempo_factor = 1 - (tempo / 240)  # Normalize to typical max tempo
        density_factor = 1 - min(notes_per_second / 15, 1)
        duration_factor = min(avg_duration, 1)
        
        stability = (tempo_factor * 0.3 + 
                    density_factor * 0.4 + 
                    duration_factor * 0.3)
        
        return max(min(stability, 1), 0)  # Ensure result is between 0 and 1

    def _determine_emotions(self, avg_pitch, pitch_range, tempo, velocity, 
                          avg_duration, harmonic_complexity, rhythmic_stability, key):
        """Determine primary and secondary emotions"""
        scores = {}
        
        for emotion, criteria in self.emotion_mappings.items():
            score = 0
            
            # Pitch range match
            if criteria['pitch_range'][0] <= avg_pitch <= criteria['pitch_range'][1]:
                score += 2
                
            # Tempo match
            if criteria['tempo'][0] <= tempo <= criteria['tempo'][1]:
                score += 2
                
            # Velocity match
            if criteria['velocity_range'][0] <= velocity <= criteria['velocity_range'][1]:
                score += 1
                
            # Key quality match
            if any(quality in key.lower() for quality in criteria['key_qualities']):
                score += 2
                
            # Duration profile match
            if criteria['duration_profile'] == 'long' and avg_duration > 0.5:
                score += 1
            elif criteria['duration_profile'] == 'medium' and 0.2 <= avg_duration <= 0.5:
                score += 1
            elif criteria['duration_profile'] == 'short' and avg_duration < 0.2:
                score += 1
                
            # Rhythmic stability match
            if criteria['rhythmic_stability'] == 'high' and rhythmic_stability > 0.7:
                score += 1
            elif criteria['rhythmic_stability'] == 'medium' and 0.3 <= rhythmic_stability <= 0.7:
                score += 1
            elif criteria['rhythmic_stability'] == 'low' and rhythmic_stability < 0.3:
                score += 1
                
            # Harmonic complexity match
            if criteria['harmonic_complexity'] == 'high' and harmonic_complexity > 0.7:
                score += 1
            elif criteria['harmonic_complexity'] == 'medium' and 0.3 <= harmonic_complexity <= 0.7:
                score += 1
            elif criteria['harmonic_complexity'] == 'low' and harmonic_complexity < 0.3:
                score += 1
                
            scores[emotion] = score
            
        # Sort by score
        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'primary': sorted_emotions[0][0],
            'secondary': sorted_emotions[1][0]
        }

    def _calculate_intensity(self, velocity, notes_per_second, pitch_range, rhythmic_stability):
        """Enhanced intensity calculation with rhythmic factor"""
        velocity_norm = velocity / 127
        notes_density_norm = min(notes_per_second / 15, 1)
        range_norm = pitch_range / 88
        rhythmic_factor = 1 - rhythmic_stability  # Less stability = more intensity
        
        intensity = (velocity_norm * 0.3 + 
                    notes_density_norm * 0.3 + 
                    range_norm * 0.2 +
                    rhythmic_factor * 0.2)
        
        return 1 + (intensity * 4)

    def _calculate_depth(self, pitch_range, avg_duration, notes_per_second, harmonic_complexity):
        """Enhanced depth calculation including harmonic complexity"""
        range_norm = pitch_range / 88
        duration_norm = min(avg_duration / 2, 1)
        complexity_norm = 1 - min(notes_per_second / 15, 1)
        
        depth = (range_norm * 0.25 + 
                duration_norm * 0.25 + 
                complexity_norm * 0.25 +
                harmonic_complexity * 0.25)
        
        return 1 + (depth * 4)

    def _calculate_movement_mapping(self, tempo, notes_per_second, pitch_range, rhythmic_stability):
        """Enhanced movement calculation with rhythmic stability factor"""
        tempo_norm = tempo / 200
        density_norm = min(notes_per_second / 15, 1)
        range_norm = pitch_range / 88
        movement_factor = 1 - rhythmic_stability
        
        movement = (tempo_norm * 0.3 + 
                   density_norm * 0.3 + 
                   range_norm * 0.2 +
                   movement_factor * 0.2) * 100
        
        return movement

    def _calculate_texture_mapping(self, notes_per_second, avg_duration, harmonic_complexity):
        """Enhanced texture calculation including harmonic complexity"""
        density_norm = min(notes_per_second / 15, 1)
        duration_norm = 1 - min(avg_duration / 2, 1)
        
        texture = (density_norm * 0.4 + 
                  duration_norm * 0.3 +
                  harmonic_complexity * 0.3)
        
        return texture

    def _calculate_color_warmth(self, avg_pitch, velocity, key, harmonic_complexity):
        """Enhanced color warmth calculation"""
        key_warmth = 0.8 if 'major' in key.lower() else 0.3
        pitch_norm = (avg_pitch - 21) / (108 - 21)
        velocity_norm = velocity / 127
        
        # Complex harmonies tend to feel "cooler"
        harmony_factor = 1 - harmonic_complexity
        
        warmth = (key_warmth * 0.3 + 
                 pitch_norm * 0.3 + 
                 velocity_norm * 0.2 +
                 harmony_factor * 0.2)
        
        return warmth

    def _calculate_complexity(self, pitch_range, notes_per_second, avg_duration, 
                            harmonic_complexity, rhythmic_stability):
        """Enhanced complexity calculation including all factors"""
        range_norm = pitch_range / 88
        density_norm = min(notes_per_second / 15, 1)
        duration_complexity = 1 - min(avg_duration / 2, 1)
        rhythmic_complexity = 1 - rhythmic_stability
        
        complexity = (range_norm * 0.2 + 
                     density_norm * 0.2 + 
                     duration_complexity * 0.2 +
                     harmonic_complexity * 0.2 +
                     rhythmic_complexity * 0.2)
        
        return complexity
    
@dataclass
class MatchResult:
    artwork_path: str
    midi_path: str
    similarity_score: float
    feature_scores: Dict[str, float]
    emotional_match: Dict[str, float]

class ArtworkMusicMatcher:
    def __init__(self):
        # Adjusted weights without style component
        self.weights = {
            'emotional': 0.45,  # Increased from 0.4
            'movement': 0.20,   # Increased from 0.15
            'texture': 0.15,
            'color': 0.12,      # Slightly increased
            'complexity': 0.08   # Slightly increased
        }

        # Define emotional families
        self.emotion_families = {
            'contemplative': ['transcendent', 'melancholic', 'somber'],
            'energetic': ['dynamic', 'turbulent', 'euphoric'],
            'peaceful': ['serene', 'intimate'],
            'dark': ['mysterious', 'unsettling']
        }

    def _calculate_emotional_similarity(self, artwork_emotion: Dict, midi_features: MidiFeatures) -> Dict[str, float]:
        """Enhanced emotional similarity calculation"""
        # Check for exact emotion match
        if artwork_emotion['emotion'] == midi_features.emotion:
            emotion_match = 1.0
        elif artwork_emotion['emotion'] == midi_features.secondary_emotion:
            emotion_match = 0.8
        else:
            # Check for emotional family match
            artwork_family = None
            midi_family = None
            
            for family, emotions in self.emotion_families.items():
                if artwork_emotion['emotion'] in emotions:
                    artwork_family = family
                if midi_features.emotion in emotions:
                    midi_family = family
                    
            if artwork_family and artwork_family == midi_family:
                emotion_match = 0.6
            else:
                emotion_match = 0.2
        
        # Compare intensity and depth
        intensity_diff = abs(artwork_emotion['intensity'] - midi_features.intensity) / 5.0
        depth_diff = abs(artwork_emotion['depth'] - midi_features.depth) / 5.0
        
        # Calculate total emotional similarity
        total_emotional = (
            0.5 * emotion_match +
            0.25 * (1 - intensity_diff) +
            0.25 * (1 - depth_diff)
        )
        
        return {
            'emotion_match': emotion_match,
            'intensity_similarity': 1 - intensity_diff,
            'depth_similarity': 1 - depth_diff,
            'total': total_emotional
        }

    def _calculate_texture_similarity(self, visual_features: Dict, midi_features: MidiFeatures) -> float:
        """Improved texture calculation with bounds checking"""
        artwork_texture = max(0, min(1, (
            0.4 * visual_features['edge_density'] +
            0.3 * (1 - visual_features['homogeneity']) +
            0.3 * min(1.0, visual_features['contrast'] / 300.0)
        )))
        
        texture_diff = abs(artwork_texture - midi_features.texture_density)
        return max(0, 1 - texture_diff)

    def _calculate_movement_similarity(self, visual_features: Dict, midi_features: MidiFeatures) -> float:
        """Calculate movement similarity"""
        # Scale artwork movement intensity to 0-100 range for comparison
        artwork_movement = visual_features['movement_intensity']
        scaled_artwork_movement = (artwork_movement / visual_features['movement_variance']) * 50
        
        movement_diff = abs(scaled_artwork_movement - midi_features.movement) / 100.0
        return 1 - movement_diff

    def _calculate_color_similarity(self, visual_features: Dict, midi_features: MidiFeatures) -> float:
        """Calculate color similarity"""
        color_diff = abs(visual_features['warm_color_ratio'] - midi_features.color_warmth)
        return 1 - color_diff

    def _calculate_complexity_similarity(self, visual_features: Dict, midi_features: MidiFeatures) -> float:
        """Calculate complexity similarity"""
        artwork_complexity = visual_features['complexity'] / 12.0  # Normalize to 0-1 range
        complexity_diff = abs(artwork_complexity - midi_features.complexity)
        return 1 - complexity_diff

    def _adjust_weights(self, artwork_features: Dict, artwork_emotion: Dict) -> Dict[str, float]:
        """Dynamically adjust weights based on artwork characteristics"""
        weights = self.weights.copy()
        
        # Adjust for high emotional intensity
        if artwork_emotion['intensity'] > 4:
            weights['emotional'] *= 1.2
            weights['color'] *= 1.1
        
        # Adjust for highly detailed artwork
        if artwork_features['edge_density'] > 0.15:
            weights['texture'] *= 1.3
            weights['movement'] *= 0.9
        
        # Adjust for dynamic scenes
        if artwork_features['movement_intensity'] > 50:
            weights['movement'] *= 1.3
            weights['texture'] *= 0.9
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    def match(self, artwork_analysis: Dict, midi_analysis: MidiFeatures) -> MatchResult:
        """Calculate similarity between artwork and MIDI piece"""
        # Calculate emotional similarity
        emotional_scores = self._calculate_emotional_similarity(
            artwork_analysis['emotional'],
            midi_analysis
        )
        
        # Calculate feature similarities
        feature_scores = {
            'movement': self._calculate_movement_similarity(
                artwork_analysis['visual'],
                midi_analysis
            ),
            'texture': self._calculate_texture_similarity(
                artwork_analysis['visual'],
                midi_analysis
            ),
            'color': self._calculate_color_similarity(
                artwork_analysis['visual'],
                midi_analysis
            ),
            'complexity': self._calculate_complexity_similarity(
                artwork_analysis['visual'],
                midi_analysis
            )
        }
        
        # Get dynamic weights
        weights = self._adjust_weights(
            artwork_analysis['visual'],
            artwork_analysis['emotional']
        )
        
        # Calculate weighted total score
        total_score = (
            weights['emotional'] * emotional_scores['total'] +
            weights['movement'] * feature_scores['movement'] +
            weights['texture'] * feature_scores['texture'] +
            weights['color'] * feature_scores['color'] +
            weights['complexity'] * feature_scores['complexity']
        )
        
        return MatchResult(
            artwork_path=artwork_analysis['path'],
            midi_path=midi_analysis.midi_path,
            similarity_score=total_score,
            feature_scores=feature_scores,
            emotional_match=emotional_scores
        )

    def find_best_matches(self, artworks: List[Dict], midi_pieces: List[MidiFeatures], 
                         num_matches: int = 3) -> Dict[str, List[MatchResult]]:
        """Find the best matching pairs between artworks and MIDI pieces"""
        all_matches = {}
        
        for artwork in artworks:
            matches = []
            for midi in midi_pieces:
                match_result = self.match(artwork, midi)
                matches.append(match_result)
            
            # Sort matches by similarity score
            matches.sort(key=lambda x: x.similarity_score, reverse=True)
            all_matches[artwork['path']] = matches[:num_matches]
        
        return all_matches

import os
import json
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Tuple
import random
from tqdm import tqdm

@dataclass
class ArtworkData:
    path: str
    emotion: str
    intensity: float
    depth: float
    timestamp: str

@dataclass
class MatchResult:
    artwork_path: str
    midi_path: str
    similarity_score: float
    feature_scores: Dict[str, float]
    emotional_match: Dict[str, float]

class ArtworkMusicMatcher:
    def __init__(self):
        # Keep existing weights
        self.weights = {
            'emotional': 0.45,
            'movement': 0.20,
            'texture': 0.15,
            'color': 0.12,
            'complexity': 0.08
        }

        self.emotion_families = {
            'contemplative': ['transcendent', 'melancholic', 'somber'],
            'energetic': ['dynamic', 'turbulent', 'euphoric'],
            'peaceful': ['serene', 'intimate'],
            'dark': ['mysterious', 'unsettling']
        }

    def load_art_emotions(self, json_path: str) -> Tuple[Dict[str, ArtworkData], Dict]:
        """Load and parse the precomputed art emotions with nested JSON structure"""
        try:
            print(f"\nAttempting to load art emotions from: {json_path}")
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            metadata = data.get('metadata', {})
            results = data.get('results', {})
            
            print(f"Found {len(results)} entries in results")
            print(f"Metadata: {metadata}")
            
            valid_entries = {}
            for path, info in results.items():
                try:
                    if all(k in info for k in ['emotion', 'intensity', 'depth']):
                        valid_entries[path] = ArtworkData(**info)
                except Exception as e:
                    print(f"Error processing entry for {path}: {str(e)}")
            
            print(f"Successfully processed {len(valid_entries)} valid entries")
            return valid_entries, metadata
            
        except Exception as e:
            print(f"Error loading {json_path}: {str(e)}")
            return {}, {}

    def _calculate_emotional_similarity(self, artwork: ArtworkData, midi_features: MidiFeatures) -> Dict[str, float]:
        """Enhanced emotional similarity calculation using precomputed emotions"""
        # Check for exact emotion match
        if artwork.emotion == midi_features.emotion:
            emotion_match = 1.0
        elif artwork.emotion == midi_features.secondary_emotion:
            emotion_match = 0.8
        else:
            # Check for emotional family match
            artwork_family = None
            midi_family = None
            
            for family, emotions in self.emotion_families.items():
                if artwork.emotion in emotions:
                    artwork_family = family
                if midi_features.emotion in emotions:
                    midi_family = family
                    
            if artwork_family and artwork_family == midi_family:
                emotion_match = 0.6
            else:
                emotion_match = 0.2
        
        # Compare intensity and depth
        intensity_diff = abs(artwork.intensity - midi_features.intensity) / 5.0
        depth_diff = abs(artwork.depth - midi_features.depth) / 5.0
        
        return {
            'emotion_match': emotion_match,
            'intensity_similarity': 1 - intensity_diff,
            'depth_similarity': 1 - depth_diff,
            'total': (0.5 * emotion_match + 0.25 * (1 - intensity_diff) + 0.25 * (1 - depth_diff))
        }

    def find_matches_with_coverage(
        self, 
        artworks: Dict[str, ArtworkData], 
        midi_pieces: List[MidiFeatures], 
        min_matches_per_midi: int = 1,
        max_matches_per_artwork: int = 3
    ) -> Dict[str, List[MatchResult]]:
        """Find matches with optimized emotion-based grouping"""
        
        # Group MIDIs by primary emotion
        print("\nGrouping MIDIs by emotion...")
        midi_by_emotion = {}
        for midi in midi_pieces:
            if midi.emotion not in midi_by_emotion:
                midi_by_emotion[midi.emotion] = []
            midi_by_emotion[midi.emotion].append(midi)
        
        # Group artworks by emotion
        print("Grouping artworks by emotion...")
        art_by_emotion = {}
        for path, art in artworks.items():
            if art.emotion not in art_by_emotion:
                art_by_emotion[art.emotion] = []
            art_by_emotion[art.emotion].append((path, art))
        
        # Initialize results
        artwork_matches = {path: [] for path in artworks.keys()}
        midi_usage_count = {midi.midi_path: 0 for midi in midi_pieces}
        
        print("\nFinding matches by emotion groups...")
        # First pass: Match within same emotion groups
        for emotion in tqdm(art_by_emotion.keys(), desc="Processing emotion groups"):
            if emotion not in midi_by_emotion:
                continue
                
            current_artworks = art_by_emotion[emotion]
            current_midis = midi_by_emotion[emotion]
            
            for art_path, artwork in current_artworks:
                matches = []
                # Calculate similarity with all MIDIs of same emotion
                for midi in current_midis:
                    emotional_match = self._calculate_emotional_similarity(artwork, midi)
                    match_result = MatchResult(
                        artwork_path=art_path,
                        midi_path=midi.midi_path,
                        similarity_score=emotional_match['total'],
                        feature_scores={},
                        emotional_match=emotional_match
                    )
                    matches.append(match_result)
                
                # Sort and take top matches
                matches.sort(key=lambda x: x.similarity_score, reverse=True)
                top_matches = matches[:max_matches_per_artwork]
                
                artwork_matches[art_path].extend(top_matches)
                for match in top_matches:
                    midi_usage_count[match.midi_path] += 1
        
        # Second pass: Fill in missing matches from secondary emotions
        print("\nFilling in missing matches...")
        undermatched_artworks = [
            path for path, matches in artwork_matches.items()
            if len(matches) < max_matches_per_artwork
        ]
        
        for art_path in tqdm(undermatched_artworks, desc="Processing undermatched artworks"):
            artwork = artworks[art_path]
            needed_matches = max_matches_per_artwork - len(artwork_matches[art_path])
            
            # Get all MIDIs except those with same primary emotion
            other_midis = [
                midi for midi in midi_pieces
                if midi.emotion != artwork.emotion
            ]
            
            # Find best matches from other emotions
            matches = []
            for midi in other_midis:
                emotional_match = self._calculate_emotional_similarity(artwork, midi)
                match_result = MatchResult(
                    artwork_path=art_path,
                    midi_path=midi.midi_path,
                    similarity_score=emotional_match['total'],
                    feature_scores={},
                    emotional_match=emotional_match
                )
                matches.append(match_result)
            
            # Sort and take needed matches
            matches.sort(key=lambda x: x.similarity_score, reverse=True)
            additional_matches = matches[:needed_matches]
            
            artwork_matches[art_path].extend(additional_matches)
            for match in additional_matches:
                midi_usage_count[match.midi_path] += 1
        
        # Third pass: Ensure minimum MIDI coverage
        print("\nEnsuring minimum MIDI coverage...")
        underused_midis = [
            midi_path for midi_path, count in midi_usage_count.items()
            if count < min_matches_per_midi
        ]
        
        if underused_midis:
            print(f"Found {len(underused_midis)} underused MIDIs")
            for midi_path in tqdm(underused_midis, desc="Processing underused MIDIs"):
                # Find artwork with fewest matches that isn't already matched with this MIDI
                candidate_artworks = [
                    path for path, matches in artwork_matches.items()
                    if midi_path not in [m.midi_path for m in matches]
                ]
                
                if candidate_artworks:
                    art_path = min(
                        candidate_artworks,
                        key=lambda p: len(artwork_matches[p])
                    )
                    
                    # Find the corresponding MIDI features
                    midi_features = next(m for m in midi_pieces if m.midi_path == midi_path)
                    
                    # Calculate match result
                    emotional_match = self._calculate_emotional_similarity(
                        artworks[art_path], 
                        midi_features
                    )
                    
                    match_result = MatchResult(
                        artwork_path=art_path,
                        midi_path=midi_path,
                        similarity_score=emotional_match['total'],
                        feature_scores={},
                        emotional_match=emotional_match
                    )
                    
                    artwork_matches[art_path].append(match_result)
                    midi_usage_count[midi_path] += 1
        
        return artwork_matches
    
def save_matches_to_json(
    matches: Dict[str, List[MatchResult]], 
    art_emotions: Dict[str, ArtworkData],
    output_path: str
):
    """Save matches to a JSON file with metadata"""
    
    # Convert matches to serializable format
    output_data = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_artworks": len(matches),
            "total_midis_used": len({
                match.midi_path 
                for matches_list in matches.values() 
                for match in matches_list
            }),
            "average_matches_per_artwork": sum(
                len(m) for m in matches.values()
            ) / len(matches)
        },
        "matches": {}
    }
    
    # Convert each match to a dictionary
    for artwork_path, artwork_matches in matches.items():
        artwork = art_emotions[artwork_path]
        
        output_data["matches"][artwork_path] = {
            "artwork_info": {
                "emotion": artwork.emotion,
                "intensity": artwork.intensity,
                "depth": artwork.depth
            },
            "midi_matches": [
                {
                    "midi_path": match.midi_path,
                    "similarity_score": match.similarity_score,
                    "emotional_match": match.emotional_match
                }
                for match in artwork_matches
            ]
        }
    
    # Save to file
    print(f"\nSaving matches to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Successfully saved matches to {output_path}")

def main():
    # Initialize analyzers
    midi_analyzer = MidiAnalyzer()
    matcher = ArtworkMusicMatcher()
    
    # Load precomputed art emotions
    art_emotions, metadata = matcher.load_art_emotions('data/art_emotions.json')
    if not art_emotions:
        print("No valid art emotions loaded. Exiting.")
        return
    
    print(f"\nArt Emotion Statistics:")
    print(f"Total images processed: {metadata.get('total_images', 'unknown')}")
    print(f"Failed images: {metadata.get('failed_images', 'unknown')}")
    print(f"Valid emotions loaded: {len(art_emotions)}")
    
    # Get emotion distribution
    emotion_counts = {}
    for artwork in art_emotions.values():
        emotion_counts[artwork.emotion] = emotion_counts.get(artwork.emotion, 0) + 1
    
    print("\nEmotion Distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{emotion}: {count}")
    
    # Load and analyze MIDI files
    midi_metadata = pd.read_csv('data/midi_features.csv')
    print(f"\nProcessing {len(midi_metadata)} MIDI files...")
    
    midi_results = []
    for _, midi in tqdm(midi_metadata.iterrows(), desc="Analyzing MIDI files"):
        midi_dict = midi.to_dict()
        analysis = midi_analyzer.analyze_midi(midi_dict)
        analysis.midi_path = midi_dict['original_path']
        midi_results.append(analysis)
    
    # Find matches ensuring MIDI coverage
    matches = matcher.find_matches_with_coverage(
        art_emotions,
        midi_results,
        min_matches_per_midi=1,
        max_matches_per_artwork=3
    )
    
    # Calculate and display statistics
    used_midis = set()
    match_counts = []
    
    for artwork_matches in matches.values():
        match_counts.append(len(artwork_matches))
        for match in artwork_matches:
            used_midis.add(match.midi_path)
    
    print("\nMatching Statistics:")
    print(f"Total artworks matched: {len(matches)}")
    print(f"Unique MIDIs used: {len(used_midis)}")
    print(f"Average matches per artwork: {sum(match_counts) / len(match_counts):.2f}")
    
    # Display some sample matches
    print("\nSample Matches:")
    sample_artworks = random.sample(list(matches.keys()), min(5, len(matches)))
    
    for art_path in sample_artworks:
        artwork = art_emotions[art_path]
        print(f"\nArtwork: {os.path.basename(art_path)}")
        print(f"Emotion: {artwork.emotion} (intensity: {artwork.intensity}, depth: {artwork.depth})")
        
        for i, match in enumerate(matches[art_path], 1):
            print(f"\n{i}. MIDI: {os.path.basename(match.midi_path)}")
            print(f"   Similarity Score: {match.similarity_score:.3f}")
            print(f"   Emotional Match Details:")
            for aspect, score in match.emotional_match.items():
                if aspect != 'total':
                    print(f"   - {aspect.replace('_', ' ').title()}: {score:.3f}")

    output_path = 'data/matches.json'
    save_matches_to_json(matches, art_emotions, output_path)
    print(f"\nMatches saved to {output_path}")

if __name__ == "__main__":
    main()

def test_image_analysis():
    analyzer = ArtworkAnalyzer("data/artbench")
    images = analyzer.get_random_artbench_images(analyzer.images_base_dir)
    
    results = []
    for image_path in tqdm(images, desc="Analyzing images", unit="image"):
        emotional_result = analyzer.analyze_art_emotion(image_path)
        visual_features = analyzer.extract_visual_features(image_path)
        
        results.append({
            'path': image_path,
            'emotional': emotional_result,
            'visual': visual_features
        })
    
    print("\nAnalysis Results:")
    for result in results:
        print("\n" + "="*80)
        print(f"Image: {result['path']}")
        
        print("\nEmotional Analysis:")
        if result['emotional']:
            print(f"  Primary Emotion: {result['emotional']['emotion']}")
            print(f"  Intensity: {result['emotional']['intensity']:.1f}")
            print(f"  Depth: {result['emotional']['depth']:.1f}")
        
        print("\nVisual Features:")
        visual = result['visual']
        
        print("\nColor Features:")
        print(f"  Warm Color Ratio: {visual['warm_color_ratio']:.3f}")
        print(f"  Saturation (mean/std): {visual['color_saturation_mean']:.3f}/{visual['color_saturation_std']:.3f}")
        print(f"  Value (mean/std): {visual['color_value_mean']:.3f}/{visual['color_value_std']:.3f}")
        
        print("\nTexture Features:")
        print(f"  Contrast: {visual['contrast']:.3f}")
        print(f"  Homogeneity: {visual['homogeneity']:.3f}")
        print(f"  Energy: {visual['energy']:.3f}")
        print(f"  Edge Density: {visual['edge_density']:.3f}")
        
        print("\nComposition Features:")
        print(f"  Thirds Intensity: {visual['thirds_intensity']:.3f}")
        print(f"  Horizontal Balance: {visual['horizontal_balance']:.3f}")
        print(f"  Vertical Balance: {visual['vertical_balance']:.3f}")
        print(f"  Complexity: {visual['complexity']:.3f}")
        
        print("\nMovement Features:")
        print(f"  Movement Intensity: {visual['movement_intensity']:.3f}")
        print(f"  Movement Variance: {visual['movement_variance']:.3f}")
    
def test_music_analysis():
    # Read the MIDI metadata CSV
    midi_metadata = pd.read_csv('data/midi_features.csv')
    
    # Select 10 random MIDI files
    random_midis = midi_metadata.sample(n=10)
    
    # Initialize the analyzer
    analyzer = MidiAnalyzer()
    
    print("\nAnalyzing MIDI Files:")
    print("="*80)
    
    for _, midi in tqdm(random_midis.iterrows(), total=10, desc="Processing MIDI files"):
        midi_dict = midi.to_dict()
        filename = os.path.basename(midi_dict['original_path'])
        analysis = analyzer.analyze_midi(midi_dict)
        
        print("\n" + "="*80)
        print(f"File: {filename}")
        
        print("\nMusical Features:")
        print(f"  Notes: {midi_dict['total_notes']}")
        print(f"  Tempo: {midi_dict['tempo']:.1f} BPM")
        print(f"  Key: {midi_dict['key']}")
        print(f"  Average Pitch: {midi_dict['avg_pitch']:.1f}")
        print(f"  Pitch Range: {midi_dict['pitch_range']}")
        print(f"  Notes per Second: {midi_dict['notes_per_second']:.2f}")
        print(f"  Average Duration: {midi_dict['avg_duration']:.3f}")
        
        print("\nEmotional Analysis:")
        print(f"  Primary Emotion: {analysis.emotion}")
        print(f"  Secondary Emotion: {analysis.secondary_emotion}")
        print(f"  Intensity: {analysis.intensity:.1f}/5")
        print(f"  Depth: {analysis.depth:.1f}/5")
        
        print("\nMusical Character:")
        print(f"  Harmonic Complexity: {analysis.harmonic_complexity:.3f}")
        print(f"  Rhythmic Stability: {analysis.rhythmic_stability:.3f}")
        
        print("\nVisual Feature Mappings:")
        print(f"  Movement: {analysis.movement:.1f}/100")
        print(f"  Texture Density: {analysis.texture_density:.3f}")
        print(f"  Color Warmth: {analysis.color_warmth:.3f}")
        print(f"  Overall Complexity: {analysis.complexity:.3f}")