import torch
import math
from torch import nn
import torch.nn.functional as F
import numpy as np
import traceback

class ChordProgressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.common_progressions = {
            'major': [(0,4,7), (5,9,0), (7,11,2), (0,4,7)],  # I-V-vii-I
            'minor': [(0,3,7), (5,8,0), (7,10,2), (0,3,7)]   # i-v-VII-i
        }
        
    def forward(self, harmony_features):
        chord_predictions, chord_targets = harmony_features
        
        # Handle empty tensors
        if chord_predictions.nelement() == 0 or chord_targets.nelement() == 0:
            return torch.tensor(0.0, device=chord_predictions.device)
        
        # Debug shape information
        # print(f"\nHarmony Feature Shapes:")
        # print(f"Predictions: {chord_predictions.shape}")
        # print(f"Targets: {chord_targets.shape}")
        
        # Align sequence lengths
        min_len = min(chord_predictions.size(0), chord_targets.size(0))
        chord_predictions = chord_predictions[:min_len]
        chord_targets = chord_targets[:min_len]
        
        # Calculate chord similarity
        similarity = self.chord_similarity(chord_predictions, chord_targets)
        
        # Penalize uncommon progressions
        progression_penalty = self.progression_penalty(chord_predictions)
        
        return similarity + 0.5 * progression_penalty
        
    def chord_similarity(self, pred_chords, target_chords):
        """Compute similarity between predicted and target chords with alignment"""
        if pred_chords.nelement() == 0 or target_chords.nelement() == 0:
            return torch.tensor(0.0, device=pred_chords.device)
            
        # Ensure same batch size
        batch_size = min(pred_chords.size(0), target_chords.size(0))
        pred_chords = pred_chords[:batch_size]
        target_chords = target_chords[:batch_size]
            
        # Convert to pitch class distributions
        pred_dist = torch.zeros((batch_size, 12), device=pred_chords.device)
        target_dist = torch.zeros((batch_size, 12), device=target_chords.device)
        
        # Fill distributions with improved numerical stability
        for i in range(batch_size):
            pred_notes = pred_chords[i] % 12
            pred_dist[i].scatter_(0, pred_notes.long(), torch.ones_like(pred_notes, dtype=torch.float))
            target_notes = target_chords[i] % 12
            target_dist[i].scatter_(0, target_notes.long(), torch.ones_like(target_notes, dtype=torch.float))
        
        # Add small constant before normalization to prevent division by zero
        eps = 1e-6
        pred_dist = pred_dist + eps
        target_dist = target_dist + eps
        
        # Normalize distributions
        pred_dist = pred_dist / pred_dist.sum(dim=-1, keepdim=True)
        target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True)
        
        # Compute similarity with numerical stability
        similarity = torch.sum(pred_dist * target_dist, dim=-1)
        similarity = torch.clamp(similarity, min=0.0, max=1.0)
        
        return 1 - similarity.mean()
    
    def progression_penalty(self, chord_sequence):
        """Penalize uncommon chord progressions"""
        if chord_sequence.nelement() < 4:
            return torch.tensor(0.0, device=chord_sequence.device)
            
        penalties = []
        for i in range(len(chord_sequence) - 3):
            window = chord_sequence[i:i+4]
            
            # Convert notes to pitch classes
            window = window % 12
            
            # Check if progression matches common patterns
            major_penalty = self.progression_distance(window, self.common_progressions['major'])
            minor_penalty = self.progression_distance(window, self.common_progressions['minor'])
            
            penalties.append(min(major_penalty, minor_penalty))
        
        if not penalties:
            return torch.tensor(0.0, device=chord_sequence.device)
            
        return torch.tensor(penalties, device=chord_sequence.device).mean()
        
    def progression_distance(self, progression, template):
        """Compute distance between a progression and a template"""
        distance = 0
        for i in range(min(len(progression), len(template))):
            prog_notes = set(progression[i].cpu().numpy())
            temp_notes = set(template[i])
            
            # Compute Jaccard distance
            intersection = len(prog_notes.intersection(temp_notes))
            union = len(prog_notes.union(temp_notes))
            distance += 1 - (intersection / (union + 1e-8))
            
        return distance / len(template)

class RhythmConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.beat_weights = torch.tensor([1.0, 0.5, 0.75, 0.5])  # Weights for 4/4 time
        
    def forward(self, rhythm_features):
        onset_times, durations = rhythm_features
        
        # Handle empty sequences
        if len(onset_times) == 0:
            return torch.tensor(0.0, device=self._get_device(rhythm_features))
        
        # Ensure tensors are on the correct device
        beat_weights = self.beat_weights.to(onset_times.device)
        
        # Beat consistency loss with safety checks
        beat_consistency = self.beat_alignment_loss(onset_times)
        
        # Duration patterns loss with safety checks
        duration_consistency = self.duration_pattern_loss(durations)
        
        # Combine rhythm losses with safety checks
        return torch.nan_to_num(beat_consistency + 0.5 * duration_consistency, 0.0)
    
    def beat_alignment_loss(self, onset_times):
        """Penalize notes that don't align with strong beats"""
        if len(onset_times) == 0:
            return torch.tensor(0.0, device=onset_times.device)
            
        try:
            # Convert to tensor if not already
            if not isinstance(onset_times, torch.Tensor):
                onset_times = torch.tensor(onset_times, device=self.beat_weights.device)
            
            # Compute beat positions safely
            beat_positions = (onset_times % 4).long().clamp(0, 3)  # Clamp to valid indices
            weights = self.beat_weights.to(onset_times.device)[beat_positions]
            
            # Safe mean computation
            mean_weight = weights.mean() if len(weights) > 0 else torch.tensor(1.0, device=weights.device)
            return torch.nan_to_num(1 - mean_weight, 0.0)
            
        except Exception as e:
            print(f"Warning in beat_alignment_loss: {str(e)}")
            return torch.tensor(0.0, device=onset_times.device)
    
    def duration_pattern_loss(self, durations):
        """Encourage consistent duration patterns"""
        if not isinstance(durations, torch.Tensor):
            return torch.tensor(0.0, device=self._get_device((durations,)))
            
        if len(durations) < 2:
            return torch.tensor(0.0, device=durations.device)
            
        try:
            # Add small epsilon to prevent division by zero
            eps = 1e-7
            padded_durations = durations + eps
            
            # Calculate variance in duration ratios safely
            duration_ratios = padded_durations[1:] / padded_durations[:-1]
            
            # Remove any infinite values that might have occurred
            duration_ratios = torch.nan_to_num(duration_ratios, 0.0, 0.0, 0.0)
            
            # Compute variance with safety checks
            if len(duration_ratios) == 0:
                return torch.tensor(0.0, device=durations.device)
                
            mean = duration_ratios.mean()
            squared_diff = (duration_ratios - mean) ** 2
            variance = squared_diff.mean()
            
            # Bound the result
            return torch.tanh(torch.nan_to_num(variance, 0.0))
            
        except Exception as e:
            print(f"Warning in duration_pattern_loss: {str(e)}")
            return torch.tensor(0.0, device=durations.device)
    
    def _get_device(self, features):
        """Helper to determine the device to use"""
        if isinstance(features, tuple):
            for f in features:
                if isinstance(f, torch.Tensor):
                    return f.device
        elif isinstance(features, torch.Tensor):
            return features.device
        return 'cpu'

class MelodicContourLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_interval = 12  # Maximum preferred interval size
        
    def forward(self, contour_features):
        pitch_sequence, intervals = contour_features
        
        # Handle empty sequences
        if not isinstance(pitch_sequence, torch.Tensor) or len(pitch_sequence) == 0:
            return torch.tensor(0.0, device=self._get_device(contour_features))
            
        try:
            # Interval size loss with safety checks
            interval_loss = self.interval_size_loss(intervals)
            
            # Contour smoothness loss with safety checks
            smoothness_loss = self.contour_smoothness_loss(pitch_sequence)
            
            # Phrase arch loss with safety checks
            arch_loss = self.phrase_arch_loss(pitch_sequence)
            
            # Combine losses with nan checks
            total_loss = interval_loss + 0.3 * smoothness_loss + 0.3 * arch_loss
            return torch.nan_to_num(total_loss, 0.0)
            
        except Exception as e:
            print(f"Warning in MelodicContourLoss: {str(e)}")
            return torch.tensor(0.0, device=self._get_device(contour_features))
    
    def interval_size_loss(self, intervals):
        """Penalize large melodic intervals"""
        if not isinstance(intervals, torch.Tensor) or len(intervals) == 0:
            return torch.tensor(0.0, device=self._get_device((intervals,)))
            
        return torch.mean(torch.clamp(torch.abs(intervals) - self.max_interval, min=0))
    
    def contour_smoothness_loss(self, pitch_sequence):
        """Encourage smooth melodic contours"""
        if len(pitch_sequence) < 3:
            return torch.tensor(0.0, device=pitch_sequence.device)
            
        try:
            # Calculate derivatives with safety checks
            first_derivative = pitch_sequence[1:] - pitch_sequence[:-1]
            second_derivative = first_derivative[1:] - first_derivative[:-1]
            
            # Handle potential NaN values
            second_derivative = torch.nan_to_num(second_derivative, 0.0)
            
            return torch.mean(torch.abs(second_derivative))
            
        except Exception as e:
            print(f"Warning in contour_smoothness_loss: {str(e)}")
            return torch.tensor(0.0, device=pitch_sequence.device)
    
    def phrase_arch_loss(self, pitch_sequence):
        """Encourage natural phrase arches"""
        if len(pitch_sequence) < 4:
            return torch.tensor(0.0, device=pitch_sequence.device)
            
        try:
            # Create the ideal arch shape
            x = torch.linspace(-1, 1, len(pitch_sequence), device=pitch_sequence.device)
            target_arch = -x**2
            
            # Normalize pitch sequence safely
            mean = pitch_sequence.mean()
            std = pitch_sequence.std()
            if std == 0:
                std = 1.0
            normalized_pitch = (pitch_sequence - mean) / std
            
            return F.mse_loss(normalized_pitch, target_arch)
            
        except Exception as e:
            print(f"Warning in phrase_arch_loss: {str(e)}")
            return torch.tensor(0.0, device=pitch_sequence.device)
    
    def _get_device(self, features):
        """Helper to determine the device to use"""
        if isinstance(features, tuple):
            for f in features:
                if isinstance(f, torch.Tensor):
                    return f.device
        elif isinstance(features, torch.Tensor):
            return features.device
        return 'cpu'

class MusicGenerationLoss(nn.Module):
    def __init__(self, vocab_size, pad_token_id, label_smoothing=0.15):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing
        
        # Core losses
        self.token_criterion = nn.CrossEntropyLoss(
            ignore_index=pad_token_id, 
            label_smoothing=label_smoothing
        )
        
        # Musical structure losses
        self.harmony_loss = ChordProgressionLoss()
        self.rhythm_loss = RhythmConsistencyLoss()
        self.contour_loss = MelodicContourLoss()
        
        # Token type offsets
        self.NOTE_ON_OFFSET = 0      # NOTE_ON tokens start at 0
        self.NOTE_OFF_OFFSET = 128   # NOTE_OFF tokens start at 128
        self.VELOCITY_OFFSET = 256   # Velocity tokens start at 256
        self.TIME_SHIFT_OFFSET = 384 # Start of time shift tokens
        
        self.pad_token_id = 388      # Adjust based on your vocabulary size
        self.time_step = 0.01        # Adjust based on your time quantization
        
        # Initialize smoothed weights
        self.harmony_weight = 0.3
        self.rhythm_weight = 0.3
        self.contour_weight = 0.2
        
        # Register as buffers so they persist between forward passes
        self.register_buffer('harmony_weight_smooth', torch.tensor(0.3))
        self.register_buffer('rhythm_weight_smooth', torch.tensor(0.3))
        self.register_buffer('contour_weight_smooth', torch.tensor(0.2))
        
    def forward(self, predictions, targets, attention_mask=None):
        """Forward pass returning loss dictionary"""
        # Handle tuple output from the model
        if isinstance(predictions, tuple):
            predictions, token_types = predictions
        
        device = predictions.device
        
        # Basic token loss
        token_loss = self.token_criterion(
            predictions.view(-1, self.vocab_size),
            targets.view(-1)
        )
        
        # Extract musical features
        harmony_features = self.extract_harmony_features(predictions, targets)
        rhythm_features = self.extract_rhythm_features(predictions, targets)
        contour_features = self.extract_contour_features(predictions, targets)
        
        # Initialize losses with small epsilon to avoid pure zeros
        epsilon = 1e-8
        harmony_loss = torch.tensor(epsilon, device=device)
        rhythm_loss = torch.tensor(epsilon, device=device)
        contour_loss = torch.tensor(epsilon, device=device)
        
        # Calculate individual losses if features exist
        if harmony_features is not None and len(harmony_features[0]) > 0:
            harmony_loss = self.harmony_loss(harmony_features)
        
        if rhythm_features is not None and len(rhythm_features[0]) > 0:
            rhythm_loss = self.rhythm_loss(rhythm_features)
        
        if contour_features is not None and len(contour_features[0]) > 0:
            contour_loss = self.contour_loss(contour_features)
        
        # Update smoothed weights using exponential moving average
        self.harmony_weight_smooth = 0.95 * self.harmony_weight_smooth + 0.05 * self.get_curriculum_weight(harmony_loss)
        self.rhythm_weight_smooth = 0.95 * self.rhythm_weight_smooth + 0.05 * self.get_curriculum_weight(rhythm_loss)
        self.contour_weight_smooth = 0.95 * self.contour_weight_smooth + 0.05 * self.get_curriculum_weight(contour_loss)
        
        # Calculate total loss with smoothed weights
        total_loss = token_loss
        
        if not torch.isnan(harmony_loss):
            total_loss += self.harmony_weight_smooth * harmony_loss
        if not torch.isnan(rhythm_loss):
            total_loss += self.rhythm_weight_smooth * rhythm_loss
        if not torch.isnan(contour_loss):
            total_loss += self.contour_weight_smooth * contour_loss
        
        return {
            'token_loss': token_loss,
            'harmony_loss': harmony_loss,
            'rhythm_loss': rhythm_loss,
            'contour_loss': contour_loss,
            'total_loss': total_loss
        }
    
    # def _debug_features(self, name, features):
    #     """Helper to debug extracted features"""
    #     print(f"\n{name} Features:")
    #     if isinstance(features, tuple):
    #         for i, feat in enumerate(features):
    #             if isinstance(feat, torch.Tensor):
    #                 print(f"  Feature {i}: shape={feat.shape}, "
    #                       f"non-empty={feat.numel() > 0}, "
    #                       f"range=[{feat.min().item() if feat.numel() > 0 else 'N/A'}, "
    #                       f"{feat.max().item() if feat.numel() > 0 else 'N/A'}]")
    #             else:
    #                 print(f"  Feature {i}: type={type(feat)}")
    #     else:
    #         print(f"  Single feature: type={type(features)}")

    # Token type checking helpers
    def is_note_on(self, token):
        """Check if token is a NOTE_ON token"""
        return self.NOTE_ON_OFFSET <= token < self.NOTE_OFF_OFFSET
        
    def is_note_off(self, token):
        """Check if token is a NOTE_OFF token"""
        return self.NOTE_OFF_OFFSET <= token < self.VELOCITY_OFFSET
        
    def is_time_shift(self, token):
        return token >= self.TIME_SHIFT_OFFSET
        
    def get_curriculum_weight(self, loss):
        """Implement curriculum learning by adjusting loss weights"""
        return torch.sigmoid(1 - loss.detach())
    
    def extract_harmony_features(self, predictions, targets):
        """Extract harmony features with corrected tensor handling"""
        try:
            device = predictions.device
            
            # Handle predictions
            pred_notes = predictions.argmax(dim=-1)  # [batch_size, seq_len]
            
            # Handle targets - ensure it's the right shape
            if targets.dim() == 3:
                target_notes = targets.argmax(dim=-1)  # [batch_size, seq_len]
            else:
                target_notes = targets  # Already in the right format
                
            # Process first sequence in batch
            pred_sequence = pred_notes[0]  # [seq_len]
            target_sequence = target_notes[0]  # [seq_len]
            
            # Build prediction chords
            pred_chords = []
            current_chord = []
            
            for token in pred_sequence:
                token = token.item()
                if token < 128:  # Note tokens
                    current_chord.append(token)
                    if len(current_chord) >= 1:  # Create chord for each note
                        pred_chords.append(current_chord.copy())
                else:  # Non-note tokens mark chord boundaries
                    if current_chord:
                        current_chord = []
                        
            # Build target chords similarly
            target_chords = []
            current_chord = []
            
            for token in target_sequence:
                token = token.item()
                if token < 128:
                    current_chord.append(token)
                    if len(current_chord) >= 1:
                        target_chords.append(current_chord.copy())
                else:
                    if current_chord:
                        current_chord = []
            
            # Ensure we have at least one chord
            if not pred_chords:
                pred_chords = [[0]]
            if not target_chords:
                target_chords = [[0]]
                
            # Convert to tensors with padding
            max_chord_size = max(
                max(len(chord) for chord in pred_chords),
                max(len(chord) for chord in target_chords)
            )
            
            pred_tensor = torch.zeros((len(pred_chords), max_chord_size), device=device)
            target_tensor = torch.zeros((len(target_chords), max_chord_size), device=device)
            
            for i, chord in enumerate(pred_chords):
                pred_tensor[i, :len(chord)] = torch.tensor(chord, device=device)
                
            for i, chord in enumerate(target_chords):
                target_tensor[i, :len(chord)] = torch.tensor(chord, device=device)
                
            return pred_tensor, target_tensor
            
        except Exception as e:
            print(f"Warning in extract_harmony_features: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            return None
    
    def extract_rhythm_features(self, predictions, targets):
        """Extract rhythm-related features from predictions and targets"""
        device = predictions.device
        pred_tokens = predictions.argmax(dim=-1)
        
        # Initialize tracking variables
        current_time = 0.0
        note_start_times = {}
        onset_times = []
        durations = []
        
        # Process each sequence in the batch
        for sequence in pred_tokens:
            current_time = 0.0
            active_notes = {}  # Track {note: start_time}
            
            for token in sequence:
                token = token.item()
                
                if self.is_time_shift(token):
                    delta = (token - self.TIME_SHIFT_OFFSET) * self.time_step
                    current_time += delta
                elif self.is_note_on(token):
                    note = token - self.NOTE_ON_OFFSET
                    active_notes[note] = current_time
                    onset_times.append(current_time)
                elif self.is_note_off(token):
                    note = token - self.NOTE_OFF_OFFSET
                    if note in active_notes:
                        duration = current_time - active_notes[note]
                        durations.append(duration)
                        del active_notes[note]

        # Convert to tensors with proper handling
        if onset_times:
            onset_tensor = torch.tensor(onset_times, device=device, dtype=torch.float32)
            # Normalize onset times to be between 0 and 1
            if onset_tensor.max() > 0:
                onset_tensor = onset_tensor / onset_tensor.max()
        else:
            return None
        
        if durations:
            duration_tensor = torch.tensor(durations, device=device, dtype=torch.float32)
            # Normalize durations
            if duration_tensor.max() > 0:
                duration_tensor = duration_tensor / duration_tensor.max()
        else:
            return None

        return onset_tensor, duration_tensor
    
    def get_rhythm_events(self, sequence):
        """Extract rhythm events from token sequence with improved safety"""
        device = sequence.device
        events = {
            'onset_times': [],
            'durations': []
        }
        
        try:
            current_time = 0.0
            for token in sequence.view(-1):
                if token >= self.pad_token_id:
                    continue
                    
                if self.is_time_shift(token):
                    delta = float(token - self.TIME_SHIFT_OFFSET) * self.time_step
                    current_time += delta
                elif self.is_note_on(token):
                    events['onset_times'].append(float(current_time))
                    events['durations'].append(self.time_step)  # Default duration
            
            # Convert to tensors with proper device
            return {
                k: torch.tensor(v, device=device, dtype=torch.float32) 
                if v else torch.tensor([], device=device, dtype=torch.float32) 
                for k, v in events.items()
            }
            
        except Exception as e:
            print(f"Warning in get_rhythm_events: {str(e)}")
            return {
                k: torch.tensor([], device=device, dtype=torch.float32) 
                for k in events.keys()
            }
    
    def extract_contour_features(self, predictions, targets):
        """Extract melodic contour features from predictions and targets"""
        device = predictions.device
        pred_tokens = predictions.argmax(dim=-1)
        
        # Extract pitch sequences and intervals
        pitches = []
        intervals = []
        current_sequence = []
        
        for sequence in pred_tokens:
            sequence_pitches = []
            for token in sequence:
                token = token.item()
                if self.is_note_on(token):
                    pitch = token - self.NOTE_ON_OFFSET
                    sequence_pitches.append(pitch)
                    if len(sequence_pitches) > 1:
                        interval = sequence_pitches[-1] - sequence_pitches[-2]
                        intervals.append(interval)
        
            if sequence_pitches:
                # Normalize pitches to be between 0 and 1
                seq_array = np.array(sequence_pitches)
                normalized_pitches = (seq_array - seq_array.min()) / (seq_array.max() - seq_array.min() + 1e-7)
                pitches.extend(normalized_pitches.tolist())
                current_sequence.extend(sequence_pitches)

        # Only return features if we have enough data
        if len(pitches) > 1 and len(intervals) > 0:
            pitch_tensor = torch.tensor(pitches, device=device, dtype=torch.float32)
            interval_tensor = torch.tensor(intervals, device=device, dtype=torch.float32)
            
            # Normalize intervals to be between -1 and 1
            if interval_tensor.abs().max() > 0:
                interval_tensor = interval_tensor / (interval_tensor.abs().max() + 1e-7)
            
            return pitch_tensor, interval_tensor
        
        return None
    
    def get_active_notes(self, sequence):
        """Convert token sequence to list of active notes"""
        active_notes = set()
        chords = []
        device = sequence.device
        
        for token in sequence.view(-1):
            if token >= self.pad_token_id:
                continue
                
            if self.is_note_on(token):
                note = token - self.NOTE_ON_OFFSET
                active_notes.add(note.item())
            elif self.is_note_off(token):
                note = token - self.NOTE_OFF_OFFSET
                active_notes.discard(note.item())
            
            if active_notes:
                chord = torch.tensor(list(active_notes), device=device, dtype=torch.long)
                chords.append(chord)
        
        if not chords:
            return torch.zeros((0,), device=device, dtype=torch.long)
            
        # Pad chords to the same length
        max_len = max(len(c) for c in chords)
        padded_chords = []
        for chord in chords:
            if len(chord) < max_len:
                padding = torch.zeros(max_len - len(chord), device=device, dtype=torch.long)
                chord = torch.cat([chord, padding])
            padded_chords.append(chord)
        
        return torch.stack(padded_chords)

    def get_pitch_sequence(self, sequence):
        """Extract pitch sequence from token sequence"""
        pitches = []
        for token in sequence.view(-1):
            if token >= self.pad_token_id:
                continue
                
            if self.is_note_on(token):
                pitch = token - self.NOTE_ON_OFFSET
                pitches.append(pitch)
                
        return torch.tensor(pitches)

    def get_note_ranges(self, sequences):
        """Calculate pitch range context for each position"""
        batch_size, seq_len = sequences.shape
        note_ranges = torch.zeros((batch_size, seq_len), dtype=torch.long, device=sequences.device)
        active_notes = torch.zeros((batch_size, 128), dtype=torch.bool, device=sequences.device)
        
        for i in range(seq_len):
            tokens = sequences[:, i]
            
            # Update active notes
            is_note_on = self.is_note_on_token(tokens)
            is_note_off = self.is_note_off_token(tokens)
            
            if is_note_on.any():
                notes = tokens[is_note_on] - self.tokenizer.NOTE_ON_OFFSET
                active_notes[is_note_on, notes] = True
            
            if is_note_off.any():
                notes = tokens[is_note_off] - self.tokenizer.NOTE_OFF_OFFSET
                active_notes[is_note_off, notes] = False
            
            # Calculate pitch range (mean of active notes)
            active_indices = torch.arange(128, device=sequences.device).unsqueeze(0)
            masked_indices = active_indices * active_notes.float()
            note_sum = torch.sum(masked_indices, dim=1)
            note_count = torch.sum(active_notes, dim=1)
            mean_pitch = (note_sum / note_count.clamp(min=1)).long()
            
            note_ranges[:, i] = mean_pitch
        
        return note_ranges

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