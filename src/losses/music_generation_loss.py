import torch
import math
from torch import nn
import torch.nn.functional as F

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
        
        # Token type offsets (these should match your tokenizer)
        self.NOTE_ON_OFFSET = 3  # Assuming typical values, adjust as needed
        self.NOTE_OFF_OFFSET = self.NOTE_ON_OFFSET + 128
        self.VELOCITY_OFFSET = self.NOTE_OFF_OFFSET + 128
        self.TIME_SHIFT_OFFSET = self.VELOCITY_OFFSET + 128
        self.time_step = 0.02  # 20ms time step
        
    def forward(self, predictions, targets, attention_mask=None):
        """Forward pass with debugging"""
        # Handle tuple output from the model
        if isinstance(predictions, tuple):
            predictions, token_types = predictions

        # Add debugging for predictions and targets
        # print("\nDebug Loss Computation:")
        # print(f"Predictions shape: {predictions.shape}")
        # print(f"Targets shape: {targets.shape}")
        # print(f"Predictions range: min={predictions.min().item():.4f}, max={predictions.max().item():.4f}")
        
        # Basic token loss
        token_loss = self.token_criterion(
            predictions.view(-1, self.vocab_size),
            targets.view(-1)
        )
        # print(f"Token loss: {token_loss.item():.4f}")
        
        # Extract and debug musical features
        harmony_features = self.extract_harmony_features(predictions, targets)
        rhythm_features = self.extract_rhythm_features(predictions, targets)
        contour_features = self.extract_contour_features(predictions, targets)
        
        # Debug extracted features
        # print("\nFeature Extraction Debug:")
        # self._debug_features("Harmony", harmony_features)
        # self._debug_features("Rhythm", rhythm_features)
        # self._debug_features("Contour", contour_features)
        
        # Compute musical losses with debugging
        harmony_loss = self.harmony_loss(harmony_features)
        rhythm_loss = self.rhythm_loss(rhythm_features)
        contour_loss = self.contour_loss(contour_features)
        
        # print("\nComponent Losses:")
        # print(f"Harmony loss: {harmony_loss.item():.4f}")
        # print(f"Rhythm loss: {rhythm_loss.item():.4f}")
        # print(f"Contour loss: {contour_loss.item():.4f}")
        
        # Compute weights with debugging
        harmony_weight = 0.3 * self.get_curriculum_weight(harmony_loss)
        rhythm_weight = 0.3 * self.get_curriculum_weight(rhythm_loss)
        contour_weight = 0.2 * self.get_curriculum_weight(contour_loss)
        
        # print("\nLoss Weights:")
        # print(f"Harmony weight: {harmony_weight.item():.4f}")
        # print(f"Rhythm weight: {rhythm_weight.item():.4f}")
        # print(f"Contour weight: {contour_weight.item():.4f}")
        
        # Compute total loss with safety checks
        total_loss = token_loss
        if not torch.isnan(harmony_loss):
            total_loss = total_loss + harmony_weight * harmony_loss
        if not torch.isnan(rhythm_loss):
            total_loss = total_loss + rhythm_weight * rhythm_loss
        if not torch.isnan(contour_loss):
            total_loss = total_loss + contour_weight * contour_loss
        
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
        return self.NOTE_ON_OFFSET <= token < self.NOTE_OFF_OFFSET
        
    def is_note_off(self, token):
        return self.NOTE_OFF_OFFSET <= token < self.VELOCITY_OFFSET
        
    def is_time_shift(self, token):
        return token >= self.TIME_SHIFT_OFFSET
        
    def get_curriculum_weight(self, loss):
        """Implement curriculum learning by adjusting loss weights"""
        return torch.sigmoid(1 - loss.detach())
    
    def extract_harmony_features(self, predictions, targets):
        """Extract harmony-related features from predictions and targets with alignment"""
        device = predictions.device
        pred_notes = predictions.argmax(dim=-1)
        batch_size, seq_len = pred_notes.shape
        
        # print(f"\nDebug Harmony Extraction:")
        # print(f"Pred notes shape: {pred_notes.shape}")
        # print(f"Range: [{pred_notes.min()}, {pred_notes.max()}]")
        
        # Extract chords with sequence alignment
        pred_chords = []
        target_chords = []
        
        for b in range(batch_size):
            # Process predicted sequence
            active_notes = set()
            for token in pred_notes[b]:
                token = token.item()
                
                if self.is_note_on(token):
                    note = token - self.NOTE_ON_OFFSET
                    active_notes.add(note)
                elif self.is_note_off(token):
                    note = token - self.NOTE_OFF_OFFSET
                    active_notes.discard(note)
                
                if active_notes:
                    chord = sorted(list(active_notes))
                    pred_chords.append(chord)
            
            # Process target sequence
            active_notes = set()
            for token in targets[b]:
                token = token.item()
                
                if self.is_note_on(token):
                    note = token - self.NOTE_ON_OFFSET
                    active_notes.add(note)
                elif self.is_note_off(token):
                    note = token - self.NOTE_OFF_OFFSET
                    active_notes.discard(note)
                
                if active_notes:
                    chord = sorted(list(active_notes))
                    target_chords.append(chord)
        
        # Convert to tensors with padding
        if pred_chords and target_chords:
            max_voices = max(max(len(c) for c in pred_chords), max(len(c) for c in target_chords))
            
            # Pad chords to same length
            padded_pred_chords = []
            for chord in pred_chords:
                if len(chord) < max_voices:
                    chord = chord + [0] * (max_voices - len(chord))
                padded_pred_chords.append(torch.tensor(chord, device=device))
            
            padded_target_chords = []
            for chord in target_chords:
                if len(chord) < max_voices:
                    chord = chord + [0] * (max_voices - len(chord))
                padded_target_chords.append(torch.tensor(chord, device=device))
            
            pred_chords = torch.stack(padded_pred_chords)
            target_chords = torch.stack(padded_target_chords)
        else:
            # Return dummy tensors if no chords found
            pred_chords = torch.zeros((1, 1), device=device)
            target_chords = torch.zeros((1, 1), device=device)
        
        # print(f"Extracted chords shapes - Pred: {pred_chords.shape}, Target: {target_chords.shape}")
        return pred_chords, target_chords

    
    def extract_rhythm_features(self, predictions, targets):
        """Extract rhythm-related features from predictions and targets"""
        device = predictions.device
        pred_tokens = predictions.argmax(dim=-1)
        
        # print("\nDebug Rhythm Extraction:")
        # print(f"Pred tokens shape: {pred_tokens.shape}")
        
        # Extract onset times and durations
        onset_times = []
        durations = []
        current_time = 0.0
        note_start_times = {}
        
        for sequence in pred_tokens:
            for token in sequence:
                token = token.item()
                
                if self.is_time_shift(token):
                    delta = (token - self.TIME_SHIFT_OFFSET) * self.time_step
                    current_time += delta
                elif self.is_note_on(token):
                    note = token - self.NOTE_ON_OFFSET
                    note_start_times[note] = current_time
                    onset_times.append(current_time)
                elif self.is_note_off(token):
                    note = token - self.NOTE_OFF_OFFSET
                    if note in note_start_times:
                        duration = current_time - note_start_times[note]
                        durations.append(duration)
                        del note_start_times[note]
        
        # Convert to tensors with safety checks
        if onset_times:
            onset_times = torch.tensor(onset_times, device=device, dtype=torch.float32)
        else:
            onset_times = torch.zeros(1, device=device, dtype=torch.float32)
            
        if durations:
            durations = torch.tensor(durations, device=device, dtype=torch.float32)
        else:
            durations = torch.zeros(1, device=device, dtype=torch.float32)
        
        return onset_times, durations
    
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
        
        # print("\nDebug Contour Extraction:")
        # print(f"Pred tokens shape: {pred_tokens.shape}")
        
        # Extract pitch sequence
        pitches = []
        intervals = []
        
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
            pitches.extend(sequence_pitches)
        
        # Convert to tensors with safety checks
        if pitches:
            pitches = torch.tensor(pitches, device=device, dtype=torch.float32)
        else:
            pitches = torch.zeros(1, device=device, dtype=torch.float32)
            
        if intervals:
            intervals = torch.tensor(intervals, device=device, dtype=torch.float32)
        else:
            intervals = torch.zeros(1, device=device, dtype=torch.float32)
        
        return pitches, intervals
    
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