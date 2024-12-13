import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
from torch.nn import LayerNorm
import math

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Create relative position embeddings matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer to save in state dict but not as parameter
        self.register_buffer('pe', pe)
        
        # Learnable projection for relative positions
        self.relative_projection = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Ensure we only use the needed sequence length
        rel_pos = self.pe[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Project relative positions
        rel_pos = self.relative_projection(rel_pos)
        
        return rel_pos

class RelativeTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 batch_first=True, norm_first=True):
        super().__init__()
        
        # Use PyTorch's built-in MultiheadAttention
        self.self_attn = nn.MultiheadAttention(
            d_model, 
            nhead, 
            dropout=dropout, 
            batch_first=batch_first
        )
        
        self.multihead_attn = nn.MultiheadAttention(
            d_model, 
            nhead, 
            dropout=dropout, 
            batch_first=batch_first
        )
        
        # Standard feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.norm_first = norm_first
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=None, memory_is_causal=None):
        x = tgt
        
        # Handle causal masking
        if tgt_is_causal:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        
        if self.norm_first:
            # Self attention block
            x2 = self.norm1(x)
            x = x + self.dropout1(self.self_attn(x2, x2, x2, attn_mask=tgt_mask,
                                                key_padding_mask=tgt_key_padding_mask)[0])
            
            # Cross attention block
            x2 = self.norm2(x)
            x = x + self.dropout2(self.multihead_attn(x2, memory, memory, attn_mask=memory_mask,
                                                     key_padding_mask=memory_key_padding_mask)[0])
            
            # Feedforward block
            x2 = self.norm3(x)
            x = x + self.dropout3(self.ffn(x2))
        else:
            # Self attention block
            x2 = x + self.dropout1(self.self_attn(x, x, x, attn_mask=tgt_mask,
                                                 key_padding_mask=tgt_key_padding_mask)[0])
            x = self.norm1(x2)
            
            # Cross attention block
            x2 = x + self.dropout2(self.multihead_attn(x, memory, memory, attn_mask=memory_mask,
                                                      key_padding_mask=memory_key_padding_mask)[0])
            x = self.norm2(x2)
            
            # Feedforward block
            x2 = x + self.dropout3(self.ffn(x))
            x = self.norm3(x2)
        
        return x

class RelativeMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, batch_first=True):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = self.d_head ** -0.5
        self.batch_first = batch_first
        
        # Add layer normalization for Q, K, V
        self.q_norm = nn.LayerNorm(d_model)
        self.k_norm = nn.LayerNorm(d_model)
        self.v_norm = nn.LayerNorm(d_model)
        
        # Initialize projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.r_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Add dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with smaller weights but not too small
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.r_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=0.1)  # Increased from 0.01
            if hasattr(proj, 'bias') and proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _reshape_attention_mask(self, attn_mask, batch_size, tgt_len, src_len):
        if attn_mask is None:
            return None

        if len(attn_mask.shape) == 2:
            expanded_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            expanded_mask = expanded_mask.expand(batch_size, self.num_heads, tgt_len, src_len)
            return expanded_mask
            
        elif len(attn_mask.shape) == 3:
            expanded_mask = attn_mask.unsqueeze(1)
            expanded_mask = expanded_mask.expand(batch_size, self.num_heads, tgt_len, src_len)
            return expanded_mask
            
        return attn_mask
        
    def forward(self, query, key, value, rel_pos=None, attn_mask=None, key_padding_mask=None):
        batch_size = query.size(0)
        tgt_len = query.size(1)
        src_len = key.size(1)
        
        # Debug inputs
        print(f"Input stats:")
        print(f"Query: min={query.min():.4f}, max={query.max():.4f}, mean={query.mean():.4f}")
        print(f"Key: min={key.min():.4f}, max={key.max():.4f}, mean={key.mean():.4f}")
        print(f"Value: min={value.min():.4f}, max={value.max():.4f}, mean={value.mean():.4f}")
        
        # Apply layer norm first
        query = self.q_norm(query)
        key = self.k_norm(key)
        value = self.v_norm(value)
        
        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape to (batch, head, seq_len, d_head)
        q = q.contiguous().view(batch_size, tgt_len, self.num_heads, self.d_head).transpose(1, 2)
        k = k.contiguous().view(batch_size, src_len, self.num_heads, self.d_head).transpose(1, 2)
        v = v.contiguous().view(batch_size, src_len, self.num_heads, self.d_head).transpose(1, 2)
        
        # Scale queries for stable attention
        q = q * self.scale
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Add relative position bias if provided
        if rel_pos is not None:
            r = self.r_proj(rel_pos)
            r = r.view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
            rel_pos_scores = torch.matmul(q, r.transpose(-2, -1))
            attn_scores = attn_scores + rel_pos_scores * 0.1  # Scale relative scores
        
        # Handle attention mask
        if attn_mask is not None:
            attn_mask = self._reshape_attention_mask(attn_mask, batch_size, tgt_len, src_len)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)  # Use finite value
        
        # Handle padding mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, src_len)
            key_padding_mask = key_padding_mask.expand(-1, self.num_heads, tgt_len, -1)
            attn_scores = attn_scores.masked_fill(key_padding_mask == 0, -1e9)  # Use finite value
        
        # Apply softmax with better numerical stability
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True)[0]  # Subtract max for stability
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Compute output
        output = torch.matmul(attn_probs, v)
        
        # Reshape output
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        
        # Final projection
        output = self.out_proj(output)
        
        return output, attn_probs
    
class MusicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        nhead=12,
        num_decoder_layers=8,
        dim_feedforward=3072,
        dropout=0.1,
        max_seq_length=1024,
        tokenizer=None,
        vit_model="google/vit-base-patch16-384",
        freeze_vit=True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # ViT for image features
        self.vit = ViTModel.from_pretrained(vit_model)
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
                
        # Multiple visual projection layers for different aspects of music
        self.visual_projections = nn.ModuleDict({
            'melody': nn.Sequential(
                nn.Linear(self.vit.config.hidden_size, d_model),
                LayerNorm(d_model),
                nn.Dropout(dropout)
            ),
            'rhythm': nn.Sequential(
                nn.Linear(self.vit.config.hidden_size, d_model),
                LayerNorm(d_model),
                nn.Dropout(dropout)
            ),
            'harmony': nn.Sequential(
                nn.Linear(self.vit.config.hidden_size, d_model),
                LayerNorm(d_model),
                nn.Dropout(dropout)
            )
        })
        
        # Enhanced token embeddings with musical context
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.note_range_embedding = nn.Embedding(128, d_model)  # Pitch context
        self.beat_position_embedding = nn.Embedding(32, d_model)  # Rhythmic context
        
        # Relative positional encoding
        self.rel_pos_encoder = RelativePositionalEncoding(d_model, max_seq_length)
        
        # Cross-modal attention layers
        self.cross_modal_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(3)  # One for each musical aspect
        ])
        self.cross_modal_norms = nn.ModuleList([
            LayerNorm(d_model) for _ in range(3)
        ])
        
        # Transformer decoder with PyTorch's implementation
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Calculate output sizes for each prediction head
        note_output_size = 2 * tokenizer.max_notes
        velocity_output_size = tokenizer.max_velocity
        time_output_size = tokenizer.max_time_shift

        # Prediction heads
        self.prediction_heads = nn.ModuleDict({
            'note': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.LayerNorm(d_model // 2),
                nn.Linear(d_model // 2, note_output_size)
            ),
            'velocity': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.LayerNorm(d_model // 2),
                nn.Linear(d_model // 2, velocity_output_size)
            ),
            'time': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.LayerNorm(d_model // 2),
                nn.Linear(d_model // 2, time_output_size)
            )
        })
        
        # Prediction norms
        self.prediction_norms = nn.ModuleDict({
            'note': nn.LayerNorm([note_output_size]),
            'velocity': nn.LayerNorm([velocity_output_size]),
            'time': nn.LayerNorm([time_output_size])
        })
        
        # Musical context modeling
        self.harmony_model = HarmonyModel(d_model)
        self.rhythm_model = RhythmModel(d_model)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=1/math.sqrt(2))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def gradient_checkpointing_enable(self):
        self.decoder.gradient_checkpointing = True
        if hasattr(self.vit, 'gradient_checkpointing_enable'):
            self.vit.gradient_checkpointing_enable()
            
    def gradient_checkpointing_disable(self):
        self.decoder.gradient_checkpointing = False
        if hasattr(self.vit, 'gradient_checkpointing_disable'):
            self.vit.gradient_checkpointing_disable()
        
    def forward(self, images, target_sequences=None, token_types=None, attention_mask=None):
        batch_size = images.size(0)
        device = images.device
        
        # Process images through ViT
        vit_outputs = self.vit(images)
            
        # Project visual features with scaling
        visual_features = {}
        for aspect, proj in self.visual_projections.items():
            feat = proj(vit_outputs.last_hidden_state)
            feat = feat + 1e-6
            visual_features[aspect] = feat * 0.1

        # Get embeddings and musical context
        token_emb = self.token_embedding(target_sequences)
        beat_pos = self.get_beat_positions(target_sequences)
        note_ranges = self.get_note_ranges(target_sequences)
        
        # Scale embeddings
        beat_emb = self.beat_position_embedding(beat_pos) * 0.1
        range_emb = self.note_range_embedding(note_ranges) * 0.1
        rel_pos = self.rel_pos_encoder(target_sequences) * 0.1
        
        # Combine embeddings with scaling
        embeddings = token_emb + beat_emb + range_emb + rel_pos
        embeddings = F.layer_norm(embeddings, [embeddings.size(-1)])

        # Create separate attention masks for cross-modal attention and self-attention
        if attention_mask is not None:
            # For cross-modal attention, create a mask that matches the visual feature length
            visual_attention_mask = torch.ones(
                batch_size, 
                visual_features['melody'].size(1), 
                device=device
            )  # All visual tokens are valid

            # For sequence attention, use the provided mask
            sequence_attention_mask = attention_mask.float()
            sequence_attention_mask = (1.0 - sequence_attention_mask) * -1e4
        else:
            visual_attention_mask = None
            sequence_attention_mask = None
        
        # Cross-modal attention for each musical aspect
        cross_modal_outputs = []
        for layer, norm, aspect in zip(self.cross_modal_layers, self.cross_modal_norms, visual_features.keys()):
            # Apply cross attention with correct mask shapes
            cross_out = layer(
                query=embeddings,
                key=visual_features[aspect],
                value=visual_features[aspect],
                key_padding_mask=visual_attention_mask  # Use visual attention mask here
            )[0]
            cross_out = norm(cross_out)
            cross_modal_outputs.append(cross_out)

        # Combine cross-modal outputs with mean and normalization
        combined_features = torch.stack(cross_modal_outputs, dim=0).mean(dim=0)
        combined_features = F.layer_norm(combined_features, [combined_features.size(-1)])

        # Generate causal mask for decoder
        seq_length = target_sequences.size(1)
        causal_mask = torch.triu(
            torch.full((seq_length, seq_length), float('-inf'), device=device), 
            diagonal=1
        )
        
        # Pass through transformer decoder with correct mask shapes
        decoder_output = self.decoder(
            combined_features,
            visual_features['melody'],
            tgt_mask=causal_mask,
            tgt_key_padding_mask=sequence_attention_mask,  # Use sequence mask for target
            memory_key_padding_mask=visual_attention_mask  # Use visual mask for memory
        )

        # Scale decoder output
        decoder_output = decoder_output * 0.1

        # Get predictions with gradient scaling
        predictions = {}
        for aspect, head in self.prediction_heads.items():
            with torch.cuda.amp.autocast(enabled=False):
                pred = head(decoder_output.float())
            predictions[aspect] = pred

        # Get musical context with scaling
        with torch.cuda.amp.autocast(enabled=False):
            harmony_context = self.harmony_model(decoder_output.float()) * 0.1
            rhythm_context = self.rhythm_model(decoder_output.float()) * 0.1

        # Combine predictions with normalization
        final_predictions = self.combine_predictions(
            predictions,
            harmony_context,
            rhythm_context
        )
        
        return final_predictions, token_types

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.float()

    def combine_predictions(self, predictions, harmony_context, rhythm_context):
        note_logits = predictions['note']
        velocity_logits = predictions['velocity']
        time_logits = predictions['time']
        
        # Apply layer normalization
        note_logits = self.prediction_norms['note'](note_logits)
        velocity_logits = self.prediction_norms['velocity'](velocity_logits)
        time_logits = self.prediction_norms['time'](time_logits)
        
        # Add musical context
        note_context = self.prediction_heads['note'](harmony_context)
        time_context = self.prediction_heads['time'](rhythm_context)
        
        note_logits = note_logits + note_context
        time_logits = time_logits + time_context
        
        # Combine into vocabulary-sized predictions
        batch_size, seq_len = note_logits.shape[:2]
        combined = torch.zeros(
            (batch_size, seq_len, self.vocab_size),
            device=note_logits.device
        )
        
        # Fill in predictions for each token type
        start_idx = len(self.tokenizer.special_tokens)
        
        note_end = start_idx + 2 * self.tokenizer.max_notes
        combined[:, :, start_idx:note_end] = note_logits
        
        vel_start = note_end
        vel_end = vel_start + self.tokenizer.max_velocity
        combined[:, :, vel_start:vel_end] = velocity_logits
        
        time_start = vel_end
        time_end = time_start + self.tokenizer.max_time_shift
        combined[:, :, time_start:time_end] = time_logits
        
        return combined
    
    def get_beat_positions(self, sequences):
        """Calculate beat positions for each token in the sequence"""
        batch_size, seq_len = sequences.shape
        beat_positions = torch.zeros((batch_size, seq_len), dtype=torch.long, device=sequences.device)
        current_time = torch.zeros(batch_size, device=sequences.device)
        
        for i in range(seq_len):
            tokens = sequences[:, i]
            
            # Convert current time to beat position (assuming 4/4 time)
            beat_pos = (current_time / self.tokenizer.time_step % 32).long()
            beat_positions[:, i] = beat_pos
            
            # Update time for time shift tokens
            is_time = self.is_time_token(tokens)
            time_value = torch.where(
                is_time,
                (tokens - self.tokenizer.TIME_SHIFT_OFFSET) * self.tokenizer.time_step,
                torch.zeros_like(tokens, dtype=torch.float)
            )
            current_time = current_time + time_value
        
        return beat_positions

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
    
    def sample_token(self, logits, active_notes, current_time, temperature=0.8):
        """Sample next token considering musical constraints"""
        device = logits.device
        
        # Apply constraints based on current musical state
        adjusted_logits = self.apply_musical_constraints(
            logits,
            active_notes,
            current_time
        )
        
        # Apply temperature
        logits_temp = adjusted_logits / temperature
        
        # Get probabilities
        probs = F.softmax(logits_temp, dim=-1)
        
        # Sample token
        next_token = torch.multinomial(probs, 1).item()
        
        return next_token

    def apply_musical_constraints(self, logits, active_notes, current_time):
        """Apply musical constraints to logits before sampling"""
        device = logits.device
        adjusted_logits = logits.clone()
        
        # Get token ranges for each type
        note_start = len(self.tokenizer.special_tokens)
        note_end = note_start + 2 * self.tokenizer.max_notes
        vel_start = note_end
        vel_end = vel_start + self.tokenizer.max_velocity
        time_start = vel_end
        
        # Constraint 1: Prevent extremely long notes
        if active_notes:
            oldest_note_time = min(start_time for start_time, _ in active_notes.values())
            if current_time - oldest_note_time > 2.0:  # 2 seconds max duration
                # Increase probability of note-offs for active notes
                for note in active_notes:
                    note_off_token = self.tokenizer.NOTE_OFF_OFFSET + note
                    adjusted_logits[note_off_token] += 5.0
        
        # Constraint 2: Limit simultaneous notes
        if len(active_notes) >= 8:  # Maximum 8 simultaneous notes
            # Prevent new notes, encourage note-offs
            adjusted_logits[note_start:note_end] = float('-inf')
            for note in active_notes:
                note_off_token = self.tokenizer.NOTE_OFF_OFFSET + note
                adjusted_logits[note_off_token] += 3.0
        
        # Constraint 3: Encourage reasonable velocities
        vel_range = adjusted_logits[vel_start:vel_end]
        if vel_range.max() > float('-inf'):
            # Discourage extreme velocities
            penalty = torch.abs(torch.arange(self.tokenizer.max_velocity, device=device) - 64) * 0.1
            adjusted_logits[vel_start:vel_end] -= penalty
        
        # Constraint 4: Limit time shifts
        time_range = adjusted_logits[time_start:]
        if time_range.max() > float('-inf'):
            # Prefer shorter time shifts
            penalty = torch.arange(self.tokenizer.max_time_shift, device=device) * 0.2
            adjusted_logits[time_start:] -= penalty
        
        # Constraint 5: Maintain rhythmic coherence
        if current_time % 0.5 < 0.125:  # Near beat boundary
            # Encourage events on strong beats
            adjusted_logits[note_start:note_end] += 2.0
            adjusted_logits[time_start:] += 1.0
        
        # Prevent any invalid token selections
        adjusted_logits[adjusted_logits == float('-inf')] = -1e9
        
        return adjusted_logits

    def generate(self, visual_features):
        """Generate music tokens from visual features"""
        device = next(self.parameters()).device
        
        # Initialize sequence with BOS token
        sequence = [self.tokenizer.special_tokens['BOS']]
        current_time = 0.0
        active_notes = {}
        
        while len(sequence) < self.max_seq_length:
            # Convert sequence to tensor
            curr_seq = torch.tensor([sequence], device=device)
            
            # Get musical context
            beat_pos = self.get_beat_positions(curr_seq)
            note_ranges = self.get_note_ranges(curr_seq)
            
            # Get embeddings and positions
            token_emb = self.token_embedding(curr_seq)
            beat_emb = self.beat_position_embedding(beat_pos)
            range_emb = self.note_range_embedding(note_ranges)
            rel_pos = self.rel_pos_encoder(curr_seq)
            
            # Combine embeddings
            embeddings = token_emb + beat_emb + range_emb + rel_pos
            
            # Process through model
            cross_modal_outputs = []
            for layer, norm, aspect in zip(
                self.cross_modal_layers,
                self.cross_modal_norms,
                visual_features.keys()
            ):
                cross_out = layer(
                    embeddings,
                    visual_features[aspect],
                    visual_features[aspect]
                )[0]
                cross_out = norm(cross_out + embeddings)
                cross_modal_outputs.append(cross_out)
            
            combined_features = sum(cross_modal_outputs) / len(cross_modal_outputs)
            
            # Decoder processing
            causal_mask = self.generate_square_subsequent_mask(curr_seq.size(1)).to(device)
            decoder_output = self.decoder(
                combined_features,
                visual_features['melody'],
                tgt_mask=causal_mask
            )
            
            # Get predictions
            predictions = {
                aspect: head(decoder_output)
                for aspect, head in self.prediction_heads.items()
            }
            
            # Get musical context
            harmony_context = self.harmony_model(decoder_output)
            rhythm_context = self.rhythm_model(decoder_output)
            
            # Combine predictions
            logits = self.combine_predictions(
                predictions,
                harmony_context,
                rhythm_context
            )
            
            # Sample next token
            next_token = self.sample_token(
                logits[:, -1],
                active_notes,
                current_time
            )
            
            # Update state
            sequence.append(next_token)
            
            if self.is_time_token(next_token):
                time_steps = next_token - self.tokenizer.TIME_SHIFT_OFFSET
                current_time += time_steps * self.tokenizer.time_step
            
            if current_time >= 30.0 or next_token == self.tokenizer.special_tokens['EOS']:
                break
        
        return torch.tensor([sequence], device=device)

    def is_note_on_token(self, token):
        """Check if token is a note-on event"""
        if isinstance(token, torch.Tensor):
            return (self.tokenizer.NOTE_ON_OFFSET <= token) & (token < self.tokenizer.NOTE_OFF_OFFSET)
        return self.tokenizer.NOTE_ON_OFFSET <= token < self.tokenizer.NOTE_OFF_OFFSET

    def is_note_off_token(self, token):
        """Check if token is a note-off event"""
        if isinstance(token, torch.Tensor):
            return (self.tokenizer.NOTE_OFF_OFFSET <= token) & (token < self.tokenizer.VELOCITY_OFFSET)
        return self.tokenizer.NOTE_OFF_OFFSET <= token < self.tokenizer.VELOCITY_OFFSET

    def is_velocity_token(self, token):
        """Check if token is a velocity event"""
        if isinstance(token, torch.Tensor):
            return (self.tokenizer.VELOCITY_OFFSET <= token) & (token < self.tokenizer.TIME_SHIFT_OFFSET)
        return self.tokenizer.VELOCITY_OFFSET <= token < self.tokenizer.TIME_SHIFT_OFFSET

    def is_time_token(self, token):
        """Check if token is a time shift event"""
        if isinstance(token, torch.Tensor):
            return token >= self.tokenizer.TIME_SHIFT_OFFSET
        return token >= self.tokenizer.TIME_SHIFT_OFFSET
   
    
class HarmonyModel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.chord_detector = nn.Linear(d_model, 24)  # Major/minor for all roots
        self.progression_model = nn.GRU(24, d_model, batch_first=True)
        
    def forward(self, x):
        chord_logits = self.chord_detector(x)
        harmony_context, _ = self.progression_model(chord_logits)
        return harmony_context

class RhythmModel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.beat_detector = nn.Linear(d_model, 4)  # Quarter note positions
        self.meter_model = nn.GRU(4, d_model, batch_first=True)
        
    def forward(self, x):
        beat_logits = self.beat_detector(x)
        rhythm_context, _ = self.meter_model(beat_logits)
        return rhythm_context