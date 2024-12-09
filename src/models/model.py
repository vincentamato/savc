import torch
import torch.nn as nn
from transformers import ViTModel
import math
import random
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, scale=1.0):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term) * scale
        pe[:, 0, 1::2] = torch.cos(position * div_term) * scale
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class MIDITokenizer:
    def __init__(self, max_notes=128, max_velocity=128, max_time_shift=100):
        self.max_notes = max_notes
        self.max_velocity = max_velocity
        self.max_time_shift = max_time_shift
        
        # Enhanced token types
        self.note_on_offset = 0
        self.note_off_offset = max_notes
        self.velocity_offset = 2 * max_notes
        self.time_shift_offset = 2 * max_notes + max_velocity
        self.tempo_offset = self.time_shift_offset + max_time_shift  # New tempo tokens
        self.control_offset = self.tempo_offset + 128  # New control change tokens
        
        # Special tokens
        self.bos_token = self.control_offset + 128
        self.eos_token = self.bos_token + 1
        self.pad_token = 0
        
        self.vocab_size = self.eos_token + 1

class TransformerWithStochasticDepth(nn.Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stochastic_depth_prob = 0.1
        
    def _stochastic_depth(self, x, training):
        if not training or random.random() > self.stochastic_depth_prob:
            return x
        return torch.zeros_like(x)
        
    def forward(self, src, tgt, *args, **kwargs):
        output = super().forward(src, tgt, *args, **kwargs)
        return self._stochastic_depth(output, self.training)

class MusicGenerationTransformer(nn.Module):
    def __init__(
        self,
        vit_name="google/vit-large-patch16-384",
        d_model=1024,
        nhead=16,
        num_encoder_layers=8,
        num_decoder_layers=8,
        dim_feedforward=4096,
        dropout=0.1,
        max_seq_length=1024,
        tokenizer=None,
        min_teacher_forcing=0.2,
        label_smoothing=0.1,
        temperature=1.0
    ):
        super().__init__()
        
        self.vit = ViTModel.from_pretrained(vit_name)
        self.tokenizer = tokenizer or MIDITokenizer()
        self.max_seq_length = max_seq_length
        self.min_teacher_forcing = min_teacher_forcing
        self.label_smoothing = label_smoothing
        self.temperature = temperature
        
        # Freeze ViT weights initially
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Enhanced projection layers
        self.vit_projection = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Embedding layers with weight tying
        self.token_embedding = nn.Embedding(
            self.tokenizer.vocab_size, 
            d_model, 
            padding_idx=self.tokenizer.pad_token
        )
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, scale=1.0)
        
        # Enhanced transformer with stochastic depth
        self.transformer = TransformerWithStochasticDepth(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection with weight tying
        self.output = nn.Linear(d_model, self.tokenizer.vocab_size)
        self.output.weight = self.token_embedding.weight
        
        self._init_parameters()
        
    def _init_parameters(self):
        """Enhanced parameter initialization"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'transformer' in name:
                    # Special initialization for transformer layers
                    nn.init.xavier_uniform_(p, gain=1/math.sqrt(2))
                else:
                    nn.init.xavier_normal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
            elif 'embedding' in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, tokens):
        return tokens == self.tokenizer.pad_token

    def forward(self, images, target_sequences=None, teacher_forcing_ratio=None):
        batch_size = images.size(0)
        device = images.device
        
        # Enhanced image processing
        with torch.set_grad_enabled(not self.vit.training):
            vit_output = self.vit(images)
        image_features = vit_output.last_hidden_state
        encoder_output = self.vit_projection(image_features)
        
        if not self.training or target_sequences is None:
            return self.generate(encoder_output)
            
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = max(
                self.min_teacher_forcing,
                random.random()
            )
        
        target_length = target_sequences.size(1)
        decoding_tokens = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_token,
            device=device
        )
        outputs = []
        
        for i in range(1, target_length):
            token_embeddings = self.token_embedding(decoding_tokens)
            token_embeddings = self.pos_encoder(token_embeddings)
            
            tgt_mask = self.generate_square_subsequent_mask(decoding_tokens.size(1)).to(device)
            tgt_padding_mask = self.create_padding_mask(decoding_tokens)
            
            transformer_output = self.transformer(
                encoder_output,
                token_embeddings,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask
            )
            
            output = self.output(transformer_output) / self.temperature
            outputs.append(output)
            
            if random.random() < teacher_forcing_ratio:
                next_token = target_sequences[:, i].unsqueeze(1)
            else:
                next_token = output[:, -1:].argmax(dim=-1)
            
            decoding_tokens = torch.cat([decoding_tokens, next_token], dim=1)
        
        return torch.cat(outputs, dim=1)
    
    def generate(self, encoder_output, max_length=None, beam_size=5):
        """Enhanced generation with beam search"""
        if max_length is None:
            max_length = self.max_seq_length
            
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Initialize beam search
        beams = [(torch.full((batch_size, 1), self.tokenizer.bos_token, device=device), 0.0)]
        finished_beams = []
        
        for _ in range(max_length - 1):
            candidates = []
            
            for sequence, score in beams:
                token_embeddings = self.token_embedding(sequence)
                token_embeddings = self.pos_encoder(token_embeddings)
                
                tgt_mask = self.generate_square_subsequent_mask(sequence.size(1)).to(device)
                
                transformer_output = self.transformer(
                    encoder_output,
                    token_embeddings,
                    tgt_mask=tgt_mask
                )
                
                logits = self.output(transformer_output[:, -1:]) / self.temperature
                probs = torch.softmax(logits, dim=-1)
                
                # Get top-k candidates
                values, indices = probs.topk(beam_size)
                
                for i in range(beam_size):
                    new_sequence = torch.cat([sequence, indices[:, :, i]], dim=1)
                    new_score = score - torch.log(values[:, :, i]).mean()
                    candidates.append((new_sequence, new_score))
            
            # Select top beam_size candidates
            candidates.sort(key=lambda x: x[1])
            beams = candidates[:beam_size]
            
            # Check for completed sequences
            for sequence, score in beams:
                if sequence[:, -1].eq(self.tokenizer.eos_token).all():
                    finished_beams.append((sequence, score))
            
            if len(finished_beams) >= beam_size:
                break
        
        # Return best completed sequence or best incomplete sequence
        if finished_beams:
            return min(finished_beams, key=lambda x: x[1])[0][:, 1:-1]
        return beams[0][0][:, 1:]

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)

class MusicTransformerTrainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        device, 
        max_grad_norm=1.0,
        num_warmup_steps=1000,
        num_training_steps=100000
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps, 
            num_training_steps
        )
        
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=model.tokenizer.pad_token,
            label_smoothing=model.label_smoothing
        )
    
    def train_step(self, images, target_sequences):
        self.model.train()
        self.optimizer.zero_grad()
        
        output = self.model(images, target_sequences)
        loss = self.criterion(
            output.view(-1, output.size(-1)),
            target_sequences[:, 1:].contiguous().view(-1)
        )
        
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()