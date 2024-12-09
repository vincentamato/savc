import torch
import torch.nn as nn
from transformers import ViTModel
import math
import random

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return x

class MIDITokenizer:
    def __init__(self, max_notes=128, max_velocity=128, max_time_shift=100):
        self.max_notes = max_notes
        self.max_velocity = max_velocity
        self.max_time_shift = max_time_shift
        
        self.note_on_offset = 0
        self.note_off_offset = max_notes
        self.velocity_offset = 2 * max_notes
        self.time_shift_offset = 2 * max_notes + max_velocity
        
        self.vocab_size = 2 * max_notes + max_velocity + max_time_shift + 1  # +1 for EOS
        self.eos_token = self.vocab_size - 1
        self.pad_token = 0  # Using 0 as padding token

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
        min_teacher_forcing=0.2  # Minimum teacher forcing ratio
    ):
        super().__init__()
        
        self.vit = ViTModel.from_pretrained(vit_name)
        self.tokenizer = tokenizer or MIDITokenizer()
        self.max_seq_length = max_seq_length
        self.min_teacher_forcing = min_teacher_forcing
        
        # Freeze ViT weights initially
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Project ViT features to transformer dimension
        self.vit_projection = nn.Linear(self.vit.config.hidden_size, d_model)
        
        # Embedding layers
        self.token_embedding = nn.Embedding(
            self.tokenizer.vocab_size, 
            d_model, 
            padding_idx=self.tokenizer.pad_token
        )
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output = nn.Linear(d_model, self.tokenizer.vocab_size)
        
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize the parameters of the model"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for transformer"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, tokens):
        """Create padding mask for transformer"""
        return tokens == self.tokenizer.pad_token

    def forward(self, images, target_sequences=None, teacher_forcing_ratio=None):
        """
        Forward pass with teacher forcing during training
        Args:
            images: [batch_size, channels, height, width]
            target_sequences: [batch_size, seq_len] (optional)
            teacher_forcing_ratio: Override default ratio if provided
        """
        batch_size = images.size(0)
        device = images.device
        
        # Process images with ViT
        vit_output = self.vit(images)
        image_features = vit_output.last_hidden_state
        encoder_output = self.vit_projection(image_features)
        
        if not self.training or target_sequences is None:
            return self.generate(encoder_output)
            
        # Training mode with teacher forcing
        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = max(
                self.min_teacher_forcing,
                random.random()  # Random ratio between min and 1.0
            )
        
        target_length = target_sequences.size(1)
        decoding_tokens = target_sequences[:, 0].unsqueeze(1)  # Start tokens
        outputs = []
        
        # Generate sequence token by token
        for i in range(1, target_length):
            # Embed current tokens
            token_embeddings = self.token_embedding(decoding_tokens)
            token_embeddings = self.pos_encoder(token_embeddings)
            
            # Create masks
            tgt_mask = self.generate_square_subsequent_mask(decoding_tokens.size(1)).to(device)
            tgt_padding_mask = self.create_padding_mask(decoding_tokens)
            
            # Get prediction for current position
            transformer_output = self.transformer(
                encoder_output,
                token_embeddings,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask
            )
            output = self.output(transformer_output)
            outputs.append(output)
            
            # Teacher forcing decision for next token
            if random.random() < teacher_forcing_ratio:
                next_token = target_sequences[:, i].unsqueeze(1)
            else:
                next_token = output[:, -1:].argmax(dim=-1)
            
            decoding_tokens = torch.cat([decoding_tokens, next_token], dim=1)
        
        return torch.cat(outputs, dim=1)
    
    def generate(self, encoder_output, max_length=None):
        """
        Generate sequence during inference
        Args:
            encoder_output: Encoded image features
            max_length: Optional override for max sequence length
        """
        if max_length is None:
            max_length = self.max_seq_length
            
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Start with EOS token
        current_tokens = torch.full(
            (batch_size, 1), 
            self.tokenizer.eos_token, 
            device=device
        )
        
        for _ in range(max_length - 1):
            # Embed current sequence
            token_embeddings = self.token_embedding(current_tokens)
            token_embeddings = self.pos_encoder(token_embeddings)
            
            # Create mask for current sequence
            tgt_mask = self.generate_square_subsequent_mask(
                current_tokens.size(1)
            ).to(device)
            
            # Get next token prediction
            transformer_output = self.transformer(
                encoder_output,
                token_embeddings,
                tgt_mask=tgt_mask
            )
            logits = self.output(transformer_output[:, -1:])
            next_token = torch.argmax(logits, dim=-1)
            
            # Add predicted token to sequence
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            
            # Stop if all sequences have generated EOS token
            if (next_token == self.tokenizer.eos_token).all():
                break
                
        return current_tokens[:, 1:]  # Remove initial EOS token

def scheduled_teacher_forcing(epoch, min_ratio=0.2, max_ratio=1.0, num_epochs=100):
    """Calculate teacher forcing ratio based on training progress"""
    ratio = max_ratio - (max_ratio - min_ratio) * (epoch / num_epochs)
    return max(min_ratio, ratio)