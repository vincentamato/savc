import torch
import torch.nn as nn
from transformers import ViTModel
import math
import random
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from src.data.dataset import MIDITokenizer

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

class TransformerWithStochasticDepth(nn.Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stochastic_depth_prob = 0.1
        self.gradient_checkpointing = False
        
    def _stochastic_depth(self, x, training):
        if not training or random.random() > self.stochastic_depth_prob:
            return x
        return torch.zeros_like(x)
        
    def forward(self, src, tgt, *args, **kwargs):
        output = super().forward(src, tgt, *args, **kwargs)
        return self._stochastic_depth(output, self.training)
    
    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
        # Enable gradient checkpointing for encoder and decoder
        for module in [self.encoder, self.decoder]:
            for layer in module.layers:
                layer.use_checkpoint = True
    
    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        # Disable gradient checkpointing for encoder and decoder
        for module in [self.encoder, self.decoder]:
            for layer in module.layers:
                layer.use_checkpoint = False

class MusicGenerationTransformer(nn.Module):
    def __init__(
        self,
        vit_name="google/vit-large-patch16-384",
        vit_config=None,  # Add this parameter
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
        self.vit = ViTModel.from_pretrained("google/vit-large-patch16-384")
        # Freeze ViT parameters if desired
        for param in self.vit.parameters():
            param.requires_grad = False

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.temperature = temperature

        self.vit_projection = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        self.token_embedding = nn.Embedding(
            self.tokenizer.vocab_size,
            d_model,
            padding_idx=self.tokenizer.special_tokens['PAD']
        )
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, scale=1.0)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.output = nn.Linear(d_model, self.tokenizer.vocab_size)
        self.output.weight = self.token_embedding.weight

        self._init_parameters()

    def _init_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def generate_square_subsequent_mask(self, sz):
        # True where we should NOT attend (future positions)
        return torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)

    def create_padding_mask(self, tokens):
        # Returns a boolean mask with True for padded elements
        return tokens == self.tokenizer.special_tokens['PAD']

    def forward(self, images, target_sequences=None, teacher_forcing_ratio=1.0):
        """
        Forward pass for the model. Supports teacher forcing during training.

        Args:
            images (torch.Tensor): Input images passed through the vision transformer.
            target_sequences (torch.Tensor, optional): Token sequences for training.
            teacher_forcing_ratio (float, optional): Probability of using the ground truth token
                for the next step. Defaults to 1.0.

        Returns:
            torch.Tensor: Model output logits.
        """
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            vit_output = self.vit(images)
            image_features = (
                vit_output.last_hidden_state
                if hasattr(vit_output, "last_hidden_state")
                else vit_output[0]
            )

        encoder_output = self.vit_projection(image_features)

        # If no targets or not training, perform generation
        if target_sequences is None or not self.training:
            return self.generate(encoder_output)

        # Prepare decoder inputs
        batch_size, seq_len = target_sequences.size()
        device = target_sequences.device

        token_embeddings = self.token_embedding(target_sequences)
        token_embeddings = self.pos_encoder(token_embeddings)

        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(device)
        tgt_padding_mask = self.create_padding_mask(target_sequences)

        outputs = []
        decoder_input = target_sequences[:, :1]  # Start with BOS token

        for t in range(1, seq_len):
            tgt_embeddings = self.token_embedding(decoder_input)
            tgt_embeddings = self.pos_encoder(tgt_embeddings)

            transformer_output = self.transformer(
                src=encoder_output,
                tgt=tgt_embeddings,
                tgt_mask=tgt_mask[: decoder_input.size(1), : decoder_input.size(1)],
                tgt_key_padding_mask=tgt_padding_mask[:, : decoder_input.size(1)],
            )

            logits = self.output(transformer_output[:, -1, :])  # Take the last step's output
            outputs.append(logits)

            # Apply teacher forcing
            if torch.rand(1).item() < teacher_forcing_ratio:
                next_token = target_sequences[:, t:t + 1]
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            decoder_input = torch.cat([decoder_input, next_token], dim=1)

        outputs = torch.stack(outputs, dim=1)
        return outputs


    def generate(self, encoder_output, max_length=None, beam_size=5):
        # Implement your generation logic without the iterative decoding in training
        pass

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)