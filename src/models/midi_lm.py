import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings once in log space
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MidiLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        pad_token_id: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, tokens: torch.Tensor):
        """
        Args:
            tokens: (B, L) LongTensor with token IDs
        Returns:
            logits: (B, L, vocab_size)
        """
        # Create causal mask
        seq_len = tokens.size(1)
        mask = self._generate_square_subsequent_mask(seq_len, tokens.device)

        emb = self.embedding(tokens) * math.sqrt(self.d_model)
        emb = self.pos_encoder(emb)

        output = self.transformer(emb, mask=mask)
        logits = self.output_layer(output)
        return logits

    def _generate_square_subsequent_mask(self, sz: int, device) -> torch.Tensor:
        # True where we should NOT attend to future positions
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    @torch.no_grad()
    def generate(self, start_tokens: torch.Tensor, max_length: int = 128, temperature: float = 1.0):
        # start_tokens shape: (B, start_len)
        self.eval()
        generated = start_tokens.clone()

        for _ in range(max_length):
            logits = self(generated)
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated