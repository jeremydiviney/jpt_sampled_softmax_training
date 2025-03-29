import time
import os
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from tokenizers import Tokenizer


class JPT1ModelType(Enum):
    STANDARD = "standard"
    STANDARD_SAMPLED = "standard_sampled"


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, num_head, dropout):
        super().__init__()
        assert d_model % num_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # output projection
        self.c_proj = nn.Linear(d_model, d_model)
        self.c_proj.SCALE_RESIDUAL = True  # Mark for scaling

        # regularization
        self.n_head = num_head
        self.n_embd = d_model

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, d_model, dim_feedforward, activation):
        super().__init__()
        self.c_fc = nn.Linear(d_model, dim_feedforward)
        self.activation = activation
        self.c_proj = nn.Linear(dim_feedforward, d_model)
        self.c_proj.SCALE_RESIDUAL = True  # Mark for scaling

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        return x


class TransformerDecoderLayerCustom(nn.Module):

    def __init__(self, d_model, num_head, dropout, dim_feedforward, activation):
        super().__init__()
        self.ln_1 = LlamaRMSNorm(d_model, eps=1e-6)
        self.attn = CausalSelfAttention(d_model, num_head, dropout)
        self.ln_2 = LlamaRMSNorm(d_model, eps=1e-6)
        self.mlp = MLP(d_model, dim_feedforward, activation)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerDecoderCustom(nn.Module):

    def __init__(self, seq_len, d_model, dim_feedforward, num_head, num_layers, dropout, activation):
        super().__init__()
        self.seq_len = seq_len
        self.t_layers = nn.ModuleList(
            [TransformerDecoderLayerCustom(d_model, num_head, dropout, dim_feedforward, activation) for _ in range(num_layers)]
        )

    def forward(self, x):
        # idx is of shape (B, T, C)
        B, T, C = x.size()
        assert T <= self.seq_len, f"Cannot forward sequence of length {T}, sequence length is only {self.seq_len}"

        # forward the blocks of the transformer
        for block in self.t_layers:
            x = block(x)
        return x


class JPT1(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embed_dim: int,
        token_space_dim: int,
        num_head: int,
        num_layers: int,
        dropout: float,
        tokenizer: Tokenizer,
        model_type: JPT1ModelType,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        self.model_type = model_type
        self.tokenizer = tokenizer

        self.token_list = tokenizer.get_vocab()
        self.vocab_size = len(self.token_list)

        self.embeddings = nn.Embedding(self.vocab_size, embed_dim)

        self.lookup_embeddings = self.embeddings

        self.text_token_to_idx = {token: self.token_list[token] for token in self.token_list}

        self.token_space_dim = self.lookup_embeddings.weight.shape[1]

        self.transformer = TransformerDecoderCustom(
            d_model=embed_dim,
            num_head=num_head,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation=nn.GELU(),
            num_layers=num_layers,
            seq_len=seq_len,
        )

        self.fc_ln = LlamaRMSNorm(embed_dim, eps=1e-6)

        self.fc_out = nn.Linear(embed_dim, self.vocab_size)
        # Tie weights - share the embedding matrix with the output projection
        self.embeddings.weight = self.fc_out.weight

        self.temperature = nn.Parameter(torch.tensor(0.07))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02

            if hasattr(module, "SCALE_RESIDUAL") and module.SCALE_RESIDUAL:
                # Scale by depth for residual path projections
                std = 0.02 * (2 * self.num_layers) ** -0.5

            # Use normal initialization with the calculated std
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            # Small normal initialization for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Standard init for LayerNorm
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, LlamaRMSNorm) or "LlamaRMSNorm" in module.__class__.__name__:
            # RMSNorm only has weight parameter (no bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor, do_final_projection: bool) -> torch.Tensor:
        batch_size, seq_len = x.shape
        embedded = self.embeddings(x)

        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        embedded = embedded + self.position_embedding(position_ids)

        x = self.transformer(embedded)  # [B, S, embed_dim]

        x = self.fc_ln(x)

        if do_final_projection:
            output = self.fc_out(x)
        else:
            output = x  # if sample based training, don't do final projection yet

        return output, x

    def get_token_indices(self, text_tokens: list[str]) -> list[int]:
        """
        Convert a list of text tokens to their corresponding indices.
        """
        return [self.text_token_to_idx[token] for token in text_tokens]

    def get_text_token_from_indices(self, indices: np.ndarray) -> np.ndarray:
        shape = indices.shape
        # reshape so we have have a giant batch of 1 token each so the decode_batch will return a bit array as we don't just want a blob of text yet.
        indices = indices.reshape(-1, 1)
        decoded_tokens = self.tokenizer.decode_batch(indices)
        decoded_tokens = np.array(decoded_tokens)
        return decoded_tokens.reshape(shape)
