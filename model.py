import math
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax

from flax.traverse_util import path_aware_map
from flax.core import freeze
from flax.training import train_state
from flax import traverse_util

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd = 768
    dropout: float = 0.1
    bias: bool = True

class CausalSelfAttention(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        # each head works on a portion of full embedding.
        assert config.n_embd % config.n_head == 0
        head_size = config.n_embd // config.n_head
        # key, query, value projections for all heads but in a batch.
        self.c_attn = nn.Dense(config.n_embd * 3, use_bias=config.bias)
        # output projection.
        self.c_proj = nn.Dense(config.n_embd, use_bias=config.bias)
        # regularization.
        self.attn_dropout = nn.Dropout(rate=config.dropout)
        self.resid_dropout = nn.Dropout(rate=config.dropout)
        # save params for easy access.
        self.n_head = config.n_head
        self.head_size = head_size
        self.n_embd = config.n_embd

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        B, T, C = x.shape # batch size, sequence length, n_embd
        
        # calculate query, key, values matrix for all heads in batch.
        q, k, v = self.c_attn(x).split(3, axis=-1)
        q = q.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2) # (B, nh, T, hs)
        k = k.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2) # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2) # (B, nh, T, hs)
        
        mask = jnp.tril(jnp.ones((T, T))).reshape(1, 1, T, T) # causal attention mask.
        
        # causal self-attention. self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.swapaxes(-2, -1)) * (1.0 / jnp.sqrt(k.shape[-1])) # (B, nh, T, T)
        att = jnp.where(mask == 0, float('-inf'), att)
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not train)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # re-assamble all heads outputs side by side.
        y = y.swapaxes(1, 2).reshape(B, T, self.n_embd)
        y = self.resid_dropout(self.c_proj(y), deterministic=not train)

        return y
    
class MLP(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        self.c_fc = nn.Dense(4 % config.n_embd, use_bias=config.bias)
        self.c_proj = nn.Dense(config.n_embd, use_bias=config.bias)
        self.dropout = nn.Dropout(rate=config.dropout)

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        x = self.c_fc(x)
        x = nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        x = self.dropout(x, deterministic=not train)
        return x
    
class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        self.ln1 = nn.LayerNorm(epsilon=1e-5)
        self.ln2 = nn.LayerNorm(epsilon=1e-5)
        self.attn = CausalSelfAttention(self.config)
        self.mlp = MLP(self.config)

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        x = x + self.attn(self.ln1(x), train=train)
        x = x + self.mlp(self.ln2(x), train=train)
        return x
    
class GPT(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.wte = nn.Embed(config.vocab_size, config.n_embd) # word token embeddings.
        self.wpe = nn.Embed(config.block_size, config.n_embd) # word position embeddings.
        self.drop = nn.Dropout(rate=config.dropout)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(epsilon=1e-5)

    def __call__(self, idx: jax.Array, *, train: bool, targets: Optional[jax.Array] = None):
        B, T = x.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = jnp.arange(0, T, dtype=jnp.int32).reshape(1, -1) # shape (1, T)

        # compute embedding.
        tok_emb = self.wte(x) # (B, T, n_embd)
        pos_emb = self.wpe(pos) # (1, T, n_embd)
        x = self.drop(tok_emb + pos_emb, deterministic=not train)

        # forward gpt model.
        for block in self.h:
            x = block(x, train=train)
        x = self.ln_f(x) # (B, T, n_embd)

        logits = self.wte.attend(x) # (B, T, vocab_size)

        if targets is not None:
            # if we are given some targets, calculate the loss.
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets) # (B, T)
            loss = jnp.mean(loss) # scalar
        else:
            loss = None

        return logits, loss
    
    def crop_block_size(self, params, block_size: int):
        # model surgery to decrease the block size if necessary.
        # eg. we may load a gpt2 pretrained model (block size 1024)
        # but want to use a smaller block size for simpler models.

        assert 0 < block_size <= self.config.block_size
        self.config.block_size = block_size

        def crop_weights(path: Tuple[str, ...], x):
            if path[-2:] == ('wpe', 'embedding'):
                return x[:block_size]
            return x
        
        return freeze(path_aware_map(crop_weights, params))
