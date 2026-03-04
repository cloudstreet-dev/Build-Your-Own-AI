# Transformers: Putting It All Together

You have the pieces: embeddings turn tokens into vectors, attention lets positions look at each other, and multi-head attention runs several attention patterns in parallel. Now let's assemble them into an actual transformer.

The full architecture has a few more components you haven't seen yet — a feedforward network, layer normalization, and residual connections — but none of them are complicated. The transformer's genius isn't any single piece; it's how the pieces fit together.

## The Transformer Block

The fundamental unit of a transformer is the **transformer block** (sometimes called a transformer layer). A full model is just N of these blocks stacked on top of each other.

Each block does two things:
1. **Multi-head self-attention**: let each position gather context from others
2. **Position-wise feedforward network**: process each position independently

Both operations are wrapped in **residual connections** and **layer normalization**.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeedForward(nn.Module):
    """
    Position-wise feedforward network.

    Two linear transformations with a GELU activation in between.
    The inner dimension (d_ff) is typically 4x the model dimension.
    This is where most of the model's parameters actually live.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),            # smooth version of ReLU; works better in practice
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One transformer block = attention + feedforward, with residuals and layer norm.

    The "pre-norm" variant (layer norm before attention/FFN, not after) is standard
    in modern models — it trains more stably than the original paper's "post-norm."
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm attention with residual connection
        x = x + self.attn(self.norm1(x), mask=mask)
        # Pre-norm feedforward with residual connection
        x = x + self.ff(self.norm2(x))
        return x
```

Let's pause on the residual connections: `x = x + attention(x)`. This is from ResNets, and it solves a specific problem. Without residuals, deep networks suffer from vanishing gradients — the gradient signal diminishes as it propagates backward through many layers. With residuals, there's always a direct path for gradients to flow through, bypassing the transformation. It lets you stack many blocks without the network degrading.

## The Full Language Model

```python
class MultiHeadAttention(nn.Module):
    """(Same implementation as the previous chapter — included for completeness.)"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        b, h, s, dk = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, self.d_model)

    def forward(self, x, mask=None):
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        weights = self.dropout(F.softmax(scores, dim=-1))
        return self.W_o(self.combine_heads(torch.matmul(weights, V)))


class GPTLanguageModel(nn.Module):
    """
    A small GPT-style language model.
    Architecture:
      - Token + positional embeddings
      - N transformer blocks
      - Final layer norm
      - Linear projection to vocabulary (the "language model head")
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or d_model * 4

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: the embedding matrix and the LM head share weights.
        # This is a standard trick that reduces parameters and improves performance.
        # The intuition: if token X has a large embedding, the model should be
        # more likely to predict X, which happens when the LM head row for X
        # has a large dot product with the final hidden state.
        self.lm_head.weight = self.token_emb.weight

        # Initialize weights (standard practice from GPT-2 paper)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx:     [batch, seq_len] — token indices
            targets: [batch, seq_len] — target token indices (for computing loss)

        Returns:
            logits: [batch, seq_len, vocab_size]
            loss:   scalar cross-entropy loss (if targets provided, else None)
        """
        batch, seq_len = idx.shape
        device = idx.device

        # Build causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )

        # Embeddings
        positions = torch.arange(seq_len, device=device)
        x = self.dropout(
            self.token_emb(idx) * math.sqrt(self.token_emb.embedding_dim)
            + self.pos_emb(positions)
        )

        # N transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # Final norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Reshape for cross_entropy: [batch*seq_len, vocab_size] vs [batch*seq_len]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,  # ignore padding
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """
        Autoregressively generate tokens given a prompt.
        """
        max_seq_len = self.pos_emb.num_embeddings

        for _ in range(max_new_tokens):
            # Trim to max_seq_len
            idx_cond = idx[:, -max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Take only the last position's logits (predict the next token)
            logits = logits[:, -1, :] / temperature  # [batch, vocab_size]

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]

            # Append and continue
            idx = torch.cat([idx, next_token], dim=1)

        return idx
```

## Building a Tiny GPT and Watching It Learn

Let's build the smallest GPT that's still interesting: a character-level model trained on a tiny dataset.

```python
import torch
import torch.nn.functional as F

# Our training data: a small poem-like text
text = """
To be or not to be that is the question
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune
Or to take arms against a sea of troubles
And by opposing end them to die to sleep
No more and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to tis a consummation
""".strip() * 10  # repeat to get more training data

# Build character-level vocabulary
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]

def decode(ids: list[int]) -> str:
    return ''.join(itos[i] for i in ids)

print(f"Vocabulary size: {vocab_size} characters")
print(f"Characters: {chars}")
print(f"Text length: {len(text)} characters → {len(encode(text))} tokens")
print()

# Encode everything
data = torch.tensor(encode(text), dtype=torch.long)

# Train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Model configuration (deliberately tiny)
config = {
    "vocab_size": vocab_size,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 4,
    "max_seq_len": 64,
    "dropout": 0.1,
}

model = GPTLanguageModel(**config)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

def get_batch(data, batch_size=32, block_size=64):
    """Sample random batches for training."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# Quick training run
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print("Training...")
for step in range(500):
    x, y = get_batch(train_data)
    logits, loss = model(x, targets=y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
    optimizer.step()

    if step % 100 == 0:
        # Evaluate on val
        with torch.no_grad():
            xv, yv = get_batch(val_data)
            _, val_loss = model(xv, targets=yv)
        print(f"  step {step:4d}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")

print()

# Generate some text
print("Generated text:")
prompt = "To be"
context = torch.tensor([encode(prompt)], dtype=torch.long)
generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=10)
print(prompt + decode(generated[0, len(encode(prompt)):].tolist()))
```

After 500 steps on this tiny dataset, the model should produce something that's vaguely English-shaped — not coherent sentences, but something that's clearly learned that certain letters follow other letters with specific frequencies. It has genuinely learned something from your data.

## Parameter Counting: Where Are the Weights?

```python
def count_parameters(model: nn.Module) -> dict:
    """Break down parameter count by component."""
    components = {}

    total = 0
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue  # skip non-leaf modules to avoid double counting
        if sum(p.numel() for p in module.parameters(recurse=False)) == 0:
            continue
        count = sum(p.numel() for p in module.parameters(recurse=False))
        total += count
        # Simplify the name for display
        short_name = name.split('.')[-2] + '.' + name.split('.')[-1] if '.' in name else name
        components[name] = count

    return components, total

# Build a slightly larger model to see the breakdown
model_medium = GPTLanguageModel(
    vocab_size=50257,  # GPT-2's vocab size
    d_model=768,
    n_heads=12,
    n_layers=12,
    max_seq_len=1024,
)

total = sum(p.numel() for p in model_medium.parameters())
print(f"A GPT-2 scale model: {total/1e6:.1f}M parameters")
print()

# Break it down
emb_params = sum(p.numel() for p in model_medium.token_emb.parameters())
pos_params = sum(p.numel() for p in model_medium.pos_emb.parameters())
block_params = sum(p.numel() for p in model_medium.blocks.parameters())
head_params = sum(p.numel() for p in model_medium.lm_head.parameters())

print(f"  Token embeddings:      {emb_params/1e6:7.2f}M  ({100*emb_params/total:.1f}%)")
print(f"  Positional embeddings: {pos_params/1e6:7.2f}M  ({100*pos_params/total:.1f}%)")
print(f"  Transformer blocks:    {block_params/1e6:7.2f}M  ({100*block_params/total:.1f}%)")
print(f"  LM head (tied):        {head_params/1e6:7.2f}M  (shared with embeddings)")
print(f"  Total unique:          {total/1e6:7.2f}M")
```

The feedforward layers inside each block hold the majority of the parameters — roughly 2/3. Each block's FFN has two matrices of shape `[d_model, d_ff]` and `[d_ff, d_model]` where `d_ff = 4 * d_model = 3072`. That's `2 * 768 * 3072 = 4.7M` parameters per block, for 12 blocks = 56M just in FFN layers.

## Data Flow Summary

Let's trace a single forward pass explicitly:

```
Input: [batch=2, seq_len=10]  — token IDs, integers

1. Token embedding:     [2, 10] → [2, 10, 768]
2. Add positional emb:  [2, 10, 768] (same shape, elementwise add)
3. Dropout:             [2, 10, 768]

For each of 12 transformer blocks:
  4. LayerNorm:          [2, 10, 768]
  5. Multi-head attn:    [2, 10, 768] → [2, 10, 768]  (same shape, no sequence reduction)
  6. Residual add:       [2, 10, 768]
  7. LayerNorm:          [2, 10, 768]
  8. Feedforward:        [2, 10, 768] → [2, 10, 3072] → [2, 10, 768]
  9. Residual add:       [2, 10, 768]

10. Final LayerNorm:    [2, 10, 768]
11. LM head (linear):  [2, 10, 768] → [2, 10, 50257]  — one score per vocab item

Output: [2, 10, 50257]  — logits over vocabulary for each position
```

The output at position `i` is a probability distribution over the entire vocabulary — the model's prediction for what token comes at position `i+1`. This is computed for every position simultaneously during training (thanks to the causal mask preventing peeking), which is what makes transformers trainable in parallel unlike RNNs.

## The Scaled Transformer Family

The architecture above is the decoder-only transformer (GPT-style). Variations you'll encounter:

| Variant | Architecture | Used for |
|---------|-------------|----------|
| Decoder-only (GPT) | Causal self-attention | Text generation, language modeling |
| Encoder-only (BERT) | Bidirectional self-attention | Classification, embeddings |
| Encoder-decoder (T5) | Encoder + cross-attention decoder | Translation, summarization |

In encoder-only models, there's no causal mask — every position can attend to every other position. They're trained differently (masked language modeling, not next-token prediction). In encoder-decoder models, the decoder cross-attends to the encoder's output (Q from decoder, K/V from encoder).

We're focused on decoder-only models because that's what GPT-3, GPT-4, Claude, Llama, and most current language models are.

## What You've Built

Let's be specific about what your tiny model can and cannot do. After 500 steps on a 1KB text file, it has learned:

- Character-level frequency distributions
- Some common character sequences (it'll rarely generate impossible sequences like "zxqjv")
- Very rough word-boundary patterns

It has not learned:
- Semantics
- Grammar
- Any facts about the world

That's expected. The architecture is correct. What's missing is scale — more data, more parameters, more training steps. The transformer you've built and the transformer powering GPT-4 are the same architecture; GPT-4 just has orders of magnitude more of everything.

That's the honest secret of this field: the architecture itself is not that complicated. The scale is what creates the emergent capabilities. And scale requires training — which is the next chapter.
