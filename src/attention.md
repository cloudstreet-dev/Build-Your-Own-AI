# Attention: The Mechanism That Changed Everything

In 2017, a team at Google published a paper called "Attention Is All You Need." The title was a provocation — previous models used attention as a supplement to recurrent networks. The paper's claim was that you didn't need the recurrent networks at all. Attention alone was sufficient.

They were right, and that's what made everything that came after possible.

## The Problem Attention Solves

Before attention, sequence models processed text left to right, maintaining a hidden state that summarized everything seen so far. The problem: by the time you're processing the 500th word, the model's memory of the 1st word has been squished through 499 transformations. It's like whispering a message down a line of 499 people — the original information degrades.

Attention solves this by allowing any position in the sequence to directly look at any other position. No degradation. No squishing. Direct access.

When you're reading the word "it" in "The animal didn't cross the street because **it** was too tired," you need to know that "it" refers to "animal," not "street." Attention lets the model look back at "animal" and "street" simultaneously and decide which one "it" is about, based on learned patterns.

## The Attention Mechanism

Here's the core idea, stated plainly before any code:

For each position in the sequence, we want to compute a weighted average of all other positions' representations. The weights should reflect *relevance* — positions that are more relevant to the current position should have higher weights.

To compute these relevance weights, we need three things:
- A **Query** (Q): what the current position is looking for
- **Keys** (K): what each position offers as a lookup
- **Values** (V): what each position actually contributes if selected

The query-key dot product measures relevance. Softmax turns these into weights. Those weights are applied to the values. That's attention.

If you're thinking "why does this need three separate things, couldn't we just compare the embeddings directly?" — that's a good question with a good answer. Coming right up.

## Self-Attention: The Full Picture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The fundamental attention operation.

    Args:
        Q: Queries  [batch, seq_len, d_k]
        K: Keys     [batch, seq_len, d_k]
        V: Values   [batch, seq_len, d_v]
        mask: Optional boolean mask (True = ignore this position)

    Returns:
        output: [batch, seq_len, d_v]
        weights: [batch, seq_len, seq_len]  (useful for visualization)
    """
    d_k = Q.size(-1)

    # Step 1: Compute attention scores
    # Q @ K^T: for each query, how much does it match each key?
    # Shape: [batch, seq_len_q, seq_len_k]
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Step 2: Scale by sqrt(d_k)
    # Without this, scores grow large as d_k grows, causing softmax to saturate.
    # Saturated softmax = near-zero gradients = model stops learning.
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply mask (for causal attention — can't look into the future)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # Step 4: Softmax to get probabilities
    # Each query now has a probability distribution over all keys
    weights = F.softmax(scores, dim=-1)

    # Step 5: Weighted sum of values
    output = torch.matmul(weights, V)

    return output, weights

# Let's see it work on a tiny example
seq_len = 4
d_k = 8   # key/query dimension
d_v = 8   # value dimension

# Random queries, keys, values for demonstration
Q = torch.randn(1, seq_len, d_k)
K = torch.randn(1, seq_len, d_k)
V = torch.randn(1, seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)

print(f"Input sequence length: {seq_len}")
print(f"Q, K shape: {Q.shape}")
print(f"V shape:    {V.shape}")
print(f"Output shape: {output.shape}")
print(f"\nAttention weights (each row sums to 1):")
print(weights[0].detach())
print(f"\nRow sums: {weights[0].sum(dim=-1).detach()}")
```

## Why Q, K, V? (The Answer)

Let's say you tried to do attention using the embeddings directly: compare position `i` to position `j` by taking the dot product of their embedding vectors. The problem: the embedding vector has to serve two roles simultaneously. It has to describe what *this* token *is* (to answer queries from other positions) and what *this* token *wants* (to query other positions). These can be different things.

For a concrete example: the word "bank" in "river bank" and "bank account" has the same embedding but should answer queries differently. The Q, K, V projections are learned linear transformations that let the model develop separate "search vocabularies" for querying vs. being queried.

Technically:
- `Q = X @ W_Q` — transform embedding into "what am I looking for?"
- `K = X @ W_K` — transform embedding into "what do I offer for lookup?"
- `V = X @ W_V` — transform embedding into "what do I actually contribute?"

Where `X` is the input embedding and `W_Q`, `W_K`, `W_V` are learned weight matrices. The model learns all three matrices end-to-end.

## Multi-Head Attention

One attention head looks at the sequence through one lens. Multi-head attention runs several attention operations in parallel, each with different learned projections, then combines the results.

Why? Different heads learn to attend to different types of relationships. One head might learn syntactic dependencies (subject-verb agreement). Another might learn coreference (pronoun resolution). Another might track long-range topic consistency. With a single head, these would compete. With multiple heads, they each get their own attention pattern.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension per head

        # These project the input into Q, K, V for ALL heads at once
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # output projection

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (n_heads, d_k) then transpose.
        Input:  [batch, seq_len, d_model]
        Output: [batch, n_heads, seq_len, d_k]
        """
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse of split_heads.
        Input:  [batch, n_heads, seq_len, d_k]
        Output: [batch, seq_len, d_model]
        """
        batch, n_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        # Project to Q, K, V
        Q = self.split_heads(self.W_q(x))  # [batch, n_heads, seq_len, d_k]
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # Scaled dot-product attention on each head
        # Q, K, V are [batch, n_heads, seq_len, d_k]
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        attended = torch.matmul(weights, V)  # [batch, n_heads, seq_len, d_k]

        # Combine heads and project
        combined = self.combine_heads(attended)        # [batch, seq_len, d_model]
        return self.W_o(combined)                      # [batch, seq_len, d_model]

# Test it
d_model = 64
n_heads = 8
seq_len = 10
batch_size = 2

mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
x = torch.randn(batch_size, seq_len, d_model)
out = mha(x)

print(f"MultiHeadAttention")
print(f"  Input:  {x.shape}")
print(f"  Output: {out.shape}")
print(f"  Heads: {n_heads}, d_k per head: {d_model // n_heads}")
print(f"  Parameters: {sum(p.numel() for p in mha.parameters()):,}")
```

## Causal Attention (For Autoregressive Models)

Language models generate text left to right. During training, when predicting position `i`, the model cannot use information from positions `i+1, i+2, ...` — those tokens don't exist yet at inference time.

We enforce this with a **causal mask**: a triangular mask that prevents each position from attending to future positions.

```python
def causal_mask(seq_len: int) -> torch.Tensor:
    """
    Returns a boolean mask where True means 'ignore this position.'
    Shape: [seq_len, seq_len]
    """
    # Upper triangle (excluding diagonal) is True (masked out)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    return mask

mask = causal_mask(6)
print("Causal mask (True = masked out, cannot attend):")
print(mask.int())
print()
print("Reading the mask: row = query position, col = key position")
print("Position 0 can only see position 0 (itself)")
print("Position 3 can see positions 0, 1, 2, 3 (but not 4 or 5)")
```

```
Causal mask (True = masked out, cannot attend):
tensor([[0, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]])
```

In the attention computation, masked positions get `-inf` before softmax. `softmax(-inf) = 0`, so those positions get zero weight. The model literally cannot see future tokens — not because we hide them, but because they contribute zero to the weighted average.

## Visualizing Attention

Attention weights are interpretable (to a degree). Let's visualize what a trained attention head might look like:

```python
import torch
import matplotlib

# Simulate what attention weights might look like for a sentence
sentence = ["The", "cat", "sat", "on", "the", "mat", "."]
seq_len = len(sentence)

# In a real model, these come from the trained Q, K, V projections.
# Here we'll construct a plausible pattern for illustration.
# Imagine head 0 learned to track subject-verb relationships:
weights_head0 = torch.tensor([
    [0.8, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01],  # "The" attends mostly to itself
    [0.2, 0.6, 0.1,  0.05, 0.02, 0.02, 0.01],  # "cat" attends to "The", itself
    [0.05, 0.4, 0.4, 0.1, 0.02, 0.02, 0.01],   # "sat" attends to "cat" (its subject)
    [0.05, 0.1, 0.1, 0.6, 0.1,  0.04, 0.01],   # "on" attends mostly to itself
    [0.3,  0.1, 0.1, 0.1, 0.3,  0.09, 0.01],   # "the" splits attention
    [0.05, 0.1, 0.1, 0.1, 0.1,  0.5,  0.05],   # "mat" attends to itself
    [0.1,  0.1, 0.3, 0.1, 0.1,  0.2,  0.1 ],   # "." attends broadly
])

# Check rows sum to 1
assert torch.allclose(weights_head0.sum(dim=-1), torch.ones(seq_len), atol=1e-4)

print("Attention weight matrix (row = query, col = key):")
print(f"{'':>8}", end="")
for w in sentence:
    print(f"{w:>8}", end="")
print()
for i, (query_word, row) in enumerate(zip(sentence, weights_head0)):
    print(f"{query_word:>8}", end="")
    for val in row:
        # Visual representation: shade by weight
        bar = "█" * int(val * 8)
        print(f"{val:>8.2f}", end="")
    print()
```

Real attention heads, once trained, reveal fascinating patterns: some track syntactic dependencies, some handle coreference, some appear to do less interpretable but apparently useful things. The field of mechanistic interpretability is dedicated to reverse-engineering what each head learned.

## The Computational Complexity

Attention has quadratic complexity in sequence length: `O(n²)`. Every position attends to every other position, and there are `n²` pairs.

This is the core scalability challenge for transformers. A sequence of 1,000 tokens requires 1,000,000 attention computations. 100,000 tokens requires 10,000,000,000. This is why long-context models are hard and expensive, and why much research has gone into approximate attention mechanisms (Longformer, FlashAttention, etc.).

```python
def attention_flops(seq_len: int, d_model: int, n_heads: int) -> dict:
    """Estimate FLOPs for one attention layer."""
    d_k = d_model // n_heads

    # QKV projections: 3 * seq_len * d_model * d_model
    qkv_proj = 3 * seq_len * d_model * d_model

    # Attention scores: n_heads * seq_len * seq_len * d_k
    attn_scores = n_heads * seq_len * seq_len * d_k

    # Attention output: n_heads * seq_len * seq_len * d_k
    attn_output = n_heads * seq_len * seq_len * d_k

    # Output projection: seq_len * d_model * d_model
    out_proj = seq_len * d_model * d_model

    total = qkv_proj + attn_scores + attn_output + out_proj
    return {
        "qkv_projection": qkv_proj,
        "attention_scores": attn_scores,
        "attention_output": attn_output,
        "output_projection": out_proj,
        "total_flops": total,
    }

print("FLOPs breakdown for one attention layer:")
print()
for seq_len in [128, 1024, 8192]:
    flops = attention_flops(seq_len, d_model=768, n_heads=12)
    print(f"seq_len={seq_len:6d}: {flops['total_flops']/1e9:.2f}B FLOPs")
    if seq_len == 128:
        for k, v in flops.items():
            print(f"  {k}: {v:,}")
```

## Putting It Together

Here's the mental model you should carry forward:

1. Each token in the sequence broadcasts a **Key** (what I can offer) and a **Value** (my actual content)
2. Each token sends out a **Query** (what I'm looking for)
3. The query from position `i` dot-products with all keys, giving relevance scores
4. Softmax converts scores to weights; these weight the values
5. Position `i`'s output is the weighted average of all values
6. Multiple heads do this independently and combine results
7. Causal models mask out future positions

The result: every position has access to context from any other position, with the model learning which positions are relevant for which tasks. No information bottleneck. No vanishing memory. Just weighted summation, done `n_heads` times in parallel.

That's why attention changed everything.
