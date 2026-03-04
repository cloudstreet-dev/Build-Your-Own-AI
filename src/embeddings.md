# Embeddings: Meaning as Geometry

A token is a number. A number is not meaning. To get from one to the other, we need embeddings.

An embedding is a vector — a list of floating-point numbers — that represents a token in a high-dimensional space. The magic (and it genuinely is, once you see it working) is that *distance and direction in this space correspond to semantic relationships*. Words that mean similar things end up near each other. The direction from "king" to "queen" is approximately the same as the direction from "man" to "woman".

You've probably heard this. Here's what it actually means computationally.

## The Lookup Table

At its simplest, an embedding layer is just a lookup table: a matrix of shape `[vocab_size, embedding_dim]` where each row is the embedding for one token.

```python
import torch
import torch.nn as nn

vocab_size = 50257   # GPT-2's vocabulary size
embedding_dim = 768  # GPT-2's embedding dimension

# This is literally just a matrix
embedding_table = nn.Embedding(vocab_size, embedding_dim)

# Look up token ID 42
token_id = torch.tensor([42])
vec = embedding_table(token_id)
print(f"Token 42 embedding shape: {vec.shape}")
print(f"First 5 values: {vec[0, :5].detach()}")
```

```
Token 42 embedding shape: torch.Size([1, 768])
First 5 values: tensor([ 0.3251, -1.2847,  0.0923,  0.7615, -0.4381])
```

At initialization, these values are random. They have no meaning. The model learns the embeddings during training — the matrix is updated via gradient descent just like any other parameter. By the end of training, token embeddings that appear in similar contexts end up with similar vectors.

Why does that happen? Because the model learns to predict the next token, and tokens that appear in similar positions (after similar words, before similar words) will cause similar prediction errors, which cause similar gradient updates. Meaning emerges from co-occurrence patterns.

## Geometry of Meaning

Let's build a tiny embedding space from scratch using a real dataset to make this concrete:

```python
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

# A tiny corpus — enough to learn some structure
corpus = [
    "the cat sat on the mat",
    "the dog sat on the floor",
    "cats and dogs are animals",
    "animals eat food",
    "cats eat fish",
    "dogs eat meat",
    "fish live in water",
    "dogs live in houses",
    "cats live in houses too",
    "the king ruled the kingdom",
    "the queen ruled the kingdom",
    "the king and queen",
    "man and woman",
    "the man walked the dog",
    "the woman walked the cat",
]

# Build vocabulary
words = " ".join(corpus).split()
vocab = ["<PAD>"] + list(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
V = len(vocab)
print(f"Vocabulary size: {V} words")

# Simple embedding model: predict if two words appear near each other (skip-gram style)
class SimpleEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.context = nn.Embedding(vocab_size, embed_dim)

    def forward(self, target, context):
        # dot product of target and context embeddings
        t = self.embed(target)      # [batch, dim]
        c = self.context(context)   # [batch, dim]
        return (t * c).sum(dim=1)   # [batch]

# Generate training pairs: (word, nearby_word) = positive examples
def get_pairs(corpus, window=2):
    pairs = []
    for sentence in corpus:
        words = sentence.split()
        for i, word in enumerate(words):
            for j in range(max(0, i-window), min(len(words), i+window+1)):
                if i != j:
                    pairs.append((word2idx[word], word2idx[words[j]]))
    return pairs

pairs = get_pairs(corpus)
print(f"Training pairs: {len(pairs)}")

# Train a tiny embedding model
model = SimpleEmbedding(V, embed_dim=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

targets = torch.tensor([p[0] for p in pairs])
contexts = torch.tensor([p[1] for p in pairs])

# Generate negative samples (random words that are NOT near the target)
def negative_sample(targets, vocab_size, k=5):
    return torch.randint(0, vocab_size, (len(targets), k))

for epoch in range(300):
    neg = negative_sample(targets, V)

    # Positive scores (should be high)
    pos_scores = model(targets, contexts)

    # Negative scores (should be low)
    neg_scores = model(
        targets.unsqueeze(1).expand_as(neg).reshape(-1),
        neg.reshape(-1)
    ).reshape(len(targets), -1)

    # Loss: positive scores up, negative scores down
    pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-8).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-8).mean()
    loss = pos_loss + neg_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")
```

Now let's look at the geometry:

```python
# Extract learned embeddings
def get_embedding(word):
    idx = word2idx[word]
    with torch.no_grad():
        return model.embed(torch.tensor([idx])).squeeze().numpy()

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def most_similar(word, top_k=5):
    vec = get_embedding(word)
    scores = []
    for w in vocab:
        if w == word or w == "<PAD>":
            continue
        sim = cosine_similarity(vec, get_embedding(w))
        scores.append((w, sim))
    return sorted(scores, key=lambda x: -x[1])[:top_k]

# Check semantic neighbors
for word in ["cat", "dog", "king", "water"]:
    if word in word2idx:
        neighbors = most_similar(word)
        neighbor_str = ", ".join(f"{w}({s:.2f})" for w, s in neighbors)
        print(f"  {word:10s} → {neighbor_str}")
```

With only 15 sentences and 8-dimensional embeddings, you'll see that "cat" and "dog" land near each other, and "king" and "queen" cluster together. It's not perfect — the corpus is tiny — but the structure emerges from pure co-occurrence statistics. Nobody told the model that cats and dogs are both animals.

## Cosine Similarity: The Right Distance Metric

When comparing embeddings, you almost always want cosine similarity rather than Euclidean distance:

```python
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two vectors."""
    a = a.float()
    b = b.float()
    return (torch.dot(a.flatten(), b.flatten()) /
            (a.norm() * b.norm())).item()

# Why cosine, not Euclidean distance?
# Consider: "dog" might have a larger-magnitude embedding than "a"
# because "dog" appears in more varied contexts.
# Cosine similarity normalizes for magnitude — it only cares about direction.

a = torch.tensor([2.0, 4.0])   # some direction, larger magnitude
b = torch.tensor([1.0, 2.0])   # same direction, smaller magnitude
c = torch.tensor([3.0, 1.0])   # different direction

print(f"cosine(a, b) = {cosine_sim(a, b):.3f}")  # should be ~1.0 (same direction)
print(f"cosine(a, c) = {cosine_sim(a, c):.3f}")  # should be less

import torch.nn.functional as F
# PyTorch has this built in:
a_norm = F.normalize(a.unsqueeze(0), dim=1)
b_norm = F.normalize(b.unsqueeze(0), dim=1)
print(f"Using F.normalize: {(a_norm * b_norm).sum().item():.3f}")
```

## The "King - Man + Woman = Queen" Thing

This is the famous word vector analogy. Let's see why it works:

The intuition: the vector from "man" to "king" encodes the concept of "royalty applied to a male." If we start from "woman" and apply the same vector offset, we should land near "queen."

```python
# With real embeddings (let's use a pre-trained example)
# In your own tiny model above, the relationships might be weak due to data size,
# but with proper training on real data they're clear.

def analogy(a, b, c, word2idx, idx2word, model, top_k=3):
    """Find d such that a:b :: c:d"""
    va = get_embedding(a)
    vb = get_embedding(b)
    vc = get_embedding(c)
    target = vb - va + vc  # the "offset" vector

    scores = []
    for word in vocab:
        if word in [a, b, c, "<PAD>"]:
            continue
        sim = cosine_similarity(target, get_embedding(word))
        scores.append((word, sim))
    return sorted(scores, key=lambda x: -x[1])[:top_k]

# With our tiny corpus:
# "king" is to "queen" as "man" is to ?
result = analogy("king", "queen", "man", word2idx, idx2word, model)
print("king:queen :: man:?")
for word, score in result:
    print(f"  {word}: {score:.3f}")
```

With a tiny dataset, the results will be noisy. With embeddings trained on millions of documents, they're remarkably clean. The point is that the *mechanism* is just vector arithmetic — addition and subtraction of floating-point arrays.

## Positional Embeddings

Here's something the story so far is missing: word embeddings encode *what* a word means, but not *where* it appears. "Dog bites man" and "Man bites dog" have the same tokens in different order. Order matters enormously.

Transformers handle this by adding a second embedding: a **positional embedding** that encodes position.

```python
import torch
import math

def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    The original transformer positional encoding from 'Attention Is All You Need'.
    Returns shape: [max_len, d_model]
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

    # Frequencies decrease geometrically
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    # Even dimensions: sine; odd dimensions: cosine
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

pe = sinusoidal_positional_encoding(max_len=100, d_model=64)
print(f"Positional encoding shape: {pe.shape}")
print(f"Position 0, first 8 dims: {pe[0, :8]}")
print(f"Position 1, first 8 dims: {pe[1, :8]}")
print(f"Position 50, first 8 dims: {pe[50, :8]}")
```

The sinusoidal design has a nice property: `PE[pos + k]` can be expressed as a linear function of `PE[pos]`, which means the model can learn to attend to "k positions ahead" without seeing that offset in training.

Modern models often use **learned positional embeddings** — another lookup table, this one indexed by position rather than token ID. The model learns the best positional encoding for its data. This is simpler and works well in practice.

```python
class TokenAndPositionEmbedding(nn.Module):
    """
    Combines token embeddings and positional embeddings.
    This is the input layer of a transformer.
    """
    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x):
        # x: [batch, seq_len] of token IDs
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        tok = self.token_emb(x)       # [batch, seq_len, d_model]
        pos = self.pos_emb(positions)  # [1, seq_len, d_model]

        # Scaling by sqrt(d_model) is from the original paper.
        # It prevents token embeddings from being swamped by positional ones.
        return self.dropout(tok * math.sqrt(self.d_model) + pos)

# Test it
emb_layer = TokenAndPositionEmbedding(
    vocab_size=1000,
    d_model=64,
    max_seq_len=128
)

# A batch of 2 sequences, each 10 tokens long
batch = torch.randint(0, 1000, (2, 10))
output = emb_layer(batch)
print(f"Input shape:  {batch.shape}")
print(f"Output shape: {output.shape}")  # [2, 10, 64]
```

## The Shape That Follows You

Notice the output shape: `[batch_size, sequence_length, embedding_dim]`. This is the fundamental tensor shape that flows through a transformer. Every subsequent operation — attention, feedforward layers, everything — works on tensors of this shape (or variants of it).

Keep that shape in mind. It'll be important in the next chapter, where we finally get to attention.

## Key Takeaways

- Embeddings convert token IDs into dense vectors of floats
- The vectors are learned during training from co-occurrence patterns
- Semantic relationships appear as geometric relationships (distance, direction)
- Cosine similarity is the right measure for comparing embedding vectors
- Positional embeddings are added to preserve word-order information
- The combined embedding is a `[batch, seq_len, d_model]` tensor

You've just built the front door of a transformer. The rest of the architecture is what happens once your tokens walk in.
