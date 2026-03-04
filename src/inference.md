# Inference: Running What You Built

Training is over. The weights are frozen. Now what?

Inference is the process of actually using a model to generate text. It sounds like the boring epilogue after the exciting training story, but inference has its own rich set of decisions — about how to sample, how to manage memory, how to handle the context window — that substantially affect what you get out of the model.

The difference between temperature 0.1 and temperature 1.0 can be the difference between a useful assistant and a very confident disaster.

## The Autoregressive Loop

Language models generate one token at a time. To generate a 100-token response, you run 100 forward passes. Each pass takes the existing sequence (prompt + all tokens generated so far) and outputs a probability distribution. You sample from that distribution, append the result, and repeat.

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(
    model,
    prompt_tokens: list[int],
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    repetition_penalty: float = 1.0,
    stop_tokens: list[int] = None,
) -> list[int]:
    """
    Full-featured autoregressive generation.
    Returns only the newly generated tokens (not the prompt).
    """
    model.eval()

    # Working context: starts with the prompt
    context = torch.tensor([prompt_tokens], dtype=torch.long)

    generated = []
    stop_tokens = set(stop_tokens or [])

    for _ in range(max_new_tokens):
        # Forward pass: get logits for the last position
        logits, _ = model(context)
        logits = logits[0, -1, :]  # [vocab_size] — prediction for NEXT token

        # Apply repetition penalty
        # Already-generated tokens get their logits divided, making them less likely
        if repetition_penalty != 1.0:
            for token_id in set(context[0].tolist()):
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

        # Apply temperature
        logits = logits / temperature

        # Apply top-k filtering
        if top_k is not None:
            # Zero out all but the top k logits
            top_k_val = min(top_k, logits.size(-1))
            kth_val = torch.topk(logits, top_k_val).values[-1]
            logits[logits < kth_val] = float('-inf')

        # Apply top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift right to include the token that crosses the threshold
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        # Check stopping condition
        if next_token in stop_tokens:
            break

        generated.append(next_token)
        context = torch.cat([context, torch.tensor([[next_token]])], dim=1)

        # Truncate context if it exceeds max_seq_len
        if context.size(1) > model.pos_emb.num_embeddings:
            context = context[:, -model.pos_emb.num_embeddings:]

    return generated
```

## Temperature: The Most Important Knob

Temperature controls the "randomness" of sampling. Here's exactly what it does:

```python
import torch
import torch.nn.functional as F

def temperature_demo():
    """Show how temperature reshapes the probability distribution."""

    # Imagine these are raw logits from the model for 5 possible next tokens
    logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.1])
    tokens = ["the", "a", "an", "this", "some"]

    print("Effect of temperature on probability distribution:")
    print(f"{'Token':>8}", end="")
    for temp in [0.1, 0.5, 1.0, 2.0]:
        print(f"  T={temp:3.1f}", end="")
    print()
    print("-" * 50)

    for i, token in enumerate(tokens):
        print(f"{token:>8}", end="")
        for temp in [0.1, 0.5, 1.0, 2.0]:
            scaled = logits / temp
            probs = F.softmax(scaled, dim=-1)
            print(f"  {probs[i].item():5.3f}", end="")
        print()

    print()
    print("Observations:")
    print("  T=0.1: Nearly all probability on 'the' (almost deterministic)")
    print("  T=1.0: Original distribution (what the model actually learned)")
    print("  T=2.0: More uniform — model is less decisive, more 'creative'")
    print()
    print("T→0: Greedy decoding (always pick the highest-probability token)")
    print("T→∞: Uniform random sampling (complete chaos)")

temperature_demo()
```

**Rule of thumb:**
- `T < 0.5`: Very focused, repetitive, safe
- `T = 0.7-0.9`: Good for most creative tasks
- `T = 1.0`: Model's raw distribution
- `T > 1.0`: Higher diversity, more unusual outputs, higher risk of incoherence

## Greedy vs. Sampling: A Subtle Point

You might think the best strategy is always to pick the highest-probability token. It's not.

```python
def show_why_greedy_fails():
    """
    Greedy decoding can get stuck in degenerate loops.
    Sampling avoids this by introducing diversity.
    """
    # Imagine this simplified token sequence
    # After "The cat sat on the mat", greedy might predict:
    # "the" → "cat" → "sat" → "on" → "the" → "mat" → "the" → "cat" → ...

    # This happens because at each step, the locally optimal choice
    # leads to a globally poor sequence.

    # Sampling breaks the loop by occasionally choosing lower-probability tokens.

    print("Greedy decoding problem:")
    print("  'The cat sat on the mat the cat sat on the mat the cat...'")
    print()
    print("Sampling (temperature=0.8):")
    print("  'The cat sat on the mat and watched the birds outside.'")
    print()
    print("Greedy is best for: tasks with a single correct answer (math, code completion)")
    print("Sampling is best for: creative tasks, open-ended generation")

show_why_greedy_fails()
```

## Top-K and Top-P: Constraining the Sample

Sampling from the full vocabulary distribution has a problem: low-probability tokens can occasionally be selected, producing genuinely incoherent outputs. Top-k and top-p filter out these tail tokens before sampling.

```python
def compare_sampling_strategies():
    """
    Compare top-k vs top-p on a sample distribution.
    """
    # A model predicting the next word after "The capital of France is"
    # True answer is "Paris" with high probability
    vocab = ["Paris", "London", "Rome", "Berlin", "the", "a", "because",
             "therefore", "xkzq", "aaaa", "!!!"]
    logits = torch.tensor([8.0, 2.0, 2.0, 1.5, 0.5, 0.3, -1.0,
                           -2.0, -5.0, -8.0, -10.0])

    probs = F.softmax(logits, dim=-1)

    print("Original distribution:")
    for token, prob in sorted(zip(vocab, probs.tolist()), key=lambda x: -x[1]):
        bar = "█" * int(prob * 40)
        print(f"  {token:12s} {prob:6.3f} {bar}")

    print()

    # Top-k=3: only consider the top 3 tokens
    k = 3
    top_k_logits = logits.clone()
    kth = torch.topk(top_k_logits, k).values[-1]
    top_k_logits[top_k_logits < kth] = float('-inf')
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    print(f"After top-k (k={k}):")
    for token, prob in zip(vocab, top_k_probs.tolist()):
        if prob > 0.001:
            print(f"  {token:12s} {prob:.3f}")

    print()

    # Top-p=0.9: include smallest set of tokens that covers 90% probability
    p = 0.9
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # First token where cumulative > p gets included (it just crossed the threshold)
    cutoff_idx = (cumulative > p).nonzero(as_tuple=True)[0][0].item()
    included = sorted_idx[:cutoff_idx + 1]

    top_p_logits = torch.full_like(logits, float('-inf'))
    top_p_logits[included] = logits[included]
    top_p_probs = F.softmax(top_p_logits, dim=-1)

    print(f"After top-p (p={p}):")
    for token, prob in zip(vocab, top_p_probs.tolist()):
        if prob > 0.001:
            print(f"  {token:12s} {prob:.3f}")

compare_sampling_strategies()
```

**Top-p (nucleus sampling)** is generally preferred over top-k because it adapts to the distribution. When the model is confident (distribution is peaked), top-p includes fewer tokens. When it's uncertain (distribution is flat), top-p includes more. Top-k is a fixed cutoff that doesn't adapt.

## The KV Cache: Making Inference Not Horrible

Here's a performance problem with naive inference: for each new token you generate, you re-compute attention over the entire sequence from scratch. Token 100 re-runs attention for tokens 1-99 plus itself. Token 101 re-runs attention for tokens 1-100. Token 500 re-runs attention for the previous 499 tokens.

That's O(n²) total computation to generate n tokens. For a 2,000-token response, you're doing quadratically more work than necessary.

The **KV cache** solves this by caching the Key and Value matrices from previous tokens:

```python
class KVCache:
    """
    Cache for Key and Value tensors across autoregressive steps.

    During inference:
    - Step 1: process full prompt, cache all K,V tensors
    - Step 2+: process only the new token, retrieve cached K,V for context
    """
    def __init__(self, n_layers: int, max_batch_size: int = 1):
        self.n_layers = n_layers
        self.cache_k = [None] * n_layers  # cached keys per layer
        self.cache_v = [None] * n_layers  # cached values per layer

    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        """Append new K, V to the cache for this layer."""
        if self.cache_k[layer_idx] is None:
            self.cache_k[layer_idx] = new_k
            self.cache_v[layer_idx] = new_v
        else:
            self.cache_k[layer_idx] = torch.cat([self.cache_k[layer_idx], new_k], dim=2)
            self.cache_v[layer_idx] = torch.cat([self.cache_v[layer_idx], new_v], dim=2)

    def get(self, layer_idx: int):
        return self.cache_k[layer_idx], self.cache_v[layer_idx]

    def clear(self):
        self.cache_k = [None] * self.n_layers
        self.cache_v = [None] * self.n_layers


def attention_with_cache(Q, K_new, V_new, cache: KVCache, layer_idx: int, mask=None):
    """
    Attention that uses the KV cache.

    Q: [batch, n_heads, 1, d_k]  -- only the new token's query
    K_new, V_new: [batch, n_heads, 1, d_k]  -- new key/value to append

    The cache gives us K, V for all previous tokens.
    We concatenate and run attention normally.
    """
    import math

    # Update cache with new K, V
    cache.update(layer_idx, K_new, V_new)

    # Retrieve full K, V (all tokens so far)
    K_full, V_full = cache.get(layer_idx)

    # Attention: Q against all K, weighted sum of all V
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K_full.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V_full)


def kv_cache_speedup():
    """Show the computational savings from KV caching."""
    d_model = 768
    n_heads = 12
    d_k = d_model // n_heads

    context_lengths = [100, 500, 1000, 2000]
    new_tokens = 500  # tokens to generate

    print("KV Cache speedup analysis:")
    print(f"Generating {new_tokens} tokens from contexts of various lengths")
    print()
    print(f"{'Context':>10} {'Without Cache':>16} {'With Cache':>12} {'Speedup':>10}")
    print("-" * 52)

    for ctx_len in context_lengths:
        # Without cache: each new token attends to [ctx_len + i] tokens
        ops_without = sum((ctx_len + i) for i in range(new_tokens))

        # With cache: each new token only processes 1 new token,
        # but still attends to all cached tokens (same attention cost)
        # The saving is in the Q, K, V *projection* steps (linear in seq_len)
        ops_with = new_tokens * ctx_len + sum(range(new_tokens))

        # The real saving: recomputing K,V projections
        # Without: (ctx_len + i) K,V projections per step
        proj_without = sum(ctx_len + i for i in range(new_tokens))
        proj_with = ctx_len + new_tokens  # compute once, cache forever

        speedup = proj_without / proj_with

        print(f"{ctx_len:10d} {proj_without:16,} {proj_with:12,} {speedup:9.1f}x")

kv_cache_speedup()
```

The memory cost is the trade-off: the KV cache stores `n_layers × 2 × batch_size × seq_len × n_heads × d_k` tensors of floats. For a 70B parameter model with a 128K context, this is tens of gigabytes. Long-context inference is expensive not in compute but in memory.

## Beam Search

Instead of sampling one token at a time, beam search maintains the top K complete sequences (beams) and expands them simultaneously, keeping the K highest-probability paths at each step.

```python
def beam_search(model, prompt_ids: list[int], beam_size: int = 4,
                max_new_tokens: int = 50) -> list[tuple[float, list[int]]]:
    """
    Beam search decoding.
    Returns beam_size sequences, each as (log_probability, token_ids).
    """
    # Initialize beams: (cumulative log prob, token sequence)
    beams = [(0.0, list(prompt_ids))]

    for step in range(max_new_tokens):
        candidates = []

        for log_prob, tokens in beams:
            with torch.no_grad():
                x = torch.tensor([tokens])
                logits, _ = model(x)
                next_log_probs = F.log_softmax(logits[0, -1, :], dim=-1)

            # Expand: consider all vocab items
            topk_vals, topk_ids = torch.topk(next_log_probs, beam_size)

            for val, token_id in zip(topk_vals.tolist(), topk_ids.tolist()):
                candidates.append((
                    log_prob + val,  # cumulative log prob
                    tokens + [token_id]
                ))

        # Keep top beam_size candidates
        beams = sorted(candidates, key=lambda x: -x[0])[:beam_size]

    return beams


# Beam search vs sampling trade-offs:
print("Beam search pros/cons:")
print("  + Finds higher-probability sequences")
print("  + More consistent/predictable output")
print("  + Good for tasks with a clear 'best' answer (translation, summarization)")
print()
print("  - Tends toward generic, boring outputs")
print("  - Can degenerate into repetitive sequences")
print("  - Quadratic memory in beam size × sequence length")
print()
print("In practice: sampling with top-p for creative tasks,")
print("greedy or beam search for factual/structured tasks.")
```

## Quantization: Running Big Models in Small Memory

Real production inference uses quantization: representing weights in lower precision to reduce memory and speed up computation.

```python
import torch

def quantization_demo():
    """
    Show the memory savings from quantization.
    This is the core concept behind 4-bit inference (llama.cpp, bitsandbytes, etc.)
    """
    d_model = 4096

    # Full precision: 32-bit float (4 bytes per parameter)
    weight_fp32 = torch.randn(d_model, d_model)

    # Half precision: 16-bit float (2 bytes per parameter)
    weight_fp16 = weight_fp32.half()

    # 8-bit quantization: (1 byte per parameter + overhead)
    # Conceptually: scale each weight to fit in [-128, 127]
    def quantize_int8(w: torch.Tensor):
        scale = w.abs().max() / 127.0
        quantized = (w / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale

    def dequantize_int8(q: torch.Tensor, scale: float):
        return q.float() * scale

    q_int8, scale = quantize_int8(weight_fp32)

    # Check quality loss
    reconstructed = dequantize_int8(q_int8, scale)
    max_error = (weight_fp32 - reconstructed).abs().max().item()
    relative_error = max_error / weight_fp32.abs().max().item()

    print("Quantization comparison:")
    print(f"  fp32 size:   {weight_fp32.nelement() * 4 / 1e6:.1f} MB")
    print(f"  fp16 size:   {weight_fp16.nelement() * 2 / 1e6:.1f} MB  (2x smaller)")
    print(f"  int8 size:   {q_int8.nelement() * 1 / 1e6:.1f} MB  (4x smaller)")
    print(f"  Max quantization error: {max_error:.6f}")
    print(f"  Relative error: {100*relative_error:.3f}%")
    print()

    # 70B model memory requirements
    params = 70e9
    print("70B parameter model memory:")
    print(f"  fp32: {params * 4 / 1e12:.1f} TB  (impractical)")
    print(f"  fp16: {params * 2 / 1e9:.0f} GB   (needs 4× A100 80GB GPUs)")
    print(f"  int8: {params * 1 / 1e9:.0f} GB   (fits on 2× A100 80GB GPUs)")
    print(f"  int4: {params * 0.5 / 1e9:.0f} GB   (fits on 1× A100 80GB GPU)")

quantization_demo()
```

4-bit quantization (4 bits per weight) is now standard for running large models on consumer hardware. Libraries like `llama.cpp` and `bitsandbytes` implement this efficiently.

## Batching: The Real Performance Lever

For serving many users, the single most important optimization is batching requests together. The matrix multiplications in a transformer are far more efficient with larger batch sizes — the GPU is underutilized with a single request.

```python
def batching_math():
    """
    Why batching matters for GPU utilization.
    """
    import time

    d_model = 2048

    W = torch.randn(d_model, d_model)
    n_trials = 100

    for batch_size in [1, 8, 32, 128]:
        x = torch.randn(batch_size, d_model)

        # Warm up
        for _ in range(10):
            _ = x @ W.T

        start = time.perf_counter()
        for _ in range(n_trials):
            out = x @ W.T
        elapsed = time.perf_counter() - start

        throughput = batch_size * n_trials / elapsed
        latency = elapsed / n_trials * 1000

        print(f"batch={batch_size:4d}: latency={latency:6.2f}ms, "
              f"throughput={throughput:8.0f} seq/sec")

batching_math()
```

This is why inference services use **continuous batching** — dynamically grouping incoming requests together as they arrive, rather than processing one at a time.

## A Complete Inference Pipeline

```python
def complete_inference_example(model, tokenizer, prompt: str) -> str:
    """
    A production-style inference call: tokenize, generate, decode.
    """
    # Tokenize
    prompt_tokens = tokenizer.encode(prompt)
    print(f"Prompt: {len(prompt_tokens)} tokens")

    # Generate (with all the options)
    with torch.no_grad():
        generated_tokens = generate(
            model=model,
            prompt_tokens=prompt_tokens,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    # Decode
    generated_text = tokenizer.decode(generated_tokens)
    print(f"Generated: {len(generated_tokens)} tokens")

    return generated_text
```

Inference is where everything you've built becomes a product. The model is frozen; what you control is how you query it. The choices — temperature, sampling strategy, context management, batching — determine whether your model is snappy and useful or slow and incoherent.

Temperature alone is not magic. But temperature, combined with top-p, combined with a KV cache, combined with sensible batching — that's what turns 100GB of floating-point numbers into something people want to use.
