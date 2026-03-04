# Fine-Tuning: Specializing Without Starting Over

Training a language model from scratch on the scale of GPT-4 costs somewhere in the range of $50-100 million and requires a cluster of thousands of GPUs running for months. Unless your startup has unusually aggressive compute budgets, you're not doing that.

What you're doing is **fine-tuning**: taking a pre-trained model and adapting it for a specific task by training on a much smaller dataset for a much shorter time. This works remarkably well, and understanding why it works helps you make better decisions about when and how to do it.

## Why Fine-Tuning Works

Pre-training on billions of tokens of text teaches a model a tremendous amount: grammar, facts, reasoning patterns, code syntax, sentiment, style, logic. The model develops rich internal representations of language.

Fine-tuning doesn't overwrite this knowledge — it builds on it. You're not teaching the model what language is; you're teaching it which aspects of its knowledge to emphasize for your particular task, and what tone/format to use.

Think of it as the difference between hiring a generalist who learns your codebase versus hiring someone fresh out of school. The generalist already has all the general skills; they just need domain-specific adaptation.

## Types of Fine-Tuning

There are several approaches, with dramatically different compute requirements:

| Method | What's Updated | Parameters Changed | Typical Use |
|--------|---------------|-------------------|-------------|
| Full fine-tuning | All weights | 100% | Significant behavior change |
| Instruction tuning | All weights | 100% | Chat/instruction following |
| LoRA | Low-rank adapters | ~0.1-1% | Efficient adaptation |
| QLoRA | LoRA on quantized model | ~0.1-1% | Very low VRAM |
| Prompt tuning | Soft prompt tokens | <0.01% | Minimal adaptation |

## Full Fine-Tuning

The simple version: take a pre-trained model, load its weights, and continue training with your dataset using a lower learning rate.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def full_finetune_example():
    """
    Demonstrates the mechanics of full fine-tuning.
    (Use a tiny model for illustration — in practice, use 7B+ parameter models.)
    """
    # Load a tiny pre-trained model
    model_name = "gpt2"  # 117M parameters, fits on CPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Add padding token (GPT-2 doesn't have one)
    tokenizer.pad_token = tokenizer.eos_token

    # Your fine-tuning dataset — format matters
    # For instruction following: prompt + completion pairs
    training_examples = [
        {
            "prompt": "Summarize the following in one sentence:",
            "text": " Transformers use self-attention to process sequences in parallel, enabling much more efficient training than recurrent networks.",
        },
        {
            "prompt": "What is gradient descent?",
            "text": " Gradient descent is an optimization algorithm that iteratively adjusts model parameters in the direction that reduces the loss function.",
        },
        {
            "prompt": "Explain tokenization briefly:",
            "text": " Tokenization converts raw text into a sequence of integer IDs by splitting text into subword units according to a learned vocabulary.",
        },
    ]

    # Format as: "<prompt><completion>" — model learns to generate the completion
    def format_example(ex):
        return ex["prompt"] + ex["text"] + tokenizer.eos_token

    # Fine-tuning uses a lower learning rate than pre-training
    # Pre-training: ~3e-4; fine-tuning: 1e-5 to 5e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    model.train()
    print("Fine-tuning steps:")

    for step, example in enumerate(training_examples * 3):  # 3 epochs
        text = format_example(example)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
        )

        # For language modeling: targets are inputs shifted by 1
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 3 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")

    print("\nFine-tuning complete.")
    return model, tokenizer

# model, tokenizer = full_finetune_example()
```

The key differences from pre-training:
1. **Lower learning rate** (10-100x lower) — preserves pre-trained knowledge
2. **Smaller dataset** (thousands to millions of examples, not billions)
3. **Fewer steps** (hours to days, not months)
4. **Task-specific data format** — the model learns the format during fine-tuning

## LoRA: Low-Rank Adaptation

Full fine-tuning updates every parameter. For a 7B parameter model, that's 7 billion gradient updates per step, 7 billion gradient values to store. If you're using Adam, that's another 14 billion values for the optimizer state. This gets expensive.

**LoRA** (Low-Rank Adaptation) makes a key observation: the change in weights during fine-tuning has a low intrinsic rank. Rather than learning a full `[d, d]` weight update `ΔW`, we decompose it into two small matrices: `ΔW = B × A` where `A` is `[r, d]` and `B` is `[d, r]`, with `r << d`.

If `d = 4096` and `r = 16`, the original weight matrix has 16M parameters. The LoRA decomposition has just `2 * 16 * 4096 = 131K` parameters — 120x smaller. You freeze the original weights and only train the small adapter matrices.

```python
import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    The forward pass computes: y = x @ W.T + x @ A.T @ B.T * scale
    where W is frozen and A, B are the trainable LoRA parameters.
    """
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,  # scaling factor; lora_alpha / rank = effective scale
        dropout: float = 0.0,
    ):
        super().__init__()

        self.original = original_linear
        # Freeze the original weights
        for param in self.original.parameters():
            param.requires_grad = False

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        self.rank = rank
        self.scale = alpha / rank

        # LoRA matrices
        # A is initialized to random Gaussian, B to zero.
        # This means ΔW = B @ A = 0 at initialization — no change to model behavior.
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features) / math.sqrt(rank)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original linear transformation (frozen)
        base = self.original(x)

        # LoRA adaptation
        # x: [batch, ..., in_features]
        lora = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        # lora: [batch, ..., out_features]

        return base + lora * self.scale

    @property
    def weight(self):
        """Returns the effective weight (original + LoRA adaptation)."""
        return self.original.weight + (self.lora_B @ self.lora_A) * self.scale


def apply_lora(model: nn.Module, rank: int = 16, alpha: float = 32.0,
               target_modules: list = None) -> nn.Module:
    """
    Replace target Linear layers with LoRA versions.

    In practice, LoRA is applied to the attention Q and V projections
    (sometimes K and the output projection too).
    """
    if target_modules is None:
        target_modules = ["W_q", "W_v"]  # standard: query and value projections

    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in target_modules:
            setattr(model, name, LoRALinear(module, rank=rank, alpha=alpha))
        else:
            apply_lora(module, rank, alpha, target_modules)  # recurse

    return model


def count_trainable(model: nn.Module) -> tuple[int, int]:
    """Count trainable vs total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# Demonstrate LoRA on a small linear layer
original = nn.Linear(4096, 4096)
lora_layer = LoRALinear(original, rank=16, alpha=32)

orig_params = sum(p.numel() for p in original.parameters())
lora_trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)

print(f"Original linear layer: {orig_params:,} parameters")
print(f"LoRA trainable parameters: {lora_trainable:,} parameters")
print(f"Reduction: {orig_params / lora_trainable:.0f}x fewer trainable parameters")
print()

# Test forward pass
x = torch.randn(2, 10, 4096)
out = lora_layer(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")

# Verify that at initialization, LoRA adds nothing (B=0, so B@A=0)
with torch.no_grad():
    base_out = original(x)
    diff = (out - base_out).abs().max()
    print(f"Max difference from original at init: {diff.item():.2e}  (should be ~0)")
```

## The LoRA Training Loop

```python
def lora_training_example():
    """
    Shows how to set up LoRA training:
    freeze everything, then apply LoRA adapters to attention layers.
    """
    from transformers import AutoModelForCausalLM

    # Load a small base model
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # Step 1: Freeze ALL parameters
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Add LoRA to the attention Q and V projections
    # In GPT-2, attention weights are in model.transformer.h[i].attn.c_attn
    # (which is a fused QKV projection — we'll add LoRA to each layer's attention)
    rank = 8
    alpha = 16.0
    lora_params = []

    for layer in model.transformer.h:
        # GPT-2's c_attn is a single linear layer doing Q, K, V jointly
        c_attn = layer.attn.c_attn
        lora_attn = LoRALinear(c_attn, rank=rank, alpha=alpha)
        layer.attn.c_attn = lora_attn
        lora_params.extend([lora_attn.lora_A, lora_attn.lora_B])

    # Step 3: Verify parameter counts
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable fraction:   {100 * trainable / total:.2f}%")

    # Step 4: Only pass trainable parameters to optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4  # can use a higher LR with LoRA since only adapters are updated
    )

    return model, optimizer

# model, optimizer = lora_training_example()
```

## Instruction Tuning: Teaching the Model to Follow Instructions

Raw pre-training teaches a model to continue text. But users want a model that answers questions, follows instructions, and has a conversational format. The transition from "text completer" to "assistant" is called **instruction tuning**.

The format typically looks like this:

```python
def format_instruction_example(instruction: str, response: str) -> str:
    """
    ChatML format — used by many open-source models.
    The model learns to generate text after 'assistant'.
    """
    return f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>"""

# During training, we only compute loss on the assistant's response
# (not on the prompt — the model shouldn't be penalized for predicting the prompt)
def mask_prompt_tokens(input_ids: torch.Tensor, tokenizer, assistant_token: str):
    """
    Returns labels where prompt tokens are -100 (ignored by cross-entropy).
    The model only learns to generate the response.
    """
    labels = input_ids.clone()

    # Find where the assistant's response starts
    assistant_ids = tokenizer.encode(assistant_token, add_special_tokens=False)

    # Mask everything before the assistant's turn
    for i, id in enumerate(input_ids[0]):
        if input_ids[0, i:i+len(assistant_ids)].tolist() == assistant_ids:
            labels[0, :i] = -100  # -100 is ignored by F.cross_entropy
            break

    return labels

# Example
example = format_instruction_example(
    "What is the capital of France?",
    "The capital of France is Paris."
)
print("Formatted instruction example:")
print(example)
```

## RLHF: The Part That Actually Makes It Useful

Here's something important that's often glossed over: instruction tuning on human-written examples isn't quite enough to get the behavior you want. Models fine-tuned this way are better at following instructions but still produce outputs that are confidently wrong, subtly harmful, or not what the user actually wanted.

**RLHF** (Reinforcement Learning from Human Feedback) fixes this by directly optimizing for human preferences:

1. **Collect comparisons**: Show humans two model outputs and ask which is better
2. **Train a reward model**: A neural network that predicts human preference scores
3. **RL optimization**: Use PPO (or similar) to optimize the LLM against the reward model
4. **KL constraint**: Penalize the model for drifting too far from the original (prevents reward hacking)

The KL constraint deserves emphasis: without it, the model learns to game the reward model rather than actually improving. This is the classic "reward hacking" problem in RL.

```python
def rlhf_kl_penalty(log_probs_current, log_probs_reference, kl_coeff=0.1):
    """
    The KL penalty that keeps the RLHF-trained model close to the reference model.

    KL(current || reference) = E[log P_current - log P_reference]

    This is added as a negative reward (penalty) to prevent the model from
    finding degenerate solutions that score high on the reward model but
    produce gibberish or unsafe outputs.
    """
    kl = log_probs_current - log_probs_reference  # elementwise KL contribution
    return kl_coeff * kl.mean()

# The full RLHF reward for a generated sequence:
def rlhf_reward(reward_model_score, kl_penalty, token_kl_penalties):
    """
    total_reward = reward_model_score - sum(kl_penalties)
    """
    return reward_model_score - token_kl_penalties.sum()
```

Modern alternatives to RLHF include **DPO** (Direct Preference Optimization), which achieves similar results without the complexity of reinforcement learning — it directly fine-tunes on preference pairs.

## When to Use What

**Full fine-tuning** when:
- You have significant compute available
- You're adapting for a completely different task domain
- You need to change the model's fundamental behavior

**LoRA** when:
- Limited GPU memory (fine-tune a 7B model on a single consumer GPU)
- Multiple adapters needed (swap between different fine-tuned versions easily)
- You want to share the adapter without sharing the full model weights

**Instruction tuning** when:
- Base model doesn't follow instructions well
- You need a specific conversational format
- You're building an assistant product

**LoRA + instruction tuning** when:
- All of the above, which is most of the time

## What Fine-Tuning Cannot Do

Fine-tuning cannot add new knowledge that wasn't in the pre-training data. It can bring out knowledge that's already there, format it differently, and adjust behavior. It cannot teach a model to know facts it has never seen.

If you fine-tune on a dataset describing your proprietary product, the model will learn to discuss it in the right format. If you fine-tune on customer service examples, it'll follow those conversational patterns. But it won't develop capabilities it doesn't have from pre-training — it won't learn to reason better, develop new skills, or understand genuinely new concepts.

This is why retrieval-augmented generation (RAG) exists as a complement to fine-tuning. Fine-tuning for behavior; RAG for knowledge.
