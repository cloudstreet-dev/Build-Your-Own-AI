# Training Loops: Teaching a Model to Care

The model you have at initialization is random noise with a shape. The weights are sampled from a small normal distribution. Run a forward pass and you'll get approximately uniform predictions over the vocabulary — the model genuinely has no preferences. It doesn't know what a word is. It doesn't know what anything is.

Training is the process of making it care. Specifically: giving it examples of text, measuring how wrong its predictions are, and adjusting its weights to make them slightly less wrong. Repeat 100 billion times. That's the entire procedure.

The remarkable thing is that this actually works.

## The Objective: Next-Token Prediction

A language model has one job: predict the next token. Given a sequence of tokens, output a probability distribution over the vocabulary for what comes next.

This is called the **causal language modeling objective** (or self-supervised learning, because the labels come from the data itself — no human annotation required).

For a sequence `[t₁, t₂, t₃, t₄, t₅]`, the model is asked:
- Given `[t₁]`, predict `t₂`
- Given `[t₁, t₂]`, predict `t₃`
- Given `[t₁, t₂, t₃]`, predict `t₄`
- Given `[t₁, t₂, t₃, t₄]`, predict `t₅`

Because of the causal mask (previous chapter), the transformer can compute all these predictions in a single forward pass. The output at position `i` predicts position `i+1`. Efficient.

## Cross-Entropy Loss

The model outputs logits — unnormalized scores for each vocabulary item. We convert these to probabilities with softmax, then measure how wrong we are with **cross-entropy loss**:

```
loss = -log(probability of the correct token)
```

If the model assigns 90% probability to the correct token: `loss = -log(0.9) = 0.105` (low, good).
If the model assigns 1% probability to the correct token: `loss = -log(0.01) = 4.6` (high, bad).
If the vocabulary has 50,000 tokens and the model is totally random: `loss = -log(1/50000) = 10.8`.

A freshly initialized model gets around 10.8. A well-trained model gets around 2-3 on held-out text (lower on training text, which it's memorized to some degree).

```python
import torch
import torch.nn.functional as F
import math

def compute_loss_example():
    """Illustrate what cross-entropy loss is measuring."""
    vocab_size = 10
    batch_size = 2
    seq_len = 5

    # Random logits (fresh model)
    logits_random = torch.randn(batch_size, seq_len, vocab_size)
    # Good model: high confidence on correct tokens
    logits_good = torch.zeros(batch_size, seq_len, vocab_size)

    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Set logits_good to have high values at correct positions
    for b in range(batch_size):
        for s in range(seq_len):
            logits_good[b, s, targets[b, s]] = 5.0  # strong signal for correct token

    # Compute losses
    loss_random = F.cross_entropy(
        logits_random.view(-1, vocab_size),
        targets.view(-1)
    )
    loss_good = F.cross_entropy(
        logits_good.view(-1, vocab_size),
        targets.view(-1)
    )

    print(f"Random model loss:    {loss_random.item():.3f}")
    print(f"Expected (log V):     {math.log(vocab_size):.3f}")
    print(f"Well-trained loss:    {loss_good.item():.3f}")
    print()

    # Perplexity = exp(loss), easier to interpret
    # "The model is as confused as if choosing randomly from N options"
    print(f"Random model perplexity:  {math.exp(loss_random.item()):.1f}  (≈ {vocab_size})")
    print(f"Good model perplexity:    {math.exp(loss_good.item()):.2f}")

compute_loss_example()
```

**Perplexity** is often reported instead of loss: `perplexity = exp(loss)`. It has a nice interpretation: a perplexity of K means the model is "as surprised as if it had to choose uniformly from K options." A perplexity of 20 on English text means the model effectively has 20 plausible next tokens at each step, even though the vocabulary is 50,000.

## Gradient Descent

Given the loss, how do we update the weights to reduce it?

**Gradient descent**: compute the gradient of the loss with respect to every parameter, then step in the opposite direction.

The gradient tells you: "if you increase this weight by a tiny amount, the loss increases/decreases by this much." Moving in the opposite direction of the gradient reduces the loss.

```python
def manual_gradient_descent():
    """Gradient descent on a toy problem, made explicit."""
    # A single parameter model: y = w * x, predict y given x
    w = torch.tensor(2.0, requires_grad=True)  # starts at 2, true value is 5

    # Fake dataset: y = 5x
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([5.0, 10.0, 15.0, 20.0])

    learning_rate = 0.1

    print("Gradient descent on y = w*x (true w=5):")
    print(f"  Starting w = {w.item():.4f}")
    print()

    for step in range(10):
        # Forward pass
        y_pred = w * x
        loss = ((y_pred - y_true) ** 2).mean()  # MSE loss

        # Backward pass: compute d(loss)/d(w)
        loss.backward()

        # Update: w = w - lr * gradient
        with torch.no_grad():
            w -= learning_rate * w.grad

        # Zero the gradient (IMPORTANT: gradients accumulate by default)
        w.grad.zero_()

        print(f"  Step {step+1:2d}: w={w.item():.4f}, loss={loss.item():.4f}")

manual_gradient_descent()
```

The model has millions of parameters, but the principle is the same. PyTorch's `loss.backward()` computes all the gradients automatically via **backpropagation** — the chain rule applied recursively to the computation graph. You write the forward pass; PyTorch figures out the backward pass.

## A Complete Training Loop

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model
    vocab_size: int = 256        # character-level for simplicity
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 2000
    eval_interval: int = 200
    eval_steps: int = 50
    grad_clip: float = 1.0

    # Optimization
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)  # AdamW betas
    warmup_steps: int = 100


def get_lr(step: int, config: TrainingConfig) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    This is the standard schedule for transformer training.
    """
    if step < config.warmup_steps:
        # Linear warmup
        return config.learning_rate * step / config.warmup_steps

    # Cosine decay
    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


class Trainer:
    def __init__(self, model: nn.Module, config: TrainingConfig, train_data, val_data):
        self.model = model
        self.config = config
        self.train_data = train_data
        self.val_data = val_data

        # Separate parameters into those that should and shouldn't have weight decay
        # (biases and layer norm parameters typically excluded from weight decay)
        decay_params = [p for n, p in model.named_parameters()
                       if p.dim() >= 2]  # weight matrices
        no_decay_params = [p for n, p in model.named_parameters()
                          if p.dim() < 2]  # biases, layer norms

        self.optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=config.learning_rate, betas=config.betas)

        self.step = 0
        self.train_losses = []
        self.val_losses = []

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == 'train' else self.val_data
        c = self.config
        ix = torch.randint(len(data) - c.max_seq_len, (c.batch_size,))
        x = torch.stack([data[i:i+c.max_seq_len] for i in ix])
        y = torch.stack([data[i+1:i+c.max_seq_len+1] for i in ix])
        return x, y

    @torch.no_grad()
    def evaluate(self) -> float:
        """Estimate validation loss over several batches."""
        self.model.eval()
        losses = []
        for _ in range(self.config.eval_steps):
            x, y = self.get_batch('val')
            _, loss = self.model(x, targets=y)
            losses.append(loss.item())
        self.model.train()
        return np.mean(losses)

    def train_step(self) -> float:
        """One step of training."""
        x, y = self.get_batch('train')

        # Update learning rate
        lr = get_lr(self.step, self.config)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Forward + backward + update
        _, loss = self.model(x, targets=y)

        self.optimizer.zero_grad(set_to_none=True)  # slightly faster than zero_grad()
        loss.backward()

        # Gradient clipping: prevents exploding gradients
        # Clips the global norm of all gradients to grad_clip
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip
        )

        self.optimizer.step()
        self.step += 1

        return loss.item()

    def train(self):
        self.model.train()
        print(f"Training for {self.config.max_steps} steps...")
        print(f"{'Step':>6} {'Train Loss':>12} {'Val Loss':>10} {'LR':>12}")
        print("-" * 45)

        running_loss = 0
        for step in range(self.config.max_steps):
            loss = self.train_step()
            running_loss += loss

            if (step + 1) % self.config.eval_interval == 0:
                avg_train_loss = running_loss / self.config.eval_interval
                val_loss = self.evaluate()
                lr = get_lr(step, self.config)

                self.train_losses.append(avg_train_loss)
                self.val_losses.append(val_loss)

                print(f"{step+1:6d} {avg_train_loss:12.4f} {val_loss:10.4f} {lr:12.2e}")
                running_loss = 0
```

## Running a Real Training Run

Let's put it all together and train on something:

```python
import math

# Re-use our Shakespeare text from the Transformers chapter
text = open('/dev/stdin').read() if False else """
To be or not to be that is the question
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune
Or to take arms against a sea of troubles
And by opposing end them to die to sleep
No more and by a sleep to say we end
""" * 50

# Encode as bytes (gives us a clean 256-token vocab)
data = torch.tensor(list(text.encode('utf-8')), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

config = TrainingConfig(
    vocab_size=256,
    d_model=64,
    n_heads=4,
    n_layers=3,
    max_seq_len=64,
    batch_size=16,
    max_steps=1000,
    eval_interval=200,
)

# Build the model (same GPTLanguageModel from previous chapter)
model = GPTLanguageModel(
    vocab_size=config.vocab_size,
    d_model=config.d_model,
    n_heads=config.n_heads,
    n_layers=config.n_layers,
    max_seq_len=config.max_seq_len,
    dropout=config.dropout,
)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} parameters")
print(f"Training data: {len(train_data):,} tokens")
print(f"Val data: {len(val_data):,} tokens")
print()

trainer = Trainer(model, config, train_data, val_data)
trainer.train()

# Generate some text
print("\nSample generation (temperature=0.8):")
prompt = b"To be"
idx = torch.tensor([list(prompt)], dtype=torch.long)
output = model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=20)
print(bytes(output[0].tolist()).decode('utf-8', errors='replace'))
```

## What Each Component of the Optimizer Does

```python
def explain_optimizer():
    """Walk through what Adam does to each parameter."""

    # Adam (and AdamW) maintain per-parameter state:
    # - m: first moment (momentum) — exponential average of gradients
    # - v: second moment — exponential average of squared gradients

    # For a single parameter update:
    # m = beta1 * m + (1 - beta1) * grad          # gradient moving average
    # v = beta2 * v + (1 - beta2) * grad^2        # squared gradient moving average
    # m_hat = m / (1 - beta1^t)                   # bias correction
    # v_hat = v / (1 - beta2^t)
    # param = param - lr * m_hat / (sqrt(v_hat) + eps)

    # The key insight: Adam normalizes the step size per parameter.
    # A parameter that has been getting large gradients gets smaller steps.
    # A parameter that has been getting small gradients gets larger steps.
    # This adaptive learning rate is why Adam works so much better than plain SGD for transformers.

    # AdamW adds weight decay directly to the parameter (not through gradient):
    # param = param * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)
    # This is the "correct" way to do weight decay with adaptive optimizers.

    beta1, beta2 = 0.9, 0.95
    eps = 1e-8

    # Simulate 10 steps for a single parameter
    param = torch.tensor(1.0)
    m = torch.tensor(0.0)
    v = torch.tensor(0.0)
    lr = 0.001

    print("Adam update trace (single parameter):")
    print(f"{'Step':>5} {'Grad':>8} {'m':>8} {'v':>8} {'update':>10} {'param':>8}")

    for t in range(1, 11):
        grad = torch.randn(1).item()  # simulate a gradient

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        update = lr * m_hat / (v_hat ** 0.5 + eps)
        param = param - update

        print(f"{t:5d} {grad:8.4f} {m:8.4f} {v:8.4f} {-update:10.6f} {param:8.4f}")

explain_optimizer()
```

## Diagnosing Training: Loss Curves

```python
def plot_training_curves(train_losses, val_losses):
    """Print an ASCII loss curve for quick sanity checking."""
    if not train_losses:
        print("No data yet.")
        return

    # Find range
    all_losses = train_losses + val_losses
    min_l, max_l = min(all_losses), max(all_losses)
    height = 15
    width = min(60, len(train_losses))

    print("\nTraining curves (▓=train, ░=val):")

    for row in range(height, -1, -1):
        threshold = min_l + (row / height) * (max_l - min_l)
        line = f"{threshold:6.2f} |"
        for i in range(min(width, len(train_losses))):
            t_idx = int(i * len(train_losses) / width)
            train_char = "▓" if train_losses[t_idx] >= threshold else " "
            val_char = "░" if val_losses[t_idx] >= threshold else " "
            line += train_char if train_losses[t_idx] >= threshold else (val_char if val_losses[t_idx] >= threshold else " ")
        print(line)

    print("       " + "+" + "-" * width)
    print(f"  loss ↑         steps →")
    print(f"\n  Healthy: train ≈ val, both decreasing")
    print(f"  Overfitting: train decreasing, val increasing (need more data or regularization)")
    print(f"  Underfitting: both high and flat (need more capacity or longer training)")

# Example with simulated curves
import math
steps = 20
fake_train = [3.0 * math.exp(-0.15 * i) + 0.5 + 0.1 * (hash(i) % 100) / 100 for i in range(steps)]
fake_val = [3.2 * math.exp(-0.13 * i) + 0.7 + 0.05 * (hash(i+100) % 100) / 100 for i in range(steps)]
plot_training_curves(fake_train, fake_val)
```

## The Dirty Secrets of Training

A few things the clean presentation above glosses over:

**Gradient clipping is not optional.** Without it, a single bad batch can send your gradients to infinity and destroy your model. The clip value of 1.0 is almost universal.

**Learning rate matters more than almost anything else.** Too high and the model diverges. Too low and it barely learns. Warmup (gradually increasing from 0) prevents instability at the start when gradients are wild.

**The 4x FFN ratio is empirical.** Nobody proved that `d_ff = 4 * d_model` is optimal. It's just what worked and everyone kept using it.

**Batch size interacts with learning rate.** Doubling the batch size approximately doubles the effective learning rate. If you change batch size, adjust learning rate accordingly (linear scaling rule).

**Most runs fail.** Real LLM training involves constant monitoring, occasional instabilities, and sometimes full restarts. Training a model from scratch requires checkpointing every few hours and being prepared for the cluster to go down.

The fundamentals, however, are exactly what you've seen: loss, gradient, update. Repeated until the model stops embarrassing itself.

That's training.
