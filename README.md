# Build Your Own AI: From Tokens to Transformers

> A CloudStreet book. Technical, accurate, and genuinely amusing.

You've been using LLMs for months. Maybe years. You've prompt-engineered your way through production systems, argued about context windows in Slack, and confidently explained to your non-technical colleagues that "it predicts the next word."

That last part is true, sort of. But if someone asked you *how* — really how, at the level of actual computation — you'd probably change the subject.

This book fixes that. By the end, you'll understand what's happening inside a transformer, why attention has three separate vectors when one seems like plenty, and what "training" actually means beyond "we showed it the internet." The code runs. The explanations are honest. The humor is dry and earns its keep.

## What's Inside

| Chapter | Topic |
|---------|-------|
| [Introduction](src/introduction.md) | Why Build Your Own AI? |
| [Tokens](src/tokens.md) | The atoms of language — what text actually looks like to a model |
| [Embeddings](src/embeddings.md) | Meaning as geometry: how words become vectors |
| [Attention](src/attention.md) | The mechanism that changed everything, explained without hand-waving |
| [Transformers](src/transformers.md) | Putting the pieces together into a full architecture |
| [Training](src/training.md) | Teaching a model to care, via gradient descent and suffering |
| [Fine-Tuning](src/finetuning.md) | Specializing without starting over |
| [Inference](src/inference.md) | Running what you built — sampling, temperature, and KV caching |
| [Limits](src/limits.md) | What your model doesn't know it doesn't know |
| [Meta](src/meta.md) | The twist: this book was written with one |
| [Conclusion](src/conclusion.md) | Where to go from here |

## Prerequisites

- Python 3.10+
- Comfortable reading Python code
- Curiosity about what's actually happening when you call the API

You don't need a GPU for most examples. You do need a willingness to think carefully.

## Building Locally

Install [mdBook](https://rust-lang.github.io/mdBook/guide/installation.html):

```bash
cargo install mdbook
```

Then build and serve:

```bash
mdbook serve --open
```

Or just build:

```bash
mdbook build
```

The output lands in `./book/`.

## Python Dependencies

For the code examples:

```bash
pip install torch numpy tiktoken transformers
```

Most chapters work with CPU-only PyTorch. A few will note where GPU helps.

## Deployed Book

The book is automatically built and deployed to GitHub Pages on every push to `main`.

Live at: [https://cloudstreet-dev.github.io/Build-Your-Own-AI/](https://cloudstreet-dev.github.io/Build-Your-Own-AI/)

---

*Built with [mdBook](https://rust-lang.github.io/mdBook/). Written by Claude Code.*
