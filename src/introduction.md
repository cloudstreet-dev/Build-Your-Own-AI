# Why Build Your Own AI?

You've used the API. You know what temperature does, roughly. You've read the phrase "transformer-based language model" enough times that it no longer triggers any reaction at all, which is its own kind of problem.

Here's the situation: you're operating a system you don't understand, and most of the time that's fine. Most of the time. Then something goes wrong — the model hallucinates a confident falsehood, ignores half the context window, or refuses to follow a simple instruction for no discernible reason — and you have no mental model for *why*. You're debugging a black box with a flashlight pointed in the wrong direction.

This book is the flashlight pointed the right direction.

## What This Book Is

A technical guide to the internals of large language models, written for developers who are already using them. Every concept is accompanied by working code. The math is explained in English first and notation second. The humor is dry and is not your problem unless you also find it funny, in which case welcome.

By the end of this book, you will understand:

- How text becomes numbers (and why those particular numbers)
- Why attention has three separate matrices when one seems like it should suffice
- What "training" actually computes, beyond the vague hand-wave about gradient descent
- How fine-tuning differs from training from scratch
- What happens at inference time, from the first token to the last
- What the architecture genuinely cannot do, and why

## What This Book Is Not

A paper survey. A comprehensive ML textbook. A guide to the latest models (they'll be obsolete before you finish reading anyway). A sales pitch.

## Who You Are

You write Python. You've called an LLM API. You can read a stack trace. You may have a rough sense that "attention" is important but aren't sure exactly what it attends to or why it couldn't just attend to everything equally and call it a day.

You're about to find out that it kind of does attend to everything, but in a very specific weighted way that turns out to be the entire secret. That's the fun of this.

## How to Use This Book

Read it in order the first time. Each chapter builds on the last. The code examples are meant to be run — they're short enough to paste into a notebook or a script, and they produce output that should make the concept click in a way prose alone won't.

If something isn't clear, that's the book's failure, not yours. The concepts are genuinely not that complicated once you strip away the intimidating notation. A transformer is a specific arrangement of matrix multiplications. That's most of it.

## A Note on Scale

The examples in this book use toy models — small architectures trained on tiny datasets, designed to fit in CPU memory and run in under a minute. Real production models have hundreds of billions of parameters and were trained on clusters of thousands of GPUs for months.

The math is identical. The engineering challenges are different. We're covering the math.

Understanding the small version perfectly is more valuable than vaguely gesturing at the large version. Once you've built a tiny transformer and watched it actually learn something, what GPT-4 is doing becomes much less mysterious. It's doing the same thing, just with more parameters, more data, and considerably more electricity.

## Let's Go

Open a Python environment. Install PyTorch if you haven't:

```bash
pip install torch numpy tiktoken
```

The first chapter is about tokens, which is where every LLM interaction actually begins. Not with your prompt. With the question of how to cut your prompt into pieces the model can process.

You've been starting from the wrong end.
