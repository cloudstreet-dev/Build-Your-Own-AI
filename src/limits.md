# What Your Model Doesn't Know It Doesn't Know

We've spent the last several chapters explaining what language models can do. This chapter is about what they can't, and — more importantly — why they can't do it while being so completely, fluently, confidently wrong about it.

This is not a complaints department. Understanding the failure modes is as essential as understanding the architecture. If you're deploying models in production and don't know why they fail the way they fail, you're going to get surprised at the worst possible times.

## The Fundamental Problem: Prediction is Not Comprehension

A language model was trained to predict the next token. That's it. The model doesn't understand text in the way you understand it. It has learned an extraordinarily rich statistical model of text — rich enough that it can discuss philosophy, debug code, and write poetry — but it has no ground truth, no world model, no way to verify what it's saying.

When the model says "Paris is the capital of France," it's not recalling a fact it knows to be true. It's predicting tokens that follow "Paris is the capital of" based on the distribution learned from training data. The output happens to be correct because "France" reliably follows "Paris is the capital of" in training data.

When the model says "The capital of Burkina Faso is Bobo-Dioulasso," it's doing the same thing — just getting it wrong. (The capital is Ouagadougou.) The model's confidence is identical in both cases.

This is the key insight: **the model cannot distinguish what it knows from what it's pattern-matching to incorrectly.** There is no confidence signal from the model that is reliably calibrated to actual accuracy.

## Hallucination: A Technical Explanation

"Hallucination" is the field's polite term for confident fabrication. Let's be precise about what's happening.

```python
import torch
import torch.nn.functional as F

def illustrate_hallucination():
    """
    Why a model generates plausible-sounding false information.
    """
    # Consider two scenarios for predicting "The CEO of Obscure Corp is ___"
    # Scenario A: the model has seen this fact many times in training
    # Scenario B: the model has seen very similar patterns but not this specific fact

    # In both cases, the model generates a high-probability continuation.
    # It has no flag for "I haven't seen this exact fact."

    print("Why hallucination looks like accurate recall:")
    print()

    # Simulated probability distributions for "The CEO of [Company] is [Name]"
    well_known = {
        "Tim Cook": 0.72,
        "Elon Musk": 0.05,
        "Jensen Huang": 0.08,
        "[other names]": 0.15,
    }

    obscure = {
        "Sarah Johnson": 0.18,  # plausible CEO name
        "Michael Chen": 0.15,   # plausible CEO name
        "John Smith": 0.20,     # common name
        "[other names]": 0.47,  # uncertainty spread around
    }

    print("'The CEO of Apple is ___':")
    for name, prob in well_known.items():
        bar = "█" * int(prob * 30)
        print(f"  {name:20s} {prob:.2f} {bar}")

    print()
    print("'The CEO of [obscure company] is ___' (not in training data):")
    print("(Model still produces a confident-looking distribution!)")
    for name, prob in obscure.items():
        bar = "█" * int(prob * 30)
        print(f"  {name:20s} {prob:.2f} {bar}")

    print()
    print("The model picks 'John Smith' (highest probability).")
    print("It says this with the same tone and confidence as the Apple answer.")
    print("There's no marker in the output indicating it's guessing.")

illustrate_hallucination()
```

Hallucination isn't a bug in the usual sense. The model is doing exactly what it was trained to do: predict plausible continuations. The problem is that "plausible" and "true" are correlated but not identical, and the training objective doesn't distinguish between them.

## What the Training Data Actually Contains

The internet — and by extension, the training data — contains a lot of wrong information. Stated confidently. In fluent prose. With links to other sources that are also wrong.

The model learned from all of it.

```python
def training_data_problems():
    """Categories of problematic training data."""

    problems = {
        "Outdated information": {
            "example": "Recommends deprecated API that was removed in 2022",
            "detection": "Hard — requires knowing current state of the world",
            "mitigation": "Knowledge cutoff awareness, RAG with recent sources",
        },
        "Confident misinformation": {
            "example": "States a false historical claim with complete certainty",
            "detection": "Very hard — model has no internal fact-checker",
            "mitigation": "Retrieval augmentation, human verification for critical facts",
        },
        "Biased representation": {
            "example": "Overrepresents English-language Western perspectives",
            "detection": "Systematic testing across demographics",
            "mitigation": "Diverse training data, targeted fine-tuning, RLHF",
        },
        "Fictional presented as factual": {
            "example": "Cites a character from a novel as a real person",
            "detection": "Hard for niche topics",
            "mitigation": "Grounding to verified sources, citations",
        },
        "Code that doesn't work": {
            "example": "Generates Python 2 syntax for a Python 3 question",
            "detection": "Run the code",
            "mitigation": "Run the code",
        },
    }

    for category, details in problems.items():
        print(f"\n{category}")
        print(f"  Example:    {details['example']}")
        print(f"  Detection:  {details['detection']}")
        print(f"  Mitigation: {details['mitigation']}")

training_data_problems()
```

## The Context Window: Hard Limits on Memory

Language models have no persistent memory between calls. Within a call, they have access to everything in the context window — but only that.

```python
def context_window_mechanics():
    """
    What the context window actually is and what happens at the boundary.
    """
    print("Context window behavior:")
    print()

    context_sizes = {
        "GPT-3.5": 16_384,
        "GPT-4": 128_000,
        "Claude 3.5 Sonnet": 200_000,
        "Our toy model": 128,
    }

    for model, tokens in context_sizes.items():
        # Rough character count
        chars = tokens * 4
        pages = chars / 2000  # ~2000 chars per page
        print(f"  {model:25s}: {tokens:7,} tokens ≈ {pages:.0f} pages")

    print()
    print("What happens at the boundary:")
    print("  - Oldest tokens are truncated (sliding window)")
    print("  - Model loses access to that context permanently")
    print("  - Cannot 'remember' earlier conversation once truncated")
    print()
    print("What 'long context' doesn't mean:")
    print("  - Doesn't mean the model attends equally well to all positions")
    print("  - 'Lost in the middle': models attend better to start/end of context")
    print("  - Quadratic attention cost means long contexts are expensive")
    print()

    # Demonstrate the lost-in-the-middle problem conceptually
    positions = ["beginning", "early middle", "middle", "late middle", "end"]
    attention_quality = [0.92, 0.75, 0.45, 0.68, 0.89]  # approximate findings from research

    print("Approximate recall by position in context (empirical):")
    for pos, qual in zip(positions, attention_quality):
        bar = "█" * int(qual * 20)
        print(f"  {pos:15s}: {bar} {qual:.0%}")

context_window_mechanics()
```

The "lost in the middle" problem is real and measurable: models reliably recall information at the beginning and end of long contexts better than information in the middle. If you're putting critical context at the middle of a 200K token window, the model may effectively ignore it.

## What Transformers Are Bad At

```python
def genuine_limitations():
    """
    Things transformers are structurally bad at, not just empirically weak at.
    """

    limitations = [
        {
            "limitation": "Exact counting",
            "why": "Counting requires maintaining precise state across arbitrary lengths. "
                   "Transformers process everything in parallel — there's no "
                   "'counter' that increments with each step.",
            "example": "Count the letter 'e' in a long string",
            "workaround": "Use code execution for exact counting tasks",
        },
        {
            "limitation": "Precise arithmetic",
            "why": "Large numbers are tokenized as individual digits. "
                   "Multi-digit multiplication requires carrying, which requires "
                   "sequential state the architecture doesn't naturally maintain.",
            "example": "47382 × 91847 = ?",
            "workaround": "Use a calculator tool / code execution",
        },
        {
            "limitation": "Logical consistency at scale",
            "why": "The model can be locally consistent but globally contradictory. "
                   "Each attention head sees local context; global consistency "
                   "isn't explicitly optimized for.",
            "example": "State that A>B, B>C, then later claim A<C",
            "workaround": "Chain-of-thought prompting, structured output, verification steps",
        },
        {
            "limitation": "Long-range planning",
            "why": "Generation is left-to-right, one token at a time. "
                   "The model can't 'look ahead' to ensure the current token "
                   "leads to a good completion 50 steps later.",
            "example": "Writing a story where all plot threads resolve satisfyingly",
            "workaround": "Outline first, then expand; multi-step generation with review",
        },
        {
            "limitation": "Knowing what it doesn't know",
            "why": "There's no 'uncertainty representation' in the output logits "
                   "that reliably distinguishes confident-and-correct from "
                   "confident-and-wrong. Calibration is imperfect and domain-dependent.",
            "example": "Ask about an obscure topic; model will sound equally sure",
            "workaround": "Retrieval augmentation, explicit uncertainty prompting, citations",
        },
        {
            "limitation": "World state tracking",
            "why": "The model sees text, not the world. It cannot perceive "
                   "what's actually true right now — only what was in training data.",
            "example": "Current stock prices, today's weather, live sports scores",
            "workaround": "Tool use / API calls for real-time data",
        },
    ]

    for item in limitations:
        print(f"\n{'='*60}")
        print(f"LIMITATION: {item['limitation']}")
        print(f"\n  Why it fails:")
        # Word-wrap manually
        words = item['why'].split()
        line = "    "
        for word in words:
            if len(line) + len(word) > 70:
                print(line)
                line = "    " + word + " "
            else:
                line += word + " "
        print(line)
        print(f"\n  Canonical example: {item['example']}")
        print(f"  Practical workaround: {item['workaround']}")

genuine_limitations()
```

## The Calibration Problem

**Calibration** measures whether a model's expressed confidence matches its actual accuracy. A well-calibrated model that claims 90% confidence should be right about 90% of the time.

LLMs are poorly calibrated, in a specific direction: they're overconfident. The fluent, assertive tone of model outputs is a property of the training data (humans write confidently), not a reflection of actual certainty.

```python
def calibration_example():
    """
    Illustrate calibration — and LLMs' tendency to be overconfident.
    """
    import numpy as np

    # Perfect calibration: when model says it's 90% confident, it's right 90% of the time
    confidences = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    print("Calibration chart: expressed confidence vs. actual accuracy")
    print()
    print(f"{'Confidence':>12} {'Perfect':>10} {'Typical LLM':>14}")
    print("-" * 40)

    np.random.seed(42)
    for conf in confidences:
        perfect = conf  # perfectly calibrated
        # LLMs tend to be overconfident — actual accuracy is lower than stated
        # (This is illustrative; actual numbers vary by model and domain)
        llm_approx = conf * 0.85 + np.random.normal(0, 0.02)
        llm_approx = max(0.4, min(1.0, llm_approx))

        bar_perfect = "█" * int(perfect * 15)
        bar_llm = "░" * int(llm_approx * 15)
        print(f"{conf:12.0%} {perfect:10.0%} ({bar_perfect:15s}) "
              f"{llm_approx:8.0%} ({bar_llm})")

    print()
    print("Implication: don't use a model's tone as a confidence signal.")
    print("The model sounds equally sure about things it knows and things it's guessing.")

calibration_example()
```

## Practical Defenses

If you're building with LLMs, these are the structural interventions that actually help:

```python
def practical_defenses():
    """Production patterns for limiting LLM failure modes."""

    defenses = {
        "Retrieval-Augmented Generation (RAG)": {
            "problem_solved": "Hallucination of facts, outdated information",
            "mechanism": "Retrieve relevant documents, include in context, ask model to cite them",
            "limitation": "Retrieval can fail; model can still hallucinate despite context",
            "when_to_use": "Factual Q&A, knowledge-base applications, documentation systems",
        },
        "Tool Use / Code Execution": {
            "problem_solved": "Arithmetic, counting, exact computation",
            "mechanism": "Give model access to Python interpreter; let it compute rather than guess",
            "limitation": "Increases latency; model must correctly write the tool call",
            "when_to_use": "Anything requiring precise numerical computation",
        },
        "Chain-of-Thought Prompting": {
            "problem_solved": "Multi-step reasoning errors, inconsistency",
            "mechanism": "Require model to show its work step by step before answering",
            "limitation": "Incorrect steps are possible even with reasoning shown",
            "when_to_use": "Complex logical problems, mathematical word problems",
        },
        "Structured Output + Validation": {
            "problem_solved": "Format failures, constraint violations",
            "mechanism": "Require JSON/schema output; validate before using",
            "limitation": "Doesn't catch semantically wrong but syntactically valid outputs",
            "when_to_use": "Whenever the output feeds into other code",
        },
        "Self-Consistency Sampling": {
            "problem_solved": "Random errors in reasoning",
            "mechanism": "Sample multiple times, take majority answer",
            "limitation": "Models can consistently be wrong; expensive",
            "when_to_use": "High-stakes decisions where compute budget allows",
        },
        "Human-in-the-loop for critical decisions": {
            "problem_solved": "All of the above",
            "mechanism": "Don't let the model make irreversible decisions autonomously",
            "limitation": "Defeats the purpose of automation for high-volume tasks",
            "when_to_use": "Medical, legal, financial, safety-critical applications",
        },
    }

    print("Practical defenses against LLM failure modes:")
    for name, details in defenses.items():
        print(f"\n{name}")
        print(f"  Solves: {details['problem_solved']}")
        print(f"  How: {details['mechanism'][:70]}...")
        print(f"  Use when: {details['when_to_use']}")

practical_defenses()
```

## The Honest Summary

Language models are very good at tasks that benefit from:
- Broad world knowledge expressed in natural language
- Pattern recognition in text
- Fluent generation in various styles and formats
- Code generation for common patterns
- Summarization and reorganization of provided content

Language models are unreliable for tasks requiring:
- Precise factual accuracy with no tolerance for error
- Real-time or post-training-cutoff information
- Exact computation
- Persistent state across sessions
- Reasoning that requires verifiable correctness

The failure mode that gets people in trouble is treating the second category like the first, because the model's output is always fluent and confident regardless of which category it's in.

A model that doesn't know the answer and a model that does both respond in grammatically correct, confident English. The only way to tell them apart is to check.

This isn't a problem that will be solved by making the model bigger. It's a property of the training objective. The model was trained to produce text that looks like the text humans write. Humans write confidently. The model writes confidently.

Build your systems accordingly.
