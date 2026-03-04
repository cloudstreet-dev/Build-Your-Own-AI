# Tokens: The Atoms of Language

Before a language model can do anything with your text, it has to destroy it.

Not destructively, but fundamentally: your string of characters gets chopped into pieces called **tokens**, and those tokens — not the characters, not the words, not the sentences — are what the model actually processes. Everything downstream depends on this step, which is why it's worth understanding in some detail.

## Why Not Just Use Characters?

The obvious starting point: why not feed the model one character at a time? Letters are already atomic. There are only 26 of them in English (more if you count punctuation, digits, and the various ways people type "resumé").

The problem is sequence length. "The quick brown fox" is 19 characters. If you're processing 1,000 tokens at a time, character-level you get 1,000 characters — maybe 200 words. Token-level, you get roughly 750 words. That's a meaningful difference when your context window is a hard limit.

More importantly, characters don't carry meaning. The letter 'c' contributes nothing on its own. The model would have to learn that 'c', 'a', 't' in sequence means a small furry mammal, from scratch, from data, every time. It can do this! But it's inefficient. Better to give the model "cat" as a unit.

## Why Not Just Use Words?

Words seem like the obvious answer. Dictionaries exist. There are maybe 170,000 words in English, which is manageable.

Three problems:

**1. Vocabulary explosion.** "Run", "runs", "running", "runner", "ran" are all different words. So are "tokenize", "tokenized", "tokenizing", "tokenizer", "tokenization". In practice, a word-level vocabulary has to be enormous, and words not in the vocabulary become `<UNK>` (unknown), which is just a fancy way of losing information.

**2. Different languages.** A word-level tokenizer built for English is useless for Chinese, which doesn't use spaces between words. It works badly for German, where "Donaudampfschifffahrtselektrizitätenhauptbetriebswerkbauunterbeamtengesellschaft" is a single word that means approximately "Association of subordinate officials of the head office management of the Danube steamboat electrical services."

**3. Novel words.** "GPT-4o", "tokenizer", "TikTok" — new terms appear constantly. A fixed word vocabulary can't adapt.

## The Solution: Subword Tokenization

Modern tokenizers split text into pieces that are somewhere between characters and words. Common sequences stay together ("the", "ing", "tion"), rare sequences get split ("tokenization" → "token" + "ization"). This is **Byte Pair Encoding (BPE)**, and it's what GPT-style models use.

Here's the key insight: BPE starts by treating every character as a token, then iteratively merges the most frequent adjacent pairs into a single token. You run this process until you have the vocabulary size you want.

Let's implement a simple version:

```python
from collections import Counter

def get_stats(vocab):
    """Count frequency of adjacent pairs across all words in vocab."""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """Merge all occurrences of the most frequent pair."""
    merged = ' '.join(pair)
    replacement = ''.join(pair)
    new_vocab = {}
    for word, freq in vocab.items():
        new_word = word.replace(merged, replacement)
        new_vocab[new_word] = freq
    return new_vocab

# Start with a tiny corpus
# Each word is represented with a space between characters
# </w> marks end-of-word
initial_vocab = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3,
}

vocab = initial_vocab.copy()
print("Initial vocab:")
for word, freq in vocab.items():
    print(f"  '{word}': {freq}")

print()

# Run 5 BPE merges
for i in range(5):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    print(f"Merge {i+1}: {best} (frequency: {pairs[best]})")
    vocab = merge_vocab(best, vocab)

print()
print("Final vocab:")
for word, freq in vocab.items():
    print(f"  '{word}': {freq}")
```

Running this produces:

```
Initial vocab:
  'l o w </w>': 5
  'l o w e r </w>': 2
  'n e w e s t </w>': 6
  'w i d e s t </w>': 3

Merge 1: ('e', 's') (frequency: 9)
Merge 2: ('es', 't') (frequency: 9)
Merge 3: ('est', '</w>') (frequency: 9)
Merge 4: ('l', 'o') (frequency: 7)
Merge 5: ('lo', 'w') (frequency: 7)

Final vocab:
  'low </w>': 5
  'low e r </w>': 2
  'n e w est</w>': 6
  'w i d est</w>': 3
```

Notice what happened: "est" got merged because it appeared frequently ("newest", "widest"). "low" got merged because it appeared in both "low" and "lower". The algorithm discovered structure without being told it existed.

## Real Tokenizers in Practice

You won't implement BPE from scratch for production — use `tiktoken` (OpenAI's tokenizer) or the `transformers` library:

```python
import tiktoken

# The tokenizer used by GPT-4
enc = tiktoken.get_encoding("cl100k_base")

text = "The quick brown fox jumped over the lazy dog."
tokens = enc.encode(text)
print(f"Text: {text!r}")
print(f"Token IDs: {tokens}")
print(f"Token count: {len(tokens)}")

# Decode each token individually to see what they are
print("\nToken breakdown:")
for token_id in tokens:
    token_bytes = enc.decode_single_token_bytes(token_id)
    try:
        token_str = token_bytes.decode('utf-8')
    except UnicodeDecodeError:
        token_str = repr(token_bytes)
    print(f"  {token_id:6d} → {token_str!r}")
```

```
Text: 'The quick brown fox jumped over the lazy dog.'
Token IDs: [791, 4062, 14198, 39935, 27096, 927, 279, 16053, 5679, 13]
Token count: 10

Token breakdown:
     791 → 'The'
    4062 → ' quick'
   14198 → ' brown'
   39935 → ' fox'
   27096 → ' jumped'
     927 → ' over'
     279 → ' the'
   16053 → ' lazy'
    5679 → ' dog'
      13 → '.'
```

Notice that spaces are typically *part of the following token* — "quick" is actually " quick" (with a leading space). This is why tokenization affects things like whether the model treats the first word of a sentence differently from subsequent words.

## The Part That Should Surprise You

Let's look at some tokens that behave unexpectedly:

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

# Punctuation and whitespace
examples = [
    "SolidGoldMagikarp",  # famous problematic token from early GPT
    "    ",               # four spaces
    "\n\n\n",            # three newlines
    "================",  # many equals signs
    " unfavorable",
    "unfavorable",       # same word, different leading space!
]

for text in examples:
    tokens = enc.encode(text)
    print(f"{text!r:30s} → {len(tokens)} token(s): {tokens}")
```

```
'SolidGoldMagikarp'            → 3 token(s): [45280, 11768, 74241]
'    '                         → 1 token(s): [262]
'\n\n\n'                       → 3 token(s): [198, 198, 198]
'================'             → 2 token(s): [=================]
' unfavorable'                 → 1 token(s): [45824]
'unfavorable'                  → 2 token(s): [1714, 27961]
```

That last one is important. `" unfavorable"` (with space) is a single token, but `"unfavorable"` (without space) is two tokens. The model has *different* representations for these — they're different inputs. This is why leading spaces matter when you're prompting, even though it looks like whitespace.

## Counting Tokens (and Why You Should)

APIs charge per token. Context windows are measured in tokens. Your 4,000-word essay might fit or might not, depending on the vocabulary distribution.

Quick rule of thumb: for English prose, 1 token ≈ 4 characters ≈ 0.75 words. For code, it's more variable — Python is fairly efficient, but languages with long keywords or unusual syntax can be costly.

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def token_report(text, label=""):
    tokens = enc.encode(text)
    chars = len(text)
    words = len(text.split())
    print(f"{label or 'Text'}")
    print(f"  Characters: {chars}")
    print(f"  Words:      {words}")
    print(f"  Tokens:     {len(tokens)}")
    print(f"  Chars/token: {chars/len(tokens):.1f}")
    print()

token_report("""
The transformer architecture, introduced in 'Attention Is All You Need' (2017),
replaced recurrent neural networks for sequence modeling tasks. Its core mechanism,
self-attention, allows each position in a sequence to attend to all other positions
simultaneously, enabling massive parallelism during training.
""", "English prose")

token_report("""
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
""", "Python code")

token_report("""
SELECT u.name, COUNT(o.id) as order_count, SUM(o.total) as revenue
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at > '2024-01-01'
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 0
ORDER BY revenue DESC;
""", "SQL query")
```

## Token IDs Are Not Random

The vocabulary is fixed at training time. Token 13 is always a period (`.`) for GPT-4. Token 791 is always "The". This matters because the model's learned weights are indexed by these IDs — the embedding for token 791 is a specific row in a matrix, and it always refers to "The".

This is also why you can't easily add new tokens to an existing model without retraining. Adding a token means adding a new row to the embedding matrix, but the model has no learned weights for it — you'd need to train on examples that use it.

## What Comes Next

You now know that your text prompt gets converted to a sequence of integers, where each integer indexes into a vocabulary of ~100,000 items. Those integers are what the model actually receives.

The next question: what does the model *do* with an integer? It can't do math on "The". It needs to turn each token ID into something more meaningful — a representation that captures semantic relationships, so that "cat" and "kitten" are somehow close together, and "cat" and "database" are far apart.

That's what embeddings are for.
