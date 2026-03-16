# SNP Foundation Models: Tokenization and Architecture

## Introduction: The Language of DNA

The central dogma of molecular biology—DNA makes RNA makes Protein—can be viewed through the lens of natural language processing (NLP). Just as Large Language Models (LLMs) learn the statistical properties of human language, **Genomic Foundation Models** learn the "language" of DNA.

| NLP Concept | Genomic Equivalent |
|---|---|
| **Text** | **DNA Sequence** (A, C, G, T) |
| **Tokenization** (Words/Subwords) | **K-mer Tokenization** (e.g., 6-mers like "ATCGAA") |
| **Language Model** (BERT, GPT) | **Genomic Language Model** (Nucleotide Transformer, DNABERT) |

### Why SNPs are Different from Text
While the architecture is similar, genomic data has unique challenges:

*   **Tiny Alphabet**: Only 4 characters (A, C, G, T) compared to thousands of words in human languages.
*   **Long Context**: Regulatory elements (enhancers) can be thousands of base pairs away from the gene they regulate.
*   **High Sensitivity**: A single letter change (SNP) can drastically alter a phenotype (e.g., cause a disease).

## Tokenization Strategies

Unlike English text, DNA has no natural delimiters like spaces. We need a strategy to break a continuous sequence of nucleotides into discrete tokens.

### K-mer Tokenization
The most common approach is **k-mer tokenization**. A k-mer is a specific sequence of length $k$.

*   **Sequence**: `ACGTACGT`
*   **6-mers (stride=1)**: `ACGTAC`, `CGTACG`, `GTACGT`...
*   **6-mers (stride=6, non-overlapping)**: `ACGTAC`, `GT....`

**Why K-mers?**

*   **Context**: A single nucleotide (A, C, G, T) carries very little information on its own. A k-mer (e.g., `TATAAA`, the TATA box) represents a functional unit or motif.
*   **Vocabulary Size**: With $k=6$, the vocabulary size is $4^6 = 4096$, which is manageable for Transformers.

!!! note "Key Insight for SNPs"
    A single SNP changes **at most one 6-mer token** (the one containing the variant position). The rest of the tokens are identical between the reference and alternate sequences.

### Template Code: Exploring K-mer Tokenization

```python
from transformers import AutoTokenizer

model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Example: ref and alt sequences differ at one position (G -> T)
ref_seq = "TATGAGAAAAAGG GTATGCCCCGT"
alt_seq = "TATGAGAAAAAAT GTATGCCCCGT"

# Tokenize both sequences
ref_tokens = tokenizer.tokenize(ref_seq.replace(" ", ""))
alt_tokens = tokenizer.tokenize(alt_seq.replace(" ", ""))

print("Ref tokens:", ref_tokens)
print("Alt tokens:", alt_tokens)

# Find which tokens differ
for i, (r, a) in enumerate(zip(ref_tokens, alt_tokens)):
    if r != a:
        print(f"  Differing token at position {i}: ref='{r}' → alt='{a}'")
```

**Expected output:**
```
Ref tokens: ['TATGAG', 'AAAAAG', 'GGTATG', 'CCCCGT']
Alt tokens: ['TATGAG', 'AAAAAA', 'TGTATG', 'CCCCGT']
  Differing token at position 1: ref='AAAAAG' → alt='AAAAAA'
  Differing token at position 2: ref='GGTATG' → alt='TGTATG'
```

Notice that a single SNP can affect **up to two adjacent tokens** (since the SNP can sit near a 6-mer boundary), but leaves all other tokens unchanged.

## Model Architecture: Encoder vs. Decoder

When building a foundation model for SNPs, we must choose between an **Encoder-only** (BERT-style) or **Decoder-only** (GPT-style) architecture.

### Encoder-Only (The Choice for SNPs)
Models like **DNABERT** and **Nucleotide Transformer** use an **Encoder-only** architecture.

*   **Mechanism**: Bidirectional attention. Every token attends to every other token in the sequence (left and right).
*   **Why it fits SNPs**: To predict the effect of a SNP, the model needs to understand the **full context**—both the upstream promoter and downstream coding regions. A SNP's function is determined by its surrounding environment.
*   **Task**: Discriminative tasks (Classification, Regression).

### Decoder-Only (Generative)
Models like **HyenaDNA** or **GenomicGPT** might use Decoder-only architectures.

*   **Mechanism**: Causal attention. Tokens only attend to previous tokens.
*   **Use Case**: Generating new DNA sequences (designing synthetic promoters).
*   **Limitation for SNPs**: Less effective for understanding the relationship between a SNP and its downstream context simultaneously.

### Summary
For **Variant Effect Prediction** (predicting if a SNP causes a disease), **Encoder-only models** are generally superior because they build a comprehensive representation of the entire sequence context.

### Template Code: Loading the Pretrained Model

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

print(f"Vocab size: {tokenizer.vocab_size}")          # ~4107 tokens
print(f"Model max length: {tokenizer.model_max_length}")  # 2048 tokens

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()
```
