# SNP Foundation Models Basics

<a href="https://colab.research.google.com/github/junwei-lu/ai4med/blob/main/codes/snp/snp_genomics_workshop.ipynb#scrollTo=J6wHfT9mYxA2" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; background: linear-gradient(135deg, #1565c0 0%, #42a5f5 100%); color: white; border-radius: 6px; text-decoration: none; font-size: 0.85em; font-weight: 600;">▶ Try in Colab</a>

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

## Architecture Principles of Genomic Foundation Models

After tokenization, a genomic foundation model still needs a neural architecture that can combine information across the whole sequence. Most current DNA foundation models follow the same high-level recipe as NLP foundation models: **token embedding + positional encoding + Transformer blocks + self-supervised pretraining objective**.

### Why Transformers Are Common

Transformers are popular for genomic sequences because they can model interactions between distant positions in the input.

*   **Token embeddings** map each nucleotide token or k-mer token into a dense vector.
*   **Positional information** tells the model where each token occurs in the sequence.
*   **Self-attention** lets one motif attend to other motifs elsewhere in the same sequence.
*   **Stacked layers** build increasingly abstract representations, from local motifs to higher-order regulatory patterns.

This is useful in genomics because the function of a SNP usually depends on **context**: nearby motifs, promoter regions, splice signals, and sometimes more distal regulatory elements.

### Encoder-Only vs. Decoder-Only

When building a genomic foundation model, we usually choose between two Transformer styles.

#### Encoder-only models
Models such as **DNABERT** and **Nucleotide Transformer** use **bidirectional attention**.

*   Each token can attend to both its left and right context.
*   This is well suited for **classification**, **regression**, and **variant effect prediction**.
*   It matches SNP analysis well, because interpreting a variant requires the full surrounding sequence.

#### Decoder-only models
Generative DNA models use **causal attention**.

*   Each token only attends to previous tokens.
*   This is useful for **sequence generation** or **design** tasks.
*   It is usually less natural than encoder-only modeling for SNP interpretation, where we want a contextual representation of the whole locus.

For most downstream SNP tasks in this course, the practical default is therefore an **encoder-only Transformer**.

## Example Model: nucleotide-transformer-v2-50m-multi-species`

As a concrete example, consider `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`, one of the most accessible pretrained genomic foundation models on Hugging Face.

### High-Level Design

This model is a **BERT-style masked language model for DNA**.

*   **Backbone**: encoder-only Transformer
*   **Model size**: about **56M parameters**
*   **Tokenizer**: primarily **6-mer tokenization**, with single-nucleotide fallback when needed
*   **Vocabulary size**: **4107** tokens
*   **Context window**: `max_position_embeddings = 2050`, corresponding to sequences of about **1000 tokens** during pretraining plus special tokens
*   **Pretraining objective**: **masked language modeling (MLM)** on DNA tokens

The model was pretrained on a **multi-species genome collection**, which helps it learn regulatory patterns that generalize beyond a single reference genome.

### Transformer Configuration

Its architecture is small enough to run in teaching settings, but still large enough to illustrate the main ideas of genomic foundation models.

*   **Hidden size**: **512**
*   **Number of Transformer layers**: **12**
*   **Attention heads**: **16**
*   **Feed-forward dimension**: **2048**
*   **Positional encoding**: **rotary positional embeddings**
*   **V2 design change**: introduces **Gated Linear Units (GLUs)** in the feed-forward sublayers

These choices reflect a common design pattern: use a moderate-size encoder so the model can learn rich sequence representations without the extreme computational cost of the largest language models.

### Why This Architecture Fits SNP Analysis

This example is especially useful for SNP tasks because:

*   **bidirectional attention** helps the model interpret a variant using both upstream and downstream context
*   **k-mer tokenization** gives each token more biological meaning than a single nucleotide alone
*   **masked pretraining** teaches the model sequence grammar before any task-specific labels are introduced
*   the model can later be adapted for **classification**, **regression**, or **embedding extraction**

In practice, we often use this pretrained encoder in one of two ways:

1.  extract embeddings from the frozen model and train a small downstream predictor, or
2.  attach a task head and fine-tune the model on labeled SNP data.

### Loading the Pretrained Model

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
