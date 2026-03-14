# Building a Foundation Model (No Code)

This lecture outlines the general steps to build a foundation model, using a tiny SNP-related dataset from Hugging Face as a running example. It is designed for beginners with backgrounds in biostatistics or biomedical research. We keep concepts simple and focus on high-level ideas.

## 1) Build the Transformer Model

### 1.1 Tokenization of SNP Data

- Why tokenize? Transformers operate on discrete tokens. Tokenization turns raw inputs into tokens and IDs that models can process.
- Options for genomic/SNP contexts:
  - K-mer tokenization: split DNA sequences into overlapping subsequences of length k (e.g., 3–6). This captures local patterns and keeps input lengths manageable.
  - Standard subword tokenization (e.g., WordPiece/BPE): for biomedical text that mentions SNPs (e.g., "rs1234" or variant descriptions like "c.76A>T"), standard tokenizers work well and are the simplest starting point.
- Practical choice in this chapter: We will use a small text dataset where SNP mentions appear in biomedical sentences. Standard Hugging Face tokenizers keep the workflow very simple. If you later work directly with DNA sequences, you can swap in a k-mer tokenizer.

### 1.2 Transformer Architecture

- Core components:
  - Embeddings for tokens and positions
  - Multi-head self-attention to learn context across the sequence
  - Feed-forward (MLP) layers and residual connections
  - Layer normalization and dropout for stability
- Practical choice: Reuse a well-tested masked language modeling backbone (e.g., a small BERT variant) to minimize engineering. This lets you focus on data and evaluation rather than low-level implementation details.

## 2) Pretraining with Masked Language Modeling (MLM)

- Idea: Randomly mask some input tokens and train the model to predict them from surrounding context.
- Why MLM for SNP-related text? Even a small corpus of sentences that mention variants helps the model learn domain-specific vocabulary and context (e.g., how SNP identifiers appear and co-occur with genes or diseases).
- Data example: A tiny biomedical text dataset on Hugging Face that contains mutation/SNP mentions (e.g., `bigbio/tmvar_v1`). We treat its text as unlabeled data for MLM, purely for illustration.
- Practical notes:
  - Use Hugging Face Datasets to stream and process text.
  - Use the Transformers `DataCollatorForLanguageModeling` to apply on-the-fly masking.
  - Train briefly on CPU or a single GPU—it’s just a toy demonstration.

## 3) Fine-Tuning for Downstream Tasks

- After pretraining, adapt the model to specific tasks related to SNPs and biomedical text:
  - Token classification (NER): find SNP mentions in sentences.
  - Sequence classification: predict whether a sentence contains a SNP mention, or categorize the type of mention.
  - (Advanced) Relation extraction: determine if an SNP is linked to a gene, disease, or phenotype in a sentence.
- Practical notes:
  - Reuse the pretrained backbone and switch the head to match your task (classification or token classification).
  - Keep evaluation simple (accuracy/F1) and prioritize readability of the pipeline.

## Why a tiny dataset?

- Purpose: Demonstrate the workflow end-to-end without heavy compute.
- Outcome: You will understand how to structure projects, choose tokenization, reuse transformer backbones, pretrain with MLM, and fine-tune for tasks—skills that scale to larger datasets later.

> Tip: Start with standard tokenizers and small models. Only introduce specialized tokenization (e.g., k-mers) when you move to raw genomic sequence modeling.
