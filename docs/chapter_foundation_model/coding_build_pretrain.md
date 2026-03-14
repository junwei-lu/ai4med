# Coding Tutorial: Build and Pretrain a Foundation Model (SNP DNA sequences)

This tutorial shows a minimal end-to-end example for pretraining a tiny masked language model (MLM) on SNP-centered DNA sequences using the Hugging Face ecosystem. We use k-mer tokenization for DNA instead of a text tokenizer.

## 1) Setup and Data (SNP sequences)

- Install: `datasets`, `transformers`, `accelerate`, `torch`.
- Load variant-centered DNA windows from a simple public dataset. We will use a small subset for speed.

```python
from datasets import load_dataset

# Variant-centered DNA sequences (e.g., ClinVar windows)
# If needed, change the config string to another SNP task available
# on the same dataset (e.g., "clinvar_5class").
raw = load_dataset("InstaDeepAI/genomics-long-range-benchmark", "clinvar_2class")

# We only need sequences for pretraining (labels are ignored here)
train_sequences = [ex.get("sequence") for ex in raw["train"]][:2000]
train_sequences = [s for s in train_sequences if isinstance(s, str) and len(s) > 0]
```

## 2) Tokenization for DNA: k-mers (non-text input)

- For DNA, we split sequences into overlapping k-mers (substrings of length k). A common choice is k=6.
- We build a tiny vocabulary: all possible 6-mers over A,C,G,T plus special tokens.
- Any k-mer containing ambiguous bases (e.g., `N`) maps to `[UNK]`.

```python
from itertools import product

# Build a 6-mer vocabulary for DNA
K = 6
alphabet = ["A", "C", "G", "T"]
kmer_list = ["".join(p) for p in product(alphabet, repeat=K)]

SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[MASK]": 2,
    "[CLS]": 3,
    "[SEP]": 4,
}

# Map each k-mer to an id after the specials
kmer_to_id = {kmer: i + len(SPECIAL_TOKENS) for i, kmer in enumerate(kmer_list)}
id_to_kmer = {v: k for k, v in kmer_to_id.items()}

PAD_ID = SPECIAL_TOKENS["[PAD]"]
UNK_ID = SPECIAL_TOKENS["[UNK]"]
MASK_ID = SPECIAL_TOKENS["[MASK]"]
CLS_ID = SPECIAL_TOKENS["[CLS]"]
SEP_ID = SPECIAL_TOKENS["[SEP]"]

def dna_to_kmers(seq: str, k: int = K):
    seq = seq.upper()
    if len(seq) < k:
        return []
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def encode_sequence(seq: str, max_length: int = 256):
    # Convert DNA → k-mers → ids, add [CLS]/[SEP], then pad/truncate
    kmers = dna_to_kmers(seq)
    token_ids = [CLS_ID]
    for t in kmers:
        if set(t) <= {"A", "C", "G", "T"}:
            token_ids.append(kmer_to_id.get(t, UNK_ID))
        else:
            token_ids.append(UNK_ID)
    token_ids.append(SEP_ID)

    token_ids = token_ids[:max_length]
    attention_mask = [1] * len(token_ids)
    # pad
    while len(token_ids) < max_length:
        token_ids.append(PAD_ID)
        attention_mask.append(0)
    return {"input_ids": token_ids, "attention_mask": attention_mask}
```

## 3) Tiny Transformer for DNA MLM

- We instantiate a very small `BertForMaskedLM` from scratch with our custom vocabulary size.

```python
import torch
from transformers import BertConfig, BertForMaskedLM

VOCAB_SIZE = len(SPECIAL_TOKENS) + len(kmer_list)

config = BertConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=256,
    max_position_embeddings=512,
    pad_token_id=PAD_ID,
)

model = BertForMaskedLM(config)
```

## 4) Prepare dataset and a simple MLM collator (k-mer aware)

- We implement a lightweight PyTorch dataset and a masking collator (80/10/10 rule) that works without a Hugging Face text tokenizer.

```python
from datasets import Dataset
import random

MAX_LEN = 256

dataset = Dataset.from_dict({"sequence": train_sequences})

def preprocess_batch(batch):
    encoded = [encode_sequence(s, MAX_LEN) for s in batch["sequence"]]
    return {
        "input_ids": [e["input_ids"] for e in encoded],
        "attention_mask": [e["attention_mask"] for e in encoded],
    }

tokenized = dataset.map(preprocess_batch, batched=True, remove_columns=["sequence"]).with_format("torch")

def mlm_data_collator(features, mlm_probability: float = 0.15):
    # Stack
    input_ids = torch.stack([f["input_ids"] for f in features])
    attention_mask = torch.stack([f["attention_mask"] for f in features])

    labels = input_ids.clone()

    # Create mask over non-special tokens
    special_ids = torch.tensor([PAD_ID, MASK_ID, CLS_ID, SEP_ID])
    is_special = (input_ids.unsqueeze(-1) == special_ids).any(-1)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    probability_matrix.masked_fill_(is_special, 0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Set unmasked positions to -100 for loss ignoring
    labels[~masked_indices] = -100

    # 80% replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = MASK_ID

    # 10% replace with random token (non-special)
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_tokens = torch.randint(len(SPECIAL_TOKENS), VOCAB_SIZE, labels.shape)
    input_ids[indices_random] = random_tokens[indices_random]
    # remaining 10% keep original

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
```

## 5) Pretrain with Trainer (short, CPU-friendly)

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="./snp-mlm-tiny",
    per_device_train_batch_size=16,
    learning_rate=5e-4,
    num_train_epochs=1,
    logging_steps=50,
    save_strategy="no",
    evaluation_strategy="no",
    seed=224,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=mlm_data_collator,
)

trainer.train()
```

## 6) Quick sanity check (mask a k-mer and predict)

```python
test_seq = train_sequences[0]
enc = encode_sequence(test_seq, MAX_LEN)
inp = torch.tensor([enc["input_ids"]])
att = torch.tensor([enc["attention_mask"]])

# Find a non-special token to mask (skip [CLS] at index 0)
pos = 1
while pos < inp.size(1) and inp[0, pos] in (PAD_ID, CLS_ID, SEP_ID):
    pos += 1
if pos < inp.size(1):
    original = int(inp[0, pos].item())
    inp[0, pos] = MASK_ID

with torch.no_grad():
    out = model(input_ids=inp, attention_mask=att)
    pred_id = out.logits[0, pos].argmax().item()

print("Original token id:", original, "Predicted id:", pred_id,
      "Predicted token:", id_to_kmer.get(pred_id, "[SPECIAL]"))
```

That’s it—an end-to-end, minimal example for SNP DNA sequences: build a k-mer tokenizer, a tiny transformer, and pretrain with masked k-mer prediction on a toy subset of variant-centered windows.

> Notes
> - The dataset field name for sequences is `sequence` in the example above. If yours differs, update the extraction accordingly.
> - This is for demonstration only. For real use, increase model size, k, data volume, and training time; consider established genomic tokenizers/models (e.g., DNABERT, Nucleotide Transformer).
