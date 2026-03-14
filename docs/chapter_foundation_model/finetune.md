## Fine-Tuning a Foundation Model with Hugging Face (SNP DNA sequences)

We fine-tune the SNP sequence encoder from the pretraining tutorial on simple downstream tasks using variant-centered DNA windows. Instead of text, inputs are DNA sequences tokenized into k-mers.

### What SNP sequence tasks can we fine-tune for?

- Sequence Classification (binary): ClinVar pathogenic vs. benign (`clinvar_2class`).
- Sequence Classification (multi-class): ClinVar 5-way labels (`clinvar_5class`).

Below we show two small recipes. They reuse the same k-mer tokenizer idea from pretraining.

### 0) Shared setup (k-mer tokenizer and tiny config)

```python
from itertools import product
from datasets import load_dataset, Dataset
import torch
from transformers import BertConfig, BertForSequenceClassification, Trainer, TrainingArguments

# --- k-mer tokenizer (same as in pretraining) ---
K = 6
alphabet = ["A", "C", "G", "T"]
kmer_list = ["".join(p) for p in product(alphabet, repeat=K)]

SPECIAL_TOKENS = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "[CLS]": 3, "[SEP]": 4}
kmer_to_id = {kmer: i + len(SPECIAL_TOKENS) for i, kmer in enumerate(kmer_list)}
id_to_kmer = {v: k for k, v in kmer_to_id.items()}

PAD_ID = SPECIAL_TOKENS["[PAD]"]
UNK_ID = SPECIAL_TOKENS["[UNK]"]
CLS_ID = SPECIAL_TOKENS["[CLS]"]
SEP_ID = SPECIAL_TOKENS["[SEP]"]

def dna_to_kmers(seq: str, k: int = K):
    seq = seq.upper()
    if len(seq) < k:
        return []
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def encode_sequence(seq: str, max_length: int = 256):
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
    while len(token_ids) < max_length:
        token_ids.append(PAD_ID)
        attention_mask.append(0)
    return {"input_ids": token_ids, "attention_mask": attention_mask}

VOCAB_SIZE = len(SPECIAL_TOKENS) + len(kmer_list)
MAX_LEN = 256

clf_config = BertConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=256,
    max_position_embeddings=512,
    pad_token_id=PAD_ID,
)
```

### 1) ClinVar binary classification (pathogenic vs benign)

```python
# Load SNP-centered sequences (labels ignored during pretrain, used here)
raw = load_dataset("InstaDeepAI/genomics-long-range-benchmark", "clinvar_2class")

# Build a tiny train/val split from the provided train split for speed
train = raw["train"].train_test_split(test_size=0.2, seed=42)

def to_features(batch):
    enc = [encode_sequence(s, MAX_LEN) for s in batch["sequence"]]
    return {
        "input_ids": [e["input_ids"] for e in enc],
        "attention_mask": [e["attention_mask"] for e in enc],
        "labels": batch["label"],  # adjust if your field is named differently
    }

train_ds = train["train"].map(to_features, batched=True, remove_columns=train["train"].column_names)
val_ds = train["test"].map(to_features, batched=True, remove_columns=train["test"].column_names)
train_ds.set_format("torch")
val_ds.set_format("torch")

model = BertForSequenceClassification(clf_config, num_labels=2)

args = TrainingArguments(
    output_dir="./snp-clinvar2",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=5e-4,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    logging_steps=50,
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

trainer.train()
```

### 2) ClinVar 5-class classification

```python
raw = load_dataset("InstaDeepAI/genomics-long-range-benchmark", "clinvar_5class")
train = raw["train"].train_test_split(test_size=0.2, seed=42)

def to_features5(batch):
    enc = [encode_sequence(s, MAX_LEN) for s in batch["sequence"]]
    return {
        "input_ids": [e["input_ids"] for e in enc],
        "attention_mask": [e["attention_mask"] for e in enc],
        "labels": batch["label"],
    }

train_ds5 = train["train"].map(to_features5, batched=True, remove_columns=train["train"].column_names)
val_ds5 = train["test"].map(to_features5, batched=True, remove_columns=train["test"].column_names)
train_ds5.set_format("torch")
val_ds5.set_format("torch")

model5 = BertForSequenceClassification(clf_config, num_labels=5)

args5 = TrainingArguments(
    output_dir="./snp-clinvar5",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=5e-4,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    logging_steps=50,
    save_strategy="no",
)

trainer5 = Trainer(
    model=model5,
    args=args5,
    train_dataset=train_ds5,
    eval_dataset=val_ds5,
)

trainer5.train()
```

### Practical notes

- Keep the same k-mer tokenizer between pretraining and fine-tuning. If you saved a pretrained checkpoint, you can load its encoder weights into the classification model to warm start.
- If your dataset uses different field names for the sequence or label, adapt the keys in the mapping functions accordingly (e.g., `"sequence"`, `"label"`).
- Start tiny (short sequences, small model, few steps) to validate the pipeline, then scale up.
