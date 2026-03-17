# Training Genomic Foundation Models

Once we have a tokenizer and an architecture (Encoder-only), we need to train the model. This involves **Pretraining** on massive unlabeled datasets and **Fine-Tuning** on specific labeled tasks.

## Pretraining Objectives

The goal of pretraining is to force the model to learn the syntax (grammar) and semantics (regulatory motifs) of DNA without human labels.

### Masked Language Modeling (MLM)
This is the standard objective for Encoder models (like BERT).

*   **Process**: Randomly mask a percentage (e.g., 15%) of the k-mer tokens in the input sequence.
*   **Goal**: Predict the original identity of the masked tokens based on the surrounding context.
*   **Formula**: $$ P(x_i \mid x_{\setminus i}) $$
*   **Biological Intuition**: To predict a masked region, the model must learn patterns like "TATA box is usually followed by a transcription start site" or "this motif pairs with that motif."

![mlm](./assets/mlm.png)

### Template Code: Applying MLM Masking

```python
import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# The DataCollator handles random masking automatically (15% by default)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15  # mask 15% of k-mer tokens
)

# Example: tokenize a DNA sequence, then apply masking
sequence = "ACGTACGTACGTACGTACGT"
tokenized = tokenizer(sequence, return_tensors="pt")

# Collate into a batch and apply masking
batch = data_collator([{"input_ids": tokenized["input_ids"][0]}])
print("Original IDs: ", tokenized["input_ids"])
print("Masked IDs:   ", batch["input_ids"])   # Some tokens replaced with [MASK]
print("Labels:       ", batch["labels"])        # -100 means 'not masked, ignore'
```
<!-- 
### Other Pretraining Tasks
Beyond MLM, models can be trained on auxiliary tasks to boost performance:

1.  **Species Classification**: Predict which species the DNA sequence comes from (e.g., Human vs. Mouse vs. Yeast). This helps the model learn evolutionary conservation.
2.  **Promoter Prediction**: Predict if a sequence is a promoter region.
3.  **Chromatin Profile Prediction**: Predict chromatin accessibility (ATAC-seq) or histone marks directly from sequence (like the **Enformer** objective).

## Datasets for Downstream SNP Tasks

We use real genomic benchmark datasets that contain paired SNP sequences with experimental labels.

### The GTEx eQTL Dataset

The **GTEx (Genotype-Tissue Expression)** project provides genome-wide eQTL associations across 54 human tissues. We use the [Genomics Long-Range Benchmark](https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark) (InstaDeepAI) which packages this data for model evaluation.

Each example contains:

| Field | Description |
|---|---|
| `ref_forward_sequence` | Full DNA window (e.g., 100kb) with the reference allele |
| `alt_forward_sequence` | Same window with the alternate (SNP) allele |
| `tissue` | The tissue context (e.g., "Whole_Blood", "Liver") |
| `distance_to_nearest_tss` | Distance to nearest transcription start site |
| `label` | `1` = causal eQTL, `0` = non-causal |

### Template Code: Loading the Dataset

```python
from datasets import load_dataset

# Load the causal eQTL benchmark
dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name="variant_effect_causal_eqtl",
    sequence_length=2048,   # use 2kb windows around each SNP
    trust_remote_code=True,
)

train_ds = dataset["train"].shuffle(seed=42).select(range(4000))
test_ds  = dataset["test"].shuffle(seed=42).select(range(800))

# Inspect one example
ex = train_ds[0]
ref, alt = ex["ref_forward_sequence"], ex["alt_forward_sequence"]

# Find the SNP position
snp_pos = next(i for i, (r, a) in enumerate(zip(ref, alt)) if r != a)
print(f"Tissue: {ex['tissue']},  Label: {ex['label']}")
print(f"Ref: ...{ref[snp_pos-5:snp_pos]}[{ref[snp_pos]}]{ref[snp_pos+1:snp_pos+6]}...")
print(f"Alt: ...{alt[snp_pos-5:snp_pos]}[{alt[snp_pos]}]{alt[snp_pos+1:snp_pos+6]}...")
```

**Expected output:**
```
Tissue: Spleen,  Label: 1
Ref: ...AAAAAA[G]GTATG...
Alt: ...AAAAAA[T]GTATG...
``` -->
