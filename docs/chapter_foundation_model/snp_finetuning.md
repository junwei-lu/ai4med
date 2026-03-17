# Fine-Tuning for SNP Prediction

## What Fine-Tuning Means

Pretraining teaches a genomic foundation model the general grammar of DNA. **Fine-tuning** is the next step: we adapt that pretrained model to a specific **downstream task** with labeled data.

For SNP applications, the downstream task is usually not “predict the next DNA token.” Instead, it is something biologically meaningful, such as:

*   does this variant affect gene expression?
*   is this variant likely pathogenic?
*   does this mutation disrupt a splice site?

So the main idea of fine-tuning is simple:

1.  start from a pretrained genomic encoder,
2.  define a labeled downstream task,
3.  add an output head appropriate for that task, and
4.  train on the downstream labels.

## General Principles for Fine-Tuning SNP Models

Although datasets differ, the workflow is usually the same.

### 1. Define the downstream prediction target

The output can be:

*   **binary classification**: causal vs non-causal variant
*   **multiclass classification**: tissue class or mechanism class
*   **regression**: effect size, expression change, or assay signal

The prediction head should match the label type.

### 2. Build the right model input

For SNP tasks, one sample often contains both:

*   the **reference** sequence, and
*   the **alternate** sequence containing the variant.

There are several valid input designs:

*   encode only the **reference** sequence
*   encode **reference and alternate separately** and compare them
*   derive a **delta embedding** or **delta prediction** between ref and alt

For variant effect problems, comparing the two alleles is often the most natural choice.

### 3. Choose how much of the pretrained model to update

There are three common strategies:

*   **feature extraction**: freeze the backbone and train only a small classifier
*   **head-only or partial fine-tuning**: update a small number of layers
*   **full or parameter-efficient fine-tuning**: adapt more of the model for best task performance

In a teaching workflow, it is usually best to start with the simplest baseline first: frozen embeddings plus a lightweight classifier.

### 4. Evaluate with task-appropriate metrics

Common evaluation choices are:

*   **AUROC / AUPRC** for imbalanced classification
*   **accuracy / F1** for balanced classification
*   **Pearson or Spearman correlation** for regression

The biological interpretation matters as much as the metric. For SNPs, we often care whether the model can rank truly functional variants above matched negatives.

## General Template for Downstream Fine-Tuning

The following template shows the standard logic independent of the exact dataset.

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import numpy as np
from sklearn.metrics import roc_auc_score

model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 1. Load a labeled downstream dataset
dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name="variant_effect_causal_eqtl",
    sequence_length=2048,
    trust_remote_code=True,
)

# 2. Choose an input representation
def preprocess(example):
    # Simple baseline: tokenize the reference sequence only.
    # More advanced SNP pipelines can encode ref/alt separately.
    tokens = tokenizer(
        example["ref_forward_sequence"],
        truncation=True,
        max_length=512,
    )
    tokens["labels"] = example["label"]
    return tokens

tokenized = dataset.map(preprocess)

# 3. Load pretrained model + downstream head
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    trust_remote_code=True,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = logits[:, 1]
    return {"auroc": roc_auc_score(labels, probs)}

training_args = TrainingArguments(
    output_dir="snp_downstream_run",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
```

This template is intentionally simple. In real SNP modeling, we often improve it by making the model compare the **reference** and **alternate** alleles explicitly.

## Common Downstream SNP Tasks

After pretraining, the same backbone can be adapted to several kinds of biological questions.

### Variant Effect Prediction (VEP)

*   **Input**: reference and alternate alleles
*   **Task**: predict how the mutation changes pathogenicity, chromatin state, or expression
*   **Metric**: AUROC / AUPRC for classification or correlation for quantitative assays
*   **Useful benchmark examples**:
    *   [`InstaDeepAI/genomics-long-range-benchmark`](https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark)
    *   `task_name="variant_effect_pathogenic_clinvar"`
    *   `task_name="variant_effect_pathogenic_omim"`

### eQTL classification

*   **Input**: a DNA window around the SNP, often with tissue context
*   **Task**: predict whether the variant is a **causal eQTL**
*   **Benchmark example**:
    *   [`InstaDeepAI/genomics-long-range-benchmark`](https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark)
    *   `task_name="variant_effect_causal_eqtl"`

### Splice effect prediction

*   **Task**: predict whether the mutation disrupts exon-intron boundaries or splicing regulation
*   **Use case**: especially relevant for rare disease interpretation

These tasks differ in labels, but the fine-tuning logic is the same: pretrained DNA encoder first, task-specific head second.

## Main Example: Fine-Tuning for eQTL Prediction

For this course, the most useful worked example is **eQTL prediction**. An **expression quantitative trait locus (eQTL)** is a variant associated with changes in gene expression.

The concrete downstream question is:

> Given a genomic sequence containing a SNP, can the model predict whether that SNP is a **causal eQTL**?

This is a good teaching example because it connects sequence modeling to a clinically meaningful outcome: **how a genetic variant changes gene regulation**.

### Why eQTL is a good example

The eQTL benchmark is particularly useful because it includes:

*   `ref_forward_sequence`
*   `alt_forward_sequence`
*   `label`
*   `tissue`
*   `distance_to_nearest_tss`

That means it naturally supports progressively better models, from simple sequence-only baselines to more biologically informed models that include tissue-specific context.

### Loading the eQTL benchmark

```python
from datasets import load_dataset

eqtl_ds = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name="variant_effect_causal_eqtl",
    sequence_length=2048,
    trust_remote_code=True,
)
```

## A Strong Baseline: Frozen Embeddings + Delta Features

Before doing full fine-tuning, it is often useful to build a strong baseline with **frozen embeddings**.

### Key idea: compare the two alleles

For SNP prediction, the most informative feature is often the difference between the representation of the alternate allele and the representation of the reference allele:

$$
\delta = \mathrm{Emb}(\mathrm{alt}) - \mathrm{Emb}(\mathrm{ref})
$$

If the pretrained model produces very different embeddings for the two alleles, that is evidence that the mutation changes the learned sequence representation.

### Template code for the eQTL baseline

```python
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

@torch.no_grad()
def extract_embeddings(sequences, batch_size=16):
    all_embs = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.hidden_states[-1] * mask).sum(1) / mask.sum(1)
        all_embs.append(pooled.cpu())
    return torch.cat(all_embs, dim=0)

train_ds = eqtl_ds["train"]
test_ds = eqtl_ds["test"]

ref_train = [ex["ref_forward_sequence"] for ex in train_ds]
alt_train = [ex["alt_forward_sequence"] for ex in train_ds]
ref_test = [ex["ref_forward_sequence"] for ex in test_ds]
alt_test = [ex["alt_forward_sequence"] for ex in test_ds]

ref_emb_train = extract_embeddings(ref_train)
alt_emb_train = extract_embeddings(alt_train)
ref_emb_test = extract_embeddings(ref_test)
alt_emb_test = extract_embeddings(alt_test)

X_train = (alt_emb_train - ref_emb_train).numpy()
X_test = (alt_emb_test - ref_emb_test).numpy()
y_train = np.array([ex["label"] for ex in train_ds])
y_test = np.array([ex["label"] for ex in test_ds])

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)

test_probs = clf.predict_proba(X_test)[:, 1]
print("Test AUROC:", roc_auc_score(y_test, test_probs))
```

This baseline is attractive because it is:

*   easy to implement,
*   cheap to train,
*   interpretable, and
*   often surprisingly competitive.

## When to Move Beyond the Baseline

If you need higher performance, you can then move to stronger adaptation methods:

*   train a small **MLP** on top of delta embeddings,
*   fine-tune a **sequence classification head** end-to-end,
*   or use **parameter-efficient tuning** such as adapters or LoRA.

The important conceptual progression is:

1.  start with a frozen pretrained model,
2.  verify that the representation already contains signal for the task,
3.  then increase task-specific adaptation only if needed.

## Practical Fine-Tuning Considerations for SNPs

Regardless of the task, a few design choices matter repeatedly.

*   **Sequence centering**: keep the SNP near the middle of the input window so both sides are visible.
*   **Reference-vs-alternate comparison**: for variant effect tasks, explicitly comparing the two alleles is usually better than using only one allele.
*   **Reverse-complement augmentation**: DNA is double stranded, so augmentation can improve robustness.
*   **Tissue conditioning**: for eQTL tasks, tissue identity can be biologically essential.
*   **Evaluation split design**: matched negatives and tissue-aware validation matter for avoiding overoptimistic results.

In short, fine-tuning turns a general DNA language model into a task-specific predictor. For SNP biology, the central question is always the same: **does the representation learned during pretraining contain enough information to separate functional from non-functional variants?**
