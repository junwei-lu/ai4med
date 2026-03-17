# Fine-Tuning for SNP Prediction


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

## General Principles for Fine-Tuning 

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

### Template Code for Downstream Fine-Tuning

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

## Example: Fine-Tuning for eQTL Prediction

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

## LoRA as Tissue-Specific Adapters

The eQTL benchmark includes a `tissue` field because regulatory effects are not universal. A variant that disrupts a liver-specific enhancer may have no detectable effect in lung tissue. A single shared fine-tuned model learns an average signal but cannot capture this tissue heterogeneity.

**LoRA is a natural fit for the multi-tissue setting**: because LoRA adds only a small set of low-rank matrices on top of a frozen base, you can train a separate adapter per tissue with minimal overhead and swap between them at inference time.

### Why tissue-specific adapters?

A shared LoRA adapter trained on all tissues compresses their signal into one set of weights. Tissue-specific adapters instead learn:

*   tissue-specific regulatory grammar (e.g., liver enhancers vs. immune enhancers)
*   tissue-specific chromatin accessibility patterns
*   variant effects present in one tissue but absent in another

Because all adapters share the same frozen base model, memory cost is fixed and the total parameter overhead grows linearly in the number of tissues rather than multiplicatively.

### LoRA recap

Instead of updating all weights $W_0 \in \mathbb{R}^{d \times k}$, LoRA injects a low-rank perturbation into each target layer:

$$W = W_0 + \frac{\alpha}{r} B A$$

where $A \in \mathbb{R}^{r \times k}$, $B \in \mathbb{R}^{d \times r}$, and $r \ll \min(d, k)$.

*   $B$ is initialized to zero so that training starts from the pretrained weights
*   $A$ is initialized from a small Gaussian
*   Only $A$ and $B$ are trained — typically less than 1% of total parameters

### Setting up the LoRA base model

```python
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"

base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    trust_remote_code=True,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["query", "key", "value"],  # attention projection layers
    task_type=TaskType.SEQ_CLS,
)

peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()
# → trainable params: ~200K | all params: ~50M | trainable: ~0.4%
```

### Training one adapter per tissue

The workflow is:

1.  group the dataset by tissue,
2.  for each tissue, initialize a fresh LoRA wrapper around the same frozen base,
3.  train on that tissue's examples,
4.  save only the LoRA weights to a tissue-specific directory.

```python
import os, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
tokenizer  = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
device     = "cuda" if torch.cuda.is_available() else "cpu"

eqtl_ds = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name="variant_effect_causal_eqtl",
    sequence_length=2048,
    trust_remote_code=True,
)

SEP = tokenizer.eos_token or tokenizer.pad_token

def tokenize_for_classification(examples):
    texts = [
        ref + SEP + alt
        for ref, alt in zip(
            examples["ref_forward_sequence"], examples["alt_forward_sequence"]
        )
    ]
    encoded = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
    encoded["labels"] = examples["label"]
    return encoded

tissues = list(set(eqtl_ds["train"]["tissue"]))

lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1,
    bias="none", target_modules=["query", "key", "value"],
    task_type=TaskType.SEQ_CLS,
)

for tissue in tissues:
    print(f"\n── Training LoRA adapter for: {tissue} ──")
    adapter_dir = f"adapters/{tissue.replace(' ', '_')}"

    train_tissue = eqtl_ds["train"].filter(lambda ex: ex["tissue"] == tissue)
    test_tissue  = eqtl_ds["test"].filter(lambda ex: ex["tissue"] == tissue)

    if len(train_tissue) < 50:
        print(f"  Skipping {tissue} — too few examples ({len(train_tissue)})")
        continue

    tok_train = train_tissue.map(tokenize_for_classification, batched=True,
                                 remove_columns=train_tissue.column_names)
    tok_test  = test_tissue.map(tokenize_for_classification, batched=True,
                                remove_columns=test_tissue.column_names)
    tok_train.set_format("torch")
    tok_test.set_format("torch")

    # Fresh base + new LoRA adapter for this tissue
    base_clf  = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, trust_remote_code=True,
    ).to(device)
    model_lora = get_peft_model(base_clf, lora_config)

    training_args = TrainingArguments(
        output_dir=adapter_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    Trainer(
        model=model_lora,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_test,
    ).train()

    # Save only the adapter weights — the base model is shared
    model_lora.save_pretrained(adapter_dir)
    print(f"  Saved adapter → {adapter_dir}")
```

### Loading and switching adapters at inference

Once saved, any tissue-specific adapter can be reloaded without re-training or storing the full model multiple times.

```python
from peft import PeftModel

def load_tissue_adapter(tissue: str, model_name: str, device: str):
    adapter_dir = f"adapters/{tissue.replace(' ', '_')}"
    base = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, trust_remote_code=True,
    ).to(device)
    return PeftModel.from_pretrained(base, adapter_dir).to(device).eval()

# Swap to the Whole Blood adapter at inference time
tissue_model = load_tissue_adapter("Whole_Blood", model_name, device)
```

### Visualizing tissue-adapter embedding separation

A useful diagnostic is to extract LoRA delta embeddings per tissue and visualize how well each adapter separates causal from non-causal variants.

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

@torch.no_grad()
def get_delta_embeddings(model, seqs_ref, seqs_alt, tokenizer, device, batch_size=16):
    """Return (alt − ref) mean-pooled hidden states from the last encoder layer."""
    model.eval()
    base = model.base_model.model
    ref_embs, alt_embs = [], []
    for seqs, store in [(seqs_ref, ref_embs), (seqs_alt, alt_embs)]:
        for i in range(0, len(seqs), batch_size):
            enc = tokenizer(
                seqs[i : i + batch_size],
                return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out  = base(**enc, output_hidden_states=True)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.hidden_states[-1] * mask).sum(1) / mask.sum(1)
            store.append(pooled.cpu())
    return (torch.cat(alt_embs) - torch.cat(ref_embs)).numpy()

results = {}
for tissue in tissues:
    adapter_dir = f"adapters/{tissue.replace(' ', '_')}"
    if not os.path.exists(adapter_dir):
        continue

    tissue_model = load_tissue_adapter(tissue, model_name, device)
    test_tissue  = eqtl_ds["test"].filter(lambda ex: ex["tissue"] == tissue)
    refs   = [ex["ref_forward_sequence"] for ex in test_tissue]
    alts   = [ex["alt_forward_sequence"]  for ex in test_tissue]
    labels = np.array([ex["label"] for ex in test_tissue])

    delta = get_delta_embeddings(tissue_model, refs, alts, tokenizer, device)
    results[tissue] = {"delta": delta, "labels": labels}

# Side-by-side PCA for the first two tissues
fig, axes = plt.subplots(1, min(2, len(results)), figsize=(12, 5))
for ax, tissue in zip(axes, list(results.keys())[:2]):
    delta  = results[tissue]["delta"]
    labels = results[tissue]["labels"]
    pca    = PCA(n_components=2, random_state=42).fit_transform(delta)
    for label, color, name in [(0, "steelblue", "Non-causal"), (1, "coral", "Causal")]:
        mask = labels == label
        ax.scatter(pca[mask, 0], pca[mask, 1], c=color, alpha=0.4, s=15, label=name)
    ax.set_title(f"LoRA Delta Embeddings\n{tissue}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend()
plt.suptitle("Tissue-Specific LoRA: Embedding Separation", fontsize=13)
plt.tight_layout()
plt.show()
```

### Design considerations

*   **Minimum tissue sample size**: tissues with fewer than ~100 training examples typically do not benefit from tissue-specific adapters; a shared adapter generalizes better in that data-scarce regime.
*   **Rank choice ($r$)**: lower rank (4–8) works well when tissue-specific data is limited; higher rank (16–32) may be justified for data-rich tissues.
*   **Shared vs. per-tissue classification head**: the LoRA adapter modifies the encoder; the final classification head can also be made tissue-specific or kept shared.
*   **Multi-task learning as an alternative**: instead of separate adapters, tissues can be encoded as additional input tokens injected at each layer — a richer but more complex approach.
*   **Inference cost**: loading a different adapter requires only a dictionary copy of ~200 K parameters, so switching tissues at inference time is fast.

Tissue-specific LoRA balances biological specificity against computational simplicity. Each adapter is small (< 1 MB), swappable in milliseconds, and trained on tissue-local data — making it a practical choice for multi-tissue regulatory genomics workflows.
