# Fine-Tuning for SNP Prediction

We will focus on predicting **Expression Quantitative Trait Loci (eQTLs)**. An eQTL is a genomic variant (SNP) that affects the expression level of a gene.

**The Task**: Given a DNA sequence containing a SNP, predict whether it is a "causal" eQTL (affects expression) or "non-causal" (no effect).

## Setup and Data

We will use the **Nucleotide Transformer** (by InstaDeep) and the GTEx eQTL benchmark.

```bash
pip install transformers peft datasets torch scikit-learn
```

## Strategy A: Feature Extraction (Frozen Model)

Instead of updating the model's weights, we use the foundation model as a **feature extractor**. We pass the DNA sequences through the frozen model and extract the final hidden-layer embeddings. These embeddings capture the "meaning" of each sequence in the context of the entire genome.

### Key Idea: The Delta Embedding

For SNP prediction, the most informative feature is the **difference** between the reference and alternate embeddings:

$$ \delta = \text{Emb}(\text{alt}) - \text{Emb}(\text{ref}) $$

A large $\|\delta\|$ means the model "sees" a big difference between the two sequences—suggesting a functional impact.

### Template Code: Extracting Embeddings and Building Features

```python
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# --- Load frozen model ---
model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

# --- Embedding extraction (mean-pool last hidden state) ---
@torch.no_grad()
def extract_embeddings(sequences, batch_size=16):
    all_embs = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        # hidden_states[-1]: (B, seq_len, dim)  →  mean-pool over seq_len
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.hidden_states[-1] * mask).sum(1) / mask.sum(1)
        all_embs.append(pooled.cpu())
    return torch.cat(all_embs, dim=0)

# --- Extract ref and alt embeddings ---
ref_seqs_train = [ex["ref_forward_sequence"] for ex in train_ds]
alt_seqs_train = [ex["alt_forward_sequence"] for ex in train_ds]

ref_embs = extract_embeddings(ref_seqs_train)
alt_embs = extract_embeddings(alt_seqs_train)

# --- Build delta feature matrix ---
delta = alt_embs - ref_embs  # (N, 512): captures the SNP's effect

# Optionally concatenate extra features (tissue, distance to TSS)
y_train = np.array([ex["label"] for ex in train_ds])

# --- Train simple classifier on delta embeddings ---
clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(delta.numpy(), y_train)

auroc = roc_auc_score(y_test, clf.predict_proba(delta_test.numpy())[:, 1])
print(f"Logistic Regression AUROC: {auroc:.4f}")
```

### Optional: MLP Classifier

For higher capacity, replace `LogisticRegression` with a small neural network:

```python
import torch.nn as nn

class SNPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

# Train with BCEWithLogitsLoss
input_dim = delta.shape[1]
mlp = SNPClassifier(input_dim).to(device)
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

X = torch.tensor(delta.numpy(), dtype=torch.float32, device=device)
y = torch.tensor(y_train, dtype=torch.float32, device=device)

for epoch in range(50):
    mlp.train()
    logits = mlp(X).squeeze(-1)
    loss = criterion(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Strategy B: Parameter-Efficient Fine-Tuning (LoRA)

For better performance, we can **fine-tune** the foundation model itself. Full fine-tuning is expensive, so we use **LoRA (Low-Rank Adaptation)**.

### Why LoRA for Genomics?

*   **Modularity**: You can have one base model and separate LoRA adapters for "Liver eQTLs", "Heart eQTLs", "Brain eQTLs", etc.
*   **Efficiency**: Updates < 1% of parameters, allowing training on consumer GPUs.

### Implementation

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# Load model with a classification head
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,   # Causal (1) vs Non-causal (0)
    trust_remote_code=True,
)

# Inject LoRA into attention layers
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                                         # low-rank dimension
    lora_alpha=16,                               # scaling factor
    lora_dropout=0.1,
    target_modules=["query", "key", "value"],    # attention projections
)
lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
# trainable params: ~200K out of 50M  (<0.5%)

# Tokenize dataset for the Trainer
def tokenize(example):
    return tokenizer(
        example["ref_forward_sequence"],
        truncation=True, max_length=512, padding="max_length"
    )

train_tok = train_ds.map(tokenize, batched=True)
train_tok = train_tok.rename_column("label", "labels")

# Train
training_args = TrainingArguments(
    output_dir="lora_eqtl",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-4,
    logging_steps=50,
)
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_tok,
)
trainer.train()
```

## Fine-Tuning Considerations for SNPs

*   **Sequence Centering**: Ensure the SNP is in the center of the input window so the model has equal context on both sides.
*   **Reverse Complement Augmentation**: DNA is double-stranded. Augmenting with reverse complements often improves generalization.
*   **Tissue as Condition**: Passing tissue type (e.g., "Liver") as an additional input feature significantly improves eQTL prediction accuracy.
