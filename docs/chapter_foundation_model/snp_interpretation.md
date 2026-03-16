# Model Interpretation in Genomics

A major challenge with "Black Box" foundation models is trust. If a model predicts a SNP is pathogenic, a biologist needs to know *why*.

## Saliency Maps (Gradient-Based Attribution)

The most direct way to interpret a genomic model is to ask: "Which token positions contributed most to this prediction?"

### The Method: Gradients w.r.t. Input Embeddings

We compute the gradient of the model's output with respect to the **input embedding matrix**. The L2 norm of the gradient at each position gives a saliency score:

$$ \text{Saliency}(i) = \left\| \frac{\partial \hat{y}}{\partial \mathbf{e}_i} \right\|_2 $$

A high score at position $i$ means that small changes to that token's embedding would strongly affect the prediction.

### Template Code: Gradient Saliency

```python
import torch

model.eval()
sequence = test_ds[0]["ref_forward_sequence"]

# Tokenize the sequence
enc = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=512).to(device)

# Step 1: Get input embeddings and enable gradient tracking
input_embeds = model.get_input_embeddings()(enc["input_ids"])  # (1, seq_len, dim)
input_embeds.requires_grad_(True)
input_embeds.retain_grad()

# Step 2: Forward pass using embeddings instead of token IDs
out = model(inputs_embeds=input_embeds,
            attention_mask=enc["attention_mask"],
            output_hidden_states=True)

# Step 3: Define target — sum of logits at the SNP token position
# (or use the classification logit for fine-tuned models)
snp_pos = 50000  # example SNP position in original sequence
snp_token_idx = snp_pos // 6 + 1  # 6-mer stride, +1 for CLS token
snp_token_idx = min(snp_token_idx, out.logits.shape[1] - 1)

target = out.logits[0, snp_token_idx].sum()
target.backward()

# Step 4: Saliency = L2 norm of gradient at each token position
saliency = input_embeds.grad[0].norm(dim=-1).detach().cpu().numpy()
# saliency.shape = (seq_len,)
print(f"Saliency shape: {saliency.shape}")
print(f"Peak saliency at token: {saliency.argmax()}, SNP token: {snp_token_idx}")
```

### Template Code: Visualizing the Saliency Map

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(saliency, color="steelblue", linewidth=1.5, alpha=0.8)
plt.fill_between(range(len(saliency)), saliency, alpha=0.2, color="steelblue")
plt.axvline(snp_token_idx, color="red", linestyle="--", linewidth=2,
            label=f"SNP token ({snp_token_idx})")
plt.xlabel("Token Position")
plt.ylabel("Saliency (‖∇embedding‖₂)")
plt.title("Gradient Saliency Along Sequence")
plt.legend()
plt.tight_layout()
plt.show()
```

### Linking to Science: Motif Discovery

When we visualize these saliency maps, we often see peaks at known regulatory motifs:

*   **Peak at `TATAAA`** → The model learned the TATA box (core promoter element).
*   **Peak at `GATA`** → The model learned a GATA transcription factor binding site.
*   **Peak at the SNP position** → The model directly links the variant to its functional impact.

If a SNP disrupts a saliency peak, it suggests the variant is **breaking a transcription factor binding site**.

![Saliency Map Example](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41588-018-0295-5/MediaObjects/41588_2018_295_Fig1_HTML.png)
*(Visualization of importance scores. Source: DeepSEA, Nature Genetics 2015)*

!!! warning "Interpretation Caveats"
    - **Gradient saliency is local**: it measures sensitivity to infinitesimal changes at the current input, not the effect of the actual SNP mutation.
    - For more robust attribution, consider **Integrated Gradients** (averages gradients along a path from a baseline) or **In Silico Mutagenesis** (explicitly substitute each position and measure the change in output).

## Attention Analysis

For Transformer models, we can also inspect the raw **Attention Weights** to see which parts of the sequence "attend" to each other.

### Template Code: Extracting and Plotting Attention

```python
import seaborn as sns

model.eval()
with torch.no_grad():
    out = model(**enc, output_attentions=True)

# Get the last layer's attention, averaged over all heads
# Shape: (n_heads, seq_len, seq_len)
last_attn = out.attentions[-1].squeeze(0).cpu().numpy()
avg_attn = last_attn.mean(axis=0)  # (seq_len, seq_len)

# Plot a window of tokens around the SNP
window = 10
start = max(0, snp_token_idx - window)
end   = min(avg_attn.shape[0], snp_token_idx + window)
attn_window = avg_attn[start:end, start:end]

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(attn_window, ax=ax, cmap="viridis", square=True)

# Highlight the SNP token row/column
snp_rel = snp_token_idx - start
ax.axhline(snp_rel, color="red", linewidth=2)
ax.axvline(snp_rel, color="red", linewidth=2)
ax.set_title("Attention Heatmap Around SNP (Last Layer, Head-Averaged)")
ax.set_xlabel("Key Token")
ax.set_ylabel("Query Token")
plt.tight_layout()
plt.show()
```

*   **Biological Insight**: Strong attention between a promoter region and a distant site may indicate an **enhancer-promoter interaction**.
*   **Caution**: High attention weight does not always imply causal importance. Use it as a guide for generating hypotheses, not as definitive evidence.

## Pathway Analysis with Embeddings

We can also interpret the **embeddings** at a higher level by checking whether they cluster along known biological dimensions.

### Workflow

1.  **Extract embeddings** for a set of SNPs (using `extract_embeddings` from the previous section).
2.  **Reduce dimensions** using PCA or UMAP.
3.  **Color by biology**: cell type, chromatin state, gene pathway.

```python
from sklearn.decomposition import PCA

# delta = alt_embs - ref_embs  (each row is one SNP)
pca = PCA(n_components=2)
coords = pca.fit_transform(delta.numpy())

# Color by label (causal vs non-causal)
import matplotlib.pyplot as plt
for label, color, name in [(0, "steelblue", "Non-causal"), (1, "coral", "Causal")]:
    mask = y == label
    plt.scatter(coords[mask, 0], coords[mask, 1],
                c=color, alpha=0.4, s=10, label=name)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("PCA of Delta Embeddings (alt − ref)")
plt.legend()
plt.tight_layout()
plt.show()
```

If causal and non-causal SNPs **separate in PCA space**, the foundation model has learned something biologically meaningful about variant impact.

## References and Further Reading

*   **Nucleotide Transformer**: Dalla-Torre et al., [The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2), *bioRxiv* 2023.
*   **Enformer**: Avsec et al., [Effective gene expression prediction from sequence by integrating long-range interactions](https://www.nature.com/articles/s41592-021-01252-x), *Nature Methods* 2021.
*   **DeepSEA**: Zhou & Troyanskaya, [Predicting effects of noncoding variants with deep learning-based sequence model](https://www.nature.com/articles/nmeth.3547), *Nature Methods* 2015.
*   **scGPT**: Cui et al., [scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI](https://www.nature.com/articles/s41592-024-02201-0), *Nature Methods* 2024.
*   **Integrated Gradients**: Sundararajan et al., [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365), *ICML* 2017.
