# Plan: SNP/Genomics Workshop Jupyter Notebook (Day 2)

## Context

The instructor needs a single comprehensive Jupyter notebook for a 3-hour workshop on SNP effect prediction using genomic foundation models. This is Day 2 of a bootcamp, mirroring the Day 1 NLP notebook (`nlp/enar_clinical_nlp_workshop.ipynb`). The same intellectual spine — tokenization, embeddings, attention, pretraining, transfer learning — is applied to genomics. Source of truth: `snp/instructions.md`.

## File to Create

`/Users/junweilu/Dropbox/Teach/AI_bootcamp/ai4med/codes/snp/snp_genomics_workshop.ipynb`

## Key Existing Patterns to Reuse

- `pip_install()` helper, GPU detection, `USE_PRECOMPUTED` flag from NLP notebook
- `plot_training_curve()`, `display_comparison_table()` helpers from NLP notebook
- matplotlib + seaborn styling (no plotly), `report_to="none"`, `save_strategy="no"`
- LoRA/PEFT patterns from `nlp/peft.md`

---

## Notebook Structure (7 sections, ~65 cells, ~23 visualizations)

### Section 0: Setup and Environment (~4 cells)

- **Markdown:** Title ("From Language Models to Genome Models: SNP Effect Prediction with Foundation Models"), TOC with time estimates, Colab GPU instructions
- **Code:** `pip_install([transformers, datasets, tokenizers, peft, accelerate, scikit-learn, ...])`, imports, GPU detect, `USE_PRECOMPUTED = False`, seeds, matplotlib defaults
- **Code:** Helper functions: `plot_training_curve()`, `display_comparison_table()`, `extract_embeddings(model, tokenizer, sequences, device, batch_size)` (new — mean-pooled hidden states), `find_snp_position(ref, alt)` (new)
- **Markdown:** Day 1 → Day 2 concept mapping table (text tokenization→DNA tokenization, decoder→encoder, SFT/GRPO→frozen probe + LoRA). "Why SNPs are different" bullet points.

---

### Section 1: Data Tour — The eQTL SNP Task (0:00–0:20, ~8 cells)

- **Markdown:** What is an eQTL? Binary classification: does this variant causally affect gene expression in a given tissue? Text diagram: `DNA: ...ACGT[A→G]CGTACGT... → label 1 or 0`
- **Code:** Load dataset: `load_dataset("InstaDeepAI/genomics-long-range-benchmark", "variant_effect_causal_eqtl", sequence_length=512)`. Classroom subset: 4000 train, 800 test.
- **Code:** Inspect one example (HTML table), find & verify SNP position with surrounding ±10 bases
- **Viz 1:** Histogram of SNP position within window → teaching: SNP is centered
- **Viz 2:** 2-panel: label balance bar chart + top-15 tissues bar chart
- **Viz 3:** Overlapping histograms of distance-to-TSS by label (log scale)
- **Markdown:** Chromosome-held-out evaluation — why random splits are dangerous in genomics (linkage disequilibrium)

---

### Section 2: DNA Tokenization and Genomic Foundation Models (0:20–0:45, ~10 cells)

- **Markdown:** Two representation styles: (A) Genomic language models (k-mer tokenization + MLM), (B) Sequence-to-function (one-hot DNA). Math: MLM objective vs CLM from Day 1
- **Code:** Load Nucleotide Transformer tokenizer, print vocab size, special tokens
- **Code:** Tokenize one ref and one alt sequence step-by-step: raw DNA → 6-mer tokens → token IDs
- **Viz 4:** Color-coded HTML of ref vs alt tokenizations, highlight differing k-mers in red
- **Markdown:** K-mer tokenization explained (non-overlapping 6-mers, vocab = 4^6 + specials)
- **Viz 5:** One-hot DNA encoding heatmap (4 rows × 20 cols) — the alternative representation
- **Viz 6:** Top-40 most frequent 6-mer tokens (horizontal bar chart), colored by GC content
- **Code:** Token IDs → embeddings via pretrained model, print shapes
- **Viz 7:** K-mer embedding cosine similarity heatmap (12-15 selected k-mers)
- **Markdown:** Masked Language Modeling objective for DNA, contrast with Day 1 CLM

---

### Section 3: Frozen Foundation-Model Embeddings for SNP Prediction (0:45–1:30, ~13 cells)

> This is the main lab.

- **Markdown:** Transfer learning recipe: pretrained (frozen) → embeddings → simple classifier. Text diagram:

```
ref ──→ [Frozen NT] ──→ mean-pool ──→ emb_ref ──┐
                                                  ├→ [ref; alt; delta; tissue; TSS] → Classifier → P(causal)
alt ──→ [Frozen NT] ──→ mean-pool ──→ emb_alt ──┘
```

- **Code:** Load frozen Nucleotide Transformer (50M), print param count
- **Viz 8:** Parameter distribution bar chart (embeddings, encoder layers, MLM head)
- **Code:** Extract embeddings for all train/test ref and alt sequences (gated by `USE_PRECOMPUTED`)
- **Code:** Compute delta embeddings (alt − ref), build feature matrix: `[ref_emb; alt_emb; delta; tissue_onehot; log(TSS)]`
- **Viz 9:** Delta embedding ‖δ‖₂ distribution histogram, colored by label
- **Viz 10:** PCA scatter of delta embeddings, colored by label
- **Code:** Logistic regression baseline → AUROC, AUPRC
- **Code:** Tiny MLP classifier (256→64→1), train ~50 epochs with BCE + pos_weight
- **Viz 11:** MLP training loss curve
- **Viz 12:** 2-panel: ROC curves (LR vs MLP) + PR curves
- **Viz 13:** Confusion matrix heatmap
- **Markdown:** Summary table of frozen embedding results

---

### Section 4: Lightweight Fine-Tuning with LoRA (1:30–2:00, ~10 cells)

- **Markdown:** Frozen → fine-tuned. LoRA math: W = W₀ + (α/r)BA. Reference Day 1 SFT.
- **Code:** Load `AutoModelForSequenceClassification` + LoRA config (r=8, α=16, target attention layers)
- **Viz 14:** Full vs LoRA parameter count bar chart
- **Code:** Tokenize dataset for Trainer (ref+alt as paired input), reduced subset (~2000 train)
- **Code:** `TrainingArguments` + `Trainer`, train or load checkpoint
- **Viz 15:** LoRA fine-tuning loss curve
- **Code:** Evaluate LoRA model → AUROC, AUPRC
- **Viz 16:** Grouped bar chart: LR-frozen vs MLP-frozen vs LoRA (AUROC, AUPRC)
- **Code:** Extract LoRA embeddings for comparison
- **Viz 17:** 2-panel PCA: frozen vs LoRA delta embeddings

---

### Section 5: Demo — Enformer and DeepSEA (2:00–2:30, ~9 cells)

> Instructor-led with precomputed results.

- **Markdown:** Sequence-to-function models. DeepSEA (2k bp CNN) vs Enformer (393k bp transformer).
- **Markdown:** Enformer variant effect scoring: `VES(v) = f(seq_alt) − f(seq_ref)`
- **Code:** Load precomputed Enformer results for ~5 example variants
- **Viz 18:** Enformer VES heatmap across tissues (variants × tissues)
- **Markdown:** DeepSEA as short-context counterpart
- **Viz 19:** Architecture comparison diagram (matplotlib): DeepSEA vs NT vs Enformer (input size, type, output)
- **Viz 20:** Cross-model variant scoring comparison (HTML table or grouped bars)
- **Markdown:** When to use which model (summary table)

---

### Section 6: Interpretation and Attribution (2:30–2:45, ~6 cells)

- **Markdown:** Attention visualization, gradient saliency, in silico mutagenesis
- **Code:** Run NT with `output_attentions=True`, extract attention around SNP site
- **Viz 21:** Attention heatmap around SNP position (last layer, ~20 token window)
- **Code:** Gradient-based saliency on input embeddings
- **Viz 22:** Saliency scores along sequence (line plot), vertical line at SNP
- **Markdown:** Interpretation caveats

---

### Section 7: Wrap-Up (2:45–3:00, ~5 cells)

- **Markdown:** Journey summary (7 steps from data to interpretation)
- **Code:** Final results DataFrame (model, AUROC, AUPRC, trainable params)
- **Viz 23:** Final grouped bar chart (3 models × metrics)
- **Markdown:** Day 1 vs Day 2 comparison table
- **Markdown:** Takeaways and further exploration pointers

---

## Visualization Summary (23 total)

| # | Type | Section | Purpose |
|---|------|---------|---------|
| 1 | Histogram | Data | SNP position within window |
| 2 | 2-panel bars | Data | Label balance + tissue distribution |
| 3 | Overlapping histograms | Data | Distance-to-TSS by label |
| 4 | Color-coded HTML | Tokenization | Ref vs alt k-mer tokenization |
| 5 | Heatmap (4×N) | Tokenization | One-hot DNA encoding |
| 6 | Horizontal bar chart | Tokenization | Top-40 k-mer frequencies |
| 7 | Cosine similarity heatmap | Tokenization | K-mer embedding similarity |
| 8 | Bar chart | Frozen Emb | Model parameter distribution |
| 9 | Colored histogram | Frozen Emb | Delta embedding magnitude |
| 10 | PCA scatter | Frozen Emb | Delta embedding space |
| 11 | Line plot | Frozen Emb | MLP training loss |
| 12 | 2-panel ROC+PR | Frozen Emb | LR vs MLP evaluation |
| 13 | Heatmap | Frozen Emb | Confusion matrix |
| 14 | Bar chart | LoRA | Full vs LoRA params |
| 15 | Line plot | LoRA | Fine-tuning loss |
| 16 | Grouped bars | LoRA | Frozen vs LoRA metrics |
| 17 | 2-panel PCA | LoRA | Embedding space before/after |
| 18 | Heatmap | Demo | Enformer VES across tissues |
| 19 | Diagram | Demo | Architecture comparison |
| 20 | Table/bars | Demo | Cross-model variant scores |
| 21 | Heatmap | Interpretation | Attention around SNP |
| 22 | Line plot | Interpretation | Saliency along sequence |
| 23 | Grouped bars | Wrap-up | Final model comparison |

---

## Design Decisions

1. **Single notebook (not 4)** — matches NLP Day 1 pattern, keeps storyline cohesive
2. **Small subsets (4000 train, 800 test, 512 bp)** — embedding extraction completes in ~15 min on T4
3. **`USE_PRECOMPUTED` flag** — gates embedding extraction and all training steps
4. **Frozen first, then LoRA** — progressive difficulty matches `instructions.md`
5. **Enformer/DeepSEA via precomputed results** — too large for T4 live inference; ship JSON files
6. **`AutoModelForMaskedLM` for frozen, `AutoModelForSequenceClassification` for LoRA** — frozen extracts hidden states; LoRA uses HF Trainer with classification loss
7. **LogReg AND MLP** — interpretable baseline + nonlinear improvement
8. **matplotlib + seaborn only** — Colab compatibility, matching NLP notebook
9. **Interpretation section is brief (2 visualizations, 15 min)** — debugging tools, not main focus

---

## Precomputed Files to Ship

1. `snp_embeddings.pt` — ref/alt train/test embeddings (each N × 512)
2. `mlp_classifier.pt` — trained MLP state dict
3. `nt_lora_eqtl/` — LoRA adapter checkpoint
4. `enformer_demo_results.json` — VES for 5 variants × 20 tissues
5. `deepsea_demo_results.json` — VES for same 5 variants

---

## Critical Files

- `nlp/enar_clinical_nlp_workshop.ipynb` — reference for all structural patterns
- `snp/instructions.md` — source of truth for content and timing
- `nlp/peft.md` — LoRA math and config to reuse in Section 4

---

## Verification Plan

1. Run end-to-end on Colab T4 with `USE_PRECOMPUTED=False`
2. Re-run with `USE_PRECOMPUTED=True` (checkpoint loading path)
3. Check all 23 visualizations render correctly
4. Verify dataset loads with `sequence_length=512`, `trust_remote_code=True`
5. Verify NT tokenizer produces 6-mer tokens
6. Confirm LoRA applies correctly (`print_trainable_parameters()`)
7. Time each section against schedule
8. Test on CPU with reduced steps to verify no crashes
