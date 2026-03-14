# Plan: ENAR Short Course Jupyter Notebook

## Context

The instructor needs a single comprehensive Jupyter notebook for a 3-hour ENAR short course on clinical NLP and LLM fine-tuning. The course follows a coherent storyline: build a tiny clinical GPT on one dataset (AGBonnet/augmented-clinical-notes), then post-train it twice (SFT, GRPO). The notebook must interleave conceptual markdown explanations with runnable code cells and rich visualizations to help students understand each stage.

**Source of truth:** `instructions.md` in the repo root.

### Key existing patterns to reuse

- **pip_install** helper and GPU detection from `materials/LLM-FT/grpo_colab.ipynb` (cell-1, cell-3)
- **inspect.signature** trick for SFTTrainer tokenizer vs processing_class from `materials/LLM-FT/llm_ft_basics_colab.ipynb` (cell-11)
- Math notation for attention/transformer from `docs/chapter_language_model/transformer.md`
- Tokenization/embedding explanations from `docs/chapter_language_model/wordvec.md`



## File to Create

`/Users/junweilu/Dropbox/Teach/AI_bootcamp/ai4med/codes/enar_shortcourse/enar_clinical_nlp_workshop.ipynb`



## Notebook Structure (7 sections, ~70 cells)

### Section 0: Setup and Environment (~3 cells)

**Markdown cell:** Title, abstract, table of contents with time estimates, Colab GPU instructions.

**Code cell — Install & imports:**

```python
pip_install(["torch", "transformers", "datasets", "tokenizers", "trl", "peft",
             "accelerate", "matplotlib", "seaborn", "numpy", "pandas",
             "scikit-learn", "wordcloud"])
```

- Print versions, detect GPU, set device
- Set `USE_PRECOMPUTED = False` flag (toggle for live vs checkpoint mode)
- Set `CHECKPOINT_DIR` path (HF Hub or local)
- Set matplotlib defaults for lecture readability, random seeds

**Code cell — Helper functions:**

- `generate_text(model, tokenizer, prompt, max_new_tokens=128)` — generation wrapper
- `plot_training_curve(log_history, metric, title)` — training curve plotter
- `display_comparison_table(...)` — HTML side-by-side renderer



### Section 1: Data Tour and Framing (0:00–0:20, ~7 cells)

**Markdown:** The clinical NLP challenge — EHRs are unstructured text; our pipeline: raw text → tokenize → pretrain → SFT → GRPO. Include a text pipeline diagram.

**Code:** Load dataset `load_dataset("AGBonnet/augmented-clinical-notes", split="train[:3000]")`, print info.

**Code:** Create train/val/test splits via `train_test_split`.

**Code:** Display one example record showing note, full_note, summary, conversation in formatted HTML.

**Visualization 1 — Text length distributions:**

- 3-panel histogram (note, full_note, summary character lengths)
- Matplotlib `plt.hist`, vertical lines for mean
- Teaching point: note fits in 256-token context, full_note doesn't

**Visualization 2 — Word cloud of clinical vocabulary:**

- Word cloud from note texts using wordcloud library
- Teaching point: clinical text has distinctive vocabulary

**Visualization 3 — Summary JSON structure:**

- Parse summaries as JSON, count key frequencies
- Horizontal bar chart of top-level keys
- Teaching point: consistent JSON schema is our target

**Markdown:** Framing recap — train on note, target is summary JSON. Data is synthetic (teaching pipeline, not clinical deployment).



### Section 2: Tokenization, Token IDs, Embeddings (0:20–0:45, ~9 cells)

**Markdown:** What is tokenization? Tokens vs words vs characters. Word-level, character-level, subword (BPE). Reference wordvec.md explanations. Why domain tokenizers matter.

**Code:** Load GPT-2's pretrained tokenizer, tokenize a clinical sentence ("The patient presented with acute myocardial infarction and was prescribed metoprolol 50mg twice daily."), show tokens and IDs.

**Code:** Train domain BPE tokenizer — exact code from instructions.md lines 56–73:

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
tok = Tokenizer(BPE(unk_token="[UNK]"))
tok.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=8000, special_tokens=["[UNK]","[PAD]","[BOS]","[EOS]"])
tok.train_from_iterator(corpus(), trainer=trainer, length=2*len(ds))
```

Then wrap with `PreTrainedTokenizerFast`.

**Visualization 4 — Side-by-side tokenization comparison:**

- Same clinical sentence tokenized by GPT-2 vs domain tokenizer
- Color-coded HTML spans (each token different background color)
- Bar below showing token count comparison
- Teaching point: domain tokenizer = fewer tokens for clinical terms

**Visualization 5 — Token frequency distribution:**

- Tokenize all training notes, plot top-50 most frequent tokens as horizontal bar chart
- Highlight clinical vs general terms with different colors

**Markdown:** From tokens to IDs to embeddings. `nn.Embedding(vocab_size, n_embd)` — lookup table concept. Position embeddings. Reference wordvec.md lines 30–50 and 149–230.

**Code:** Demonstrate embedding lookup — create `nn.Embedding(8000, 256)`, tokenize a phrase, show shapes.

**Visualization 6 — Embedding similarity heatmap (before training):**

- Pick 12–15 clinical tokens (patient, diagnosis, heart, blood, mg, pain, fever, etc.)
- Cosine similarity matrix → seaborn heatmap
- Title: "Token Embedding Similarity (Random Init)"
- Teaching point: random before training; will be meaningful after pretraining



### Section 3: Attention and the Tiny Transformer (0:45–1:15, ~8 cells)

**Markdown:** The attention mechanism — Q, K, V projections, attention formula $\text{softmax}(QK^T/\sqrt{d_k})V$, multi-head attention, causal masking for decoders. Use LaTeX from transformer.md lines 13–27 and 144–158.

**Markdown:** Transformer block diagram — Multi-Head Attention → Add & Norm → FFN → Add & Norm. Text architecture diagram of our tiny GPT:

```
Input IDs → Token Emb + Position Emb
     ↓
[Transformer Block × 4]
  - Masked Multi-Head Attention (4 heads)
  - LayerNorm + Residual
  - FFN (256 → 1024 → 256)
  - LayerNorm + Residual
     ↓
LM Head → logits over vocab (8000)
```

**Code:** Build tiny GPT — exact config from instructions.md lines 79–84:

```python
config = GPT2Config(vocab_size=len(tokenizer), n_positions=256,
                    n_embd=256, n_layer=4, n_head=4)
model = GPT2LMHeadModel(config)
```

Print total parameters.

**Visualization 7 — Parameter distribution:**

- Count params per component (embeddings, each transformer layer, LM head)
- Stacked/grouped bar chart
- Teaching point: even tiny model — embedding table and FFN dominate

**Code:** Forward pass with `output_attentions=True` on a clinical sentence, print attention tensor shapes.

**Visualization 8 — Attention heatmap (before training, single head):**

- Layer 0, Head 0 → seaborn heatmap with token labels on axes
- Teaching point: causal mask visible (upper triangle zeros), attention roughly uniform

**Visualization 9 — All 4 heads of layer 0 (2×2 grid):**

- 4-panel seaborn heatmap subplot
- Teaching point: different heads will learn different patterns after training

**Code:** Generate from untrained model → show random gibberish. "The model has architecture but no knowledge."



### Section 4: Pretraining by Next-Token Prediction (1:15–1:45, ~9 cells)

**Markdown:** What is pretraining? Next-token prediction, cross-entropy loss, causal masking. The model sees [t1,...,tn] and predicts [t2,...,tn+1].

**Code:** Prepare pretraining dataset — tokenize all note texts, concatenate, chunk into blocks of 256 tokens. Use `DataCollatorForLanguageModeling(mlm=False)`.

**Code:** Configure Trainer with TrainingArguments:

- `max_steps=500`, `batch_size=8`, `lr=5e-4`, `warmup_steps=100`
- `logging_steps=10`, `fp16=True`, `save_strategy="no"`, `report_to="none"`

**Code:** Train or load checkpoint:

```python
if not USE_PRECOMPUTED:
    trainer.train()
    model.save_pretrained("pretrained_clinical_gpt")
else:
    model = GPT2LMHeadModel.from_pretrained(CHECKPOINT_DIR + "/pretrained")
```

**Visualization 10 — Pretraining loss curve:**

- Line plot: steps vs loss
- Teaching point: rapid initial drop, then plateau

**Code:** Generate from pretrained model on 3 clinical prompts. Show outputs.

**Visualization 11 — Before vs after pretraining:**

- HTML table: same prompt → random model output vs pretrained model output
- Teaching point: model now produces clinical-sounding text

**Visualization 12 — Attention heatmap after pretraining:**

- Same clinical sentence, layer 0 head 0
- Seaborn heatmap
- Teaching point: attention patterns now structured (some heads attend to nearby tokens, others to specific terms)

**Visualization 13 — Embedding space after pretraining (PCA):**

- Extract embeddings for 50–100 clinical tokens
- PCA to 2D, scatter plot with text labels, color-coded by category (medications=blue, diagnoses=red, body parts=green, lab terms=orange)
- Teaching point: clinically related terms cluster together



### Section 5: SFT on Structured Extraction (1:45–2:20, ~10 cells)

**Markdown:** Pretraining = "continue text"; SFT = "follow instruction, produce specific output." Prompt-completion format. TRL's SFTTrainer computes loss on completion only.

**Code:** Prepare prompt-completion dataset — exact `to_sft` function from instructions.md lines 103–108:

```python
def to_sft(ex):
    prompt = ("Convert the following clinical note into a structured JSON medical record.\n\n"
              f"Clinical note:\n{ex['note']}\n\nJSON:\n")
    return {"prompt": prompt, "completion": ex["summary"]}
```

**Visualization 14 — Prompt/completion length distribution:**

- 2-panel histogram (prompt lengths, completion lengths in tokens)
- Vertical line at 256-token limit

**Code:** Configure SFTTrainer — load pretrained model, set `max_seq_length=256`, `max_steps=300`, `batch_size=4`, `lr=2e-4`. Use inspect.signature pattern for tokenizer/processing_class compatibility.

**Code:** Train or load SFT checkpoint.

**Visualization 15 — SFT training loss curve:**

- Line plot, optionally overlay with pretraining loss

**Code:** Generate JSON outputs on 5 test examples, display side-by-side with gold.

**Code:** Evaluation functions:

- `is_valid_json(text)` — try `json.loads`
- Schema compliance — check top-level keys against gold
- Compute metrics across test set

**Visualization 16 — JSON validity & schema compliance:**

- Grouped bar chart: pretrained (baseline) vs SFT model
- Metrics: JSON validity %, key coverage %
- Teaching point: SFT dramatically improves structured output

**Visualization 17 — Before/after comparison table:**

- 3 test examples: pretrained output (rambling) vs SFT output (structured JSON)
- Color-coded HTML table



### Section 6: GRPO with Rule-Based Rewards (2:20–2:45, ~9 cells)

**Markdown:** What is GRPO? Sample multiple completions per prompt, score with reward functions, update policy toward higher-reward completions. No reward model needed. `beta=0.0` skips reference model (saves memory).

**Code:** Prepare GRPO dataset — exact `to_grpo` function from instructions.md lines 127–133. Use smaller subset (~500 examples) for speed.

**Code:** Define three reward functions:

1. `json_validity_reward(completions, **kwargs)` → 1.0 if json.loads succeeds, else 0.0
2. `schema_reward(completions, **kwargs)` → fraction of required keys present
3. `reference_overlap_reward(completions, gold_summary, **kwargs)` → Jaccard-like overlap of parsed key-value pairs

**Visualization 18 — Reward function demo:**

- 4 example completions (valid+complete, valid+partial, invalid JSON, random text)
- Table or grouped bar chart showing all 3 rewards for each
- Teaching point: rewards are interpretable and deterministic

**Code:** Configure GRPOTrainer:

- `num_generations=2`, `batch_size=1`, `grad_accum=4`, `max_steps=100`
- `beta=0.0`, `max_prompt_length=200`, `max_completion_length=128`

**Code:** Train or load GRPO checkpoint.

**Visualization 19 — Reward curves during GRPO training:**

- Multi-line plot: steps vs mean reward for each component
- Teaching point: rewards improve over training

**Code:** Generate from GRPO model on same 5 test examples as SFT section.



### Section 7: Compare the Three Models (2:45–3:00, ~6 cells)

**Markdown:** Grand comparison — three checkpoints from same architecture: pretrained (clinical language), SFT (instruction following), GRPO (reward optimization).

**Code:** Generate from all three models on 3 held-out test notes.

**Visualization 20 — Side-by-side output comparison:**

- Rich HTML table: 4 columns (Clinical Note, Pretrained, SFT, GRPO)
- Color-coded: pretrained=gray, SFT=blue, GRPO=green

**Code:** Quantitative evaluation across full test set — JSON validity %, schema compliance %, reference overlap for each model. Store in pandas DataFrame.

**Visualization 21 — Final grouped bar chart:**

- 3 metric groups × 3 model bars
- Colors: gray=pretrained, blue=SFT, green=GRPO
- Teaching point: clear progression pretrained → SFT → GRPO

**Markdown — Wrap-up and takeaways:**

- Summary of the journey: data → tokens → embeddings → attention → pretraining → SFT → GRPO
- Key lessons: (1) Tokenization determines what model can learn, (2) Pretraining gives language understanding, (3) SFT teaches instruction following, (4) GRPO optimizes measurable qualities
- Intentional simplifications acknowledged (no full_note training, no reward model, synthetic data)
- Pointers for further exploration: larger models, LoRA/PEFT, real clinical data



## Visualization Summary (21 total)

| # | Type | Section | Purpose |
|||||
| 1 | 3-panel histogram | Data | Text length distributions |
| 2 | Word cloud | Data | Clinical vocabulary |
| 3 | Horizontal bar chart | Data | JSON key frequencies |
| 4 | Color-coded HTML + bar | Tokenization | General vs domain tokenizer |
| 5 | Horizontal bar chart | Tokenization | Top-50 token frequencies |
| 6 | Seaborn heatmap | Embeddings | Similarity matrix (random init) |
| 7 | Bar chart | Architecture | Parameter distribution |
| 8 | Seaborn heatmap | Attention | Single head attention (untrained) |
| 9 | 2×2 heatmap grid | Attention | All 4 heads (untrained) |
| 10 | Line plot | Pretraining | Loss curve |
| 11 | HTML table | Pretraining | Before/after generation |
| 12 | Seaborn heatmap | Pretraining | Attention (after training) |
| 13 | Scatter plot (PCA) | Pretraining | Embedding space clusters |
| 14 | 2-panel histogram | SFT | Prompt/completion lengths |
| 15 | Line plot | SFT | Loss curve |
| 16 | Grouped bar chart | SFT | JSON validity & schema metrics |
| 17 | HTML table | SFT | Before/after outputs |
| 18 | Table/bar chart | GRPO | Reward function demo |
| 19 | Multi-line plot | GRPO | Reward curves |
| 20 | HTML table | Final | 3-model output comparison |
| 21 | Grouped bar chart | Final | Quantitative metric comparison |



## Design Decisions

- **Single notebook** instead of 4 separate ones — keeps the storyline cohesive for a 3-hour session
- **USE_PRECOMPUTED flag** — toggling this lets the instructor choose live training vs instant checkpoint loading
- **Small data subset (3000 rows)** and **tiny model (4 layers, 256 dim)** — ensures everything runs on Colab T4 within time
- **No quantization/LoRA** for the tiny model — full fine-tuning is fine at this scale and avoids extra complexity
- **Matplotlib + seaborn only** (no plotly) — ensures Colab compatibility and simplicity
- **report_to="none"** and **save_strategy="no"** everywhere — avoids TensorBoard/WandB setup and disk usage



## Verification Plan

1. Run end-to-end on Google Colab T4 with `USE_PRECOMPUTED=False` to verify training completes
2. Save checkpoints and re-run with `USE_PRECOMPUTED=True` to verify checkpoint loading path
3. Check all 21 visualizations render correctly
4. Verify SFT and GRPO sections produce non-trivial outputs (JSON validity > 0% for SFT)
5. Time each section to confirm it fits the 3-hour schedule
6. Test on CPU (with `max_steps=5`) to verify the notebook doesn't crash without GPU
