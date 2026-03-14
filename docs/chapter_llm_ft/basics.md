# Basics of LLM Fine-Tuning (Beginner Guide)

This tutorial walks you through a minimal, reliable workflow to fine-tune an open LLM using Hugging Face tools. We use the smallest Llama 3 family model as an example and a simple instruction-style dataset aligned with the format used in the PEFT lecture.

You will learn to:
- Install the right libraries and pick a manageable model
- Prepare a beginner-friendly dataset and template
- Load the model with quantization (optional) and add LoRA if needed
- Set up a supervised fine-tuning trainer and run training

The guide assumes a biostatistics/biomedical background—no deep ML systems knowledge required.

## 1) Setup

```bash
# Install core libraries: model loading (transformers), datasets, training utils (accelerate),
# TRL trainer, PEFT adapters, quantization (bitsandbytes), PyTorch, and TensorBoard for logs
pip install --upgrade transformers datasets accelerate trl peft bitsandbytes torch tensorboard
```

Notes:
- `bitsandbytes` enables 8-bit/4-bit model loading to fit into modest GPUs
- If you don’t have a GPU, the code still runs but training will be slow

## 2) Choose a model

We use the smallest Llama 3 family model that is commonly accessible: Llama 3 8B. For consumer GPUs, 4-bit loading is helpful.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Choose a compact, widely used open model (8B parameters)
model_id = "meta-llama/Meta-Llama-3-8B"

# 4-bit loading reduces GPU memory usage so the model fits on consumer GPUs
# - double_quant: second quantization for improved memory/accuracy tradeoff
# - quant_type nf4: recommended 4-bit data type for LLMs
# - compute_dtype bf16: math is done in bfloat16 for speed/accuracy on modern GPUs
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Tokenizer splits text into tokens (ids); must match the model
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model with quantization and let HF infer device placement (CPU/GPU)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)
```

## 3) Prepare a simple instruction dataset

We align with the format in the PEFT lecture: OpenAI-style messages with a system instruction, a user prompt, and an assistant answer. This mirrors how you would prepare clinical data (e.g., MedCalc-style questions with ground-truth answers), but here we keep a simple schema for clarity.

Data format (JSONL), each line is one record:
```json
{"messages": [
  {"role": "system", "content": "You are a clinical calculator assistant."},
  {"role": "user", "content": "Patient Note: ...\nQuestion: ...\nAnswer:"},
  {"role": "assistant", "content": "95 mL/min"}
]}
```

Create a tiny demo dataset programmatically (replace with your real data later):

```python
import json

# Minimal demo dataset in OpenAI "messages" format
# In practice, generate many such samples from your clinical data sources
train_records = [
    {
        "messages": [
            {"role": "system", "content": "You are a clinical calculator assistant."},
            {"role": "user", "content": "Patient Note: 16-year-old female with severe hypertension...\nQuestion: Compute creatinine clearance (Cockcroft-Gault).\nAnswer:"},
            {"role": "assistant", "content": "95 mL/min"}  # final answer only; keep unit consistent
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a clinical calculator assistant."},
            {"role": "user", "content": "Patient Note: BMI example.\nQuestion: Height 1.75m, Weight 70kg.\nAnswer:"},
            {"role": "assistant", "content": "22.86"}  # numeric-only answer for BMI (unitless)
        ]
    }
]

# Write as JSONL (one JSON object per line), which HF datasets can read efficiently
with open("train_dataset.json", "w") as f:
    for r in train_records:
        f.write(json.dumps(r) + "\n")
```

Why this template?
- A consistent structure simplifies tokenization and training
- The final answer is clear and easy to evaluate later

## 4) Optional: Add LoRA (PEFT)

LoRA reduces the number of trainable parameters. It’s often used with 4-bit loading (QLoRA) to fine-tune on small GPUs.

```python
from peft import LoraConfig, get_peft_model

# LoRA adds tiny trainable adapters to selected linear layers
# - r: adapter rank (capacity)
# - lora_alpha: scaling factor for adapter updates
# - lora_dropout: regularization during training
# - target_modules: which layers to adapt; "all-linear" is a safe default for many LLMs
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# Wrap the base model so only adapters (not full model) are trained
model = get_peft_model(model, lora_config)
```

## 5) Set up the trainer (Supervised Fine-Tuning)

We use TRL’s `SFTTrainer` for simplicity. It natively supports the messages format and PEFT.

```python
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer, setup_chat_format

# Ensure chat formatting and special tokens are set (adds tokens and a chat template)
model, tokenizer = setup_chat_format(model, tokenizer)

# Load the JSONL dataset from disk
train_ds = load_dataset("json", data_files="train_dataset.json", split="train")

# Basic training config; start small for quick feedback, then scale
args = TrainingArguments(
    output_dir="llama3-8b-basics-sft",   # where checkpoints/logs are saved
    num_train_epochs=1,                   # try 1 epoch first to verify pipeline
    per_device_train_batch_size=1,        # small batch to fit in memory
    gradient_accumulation_steps=8,        # effective batch size = 1 * 8
    gradient_checkpointing=True,          # trade compute for lower memory
    learning_rate=2e-4,                   # typical LR for LoRA fine-tuning
    bf16=True,                            # use bfloat16 on supported GPUs (e.g., A100/4090)
    tf32=True,                            # faster matmul on Ampere+
    logging_steps=10,                     # log every N steps
    save_strategy="epoch",                # save at end of epoch
    report_to="none",                    # set to "tensorboard" if you want TB logs
)

# SFTTrainer handles message formatting, masking prompts, and PEFT integration
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    args=args,
    max_seq_length=2048,                  # truncate/pack sequences to this length
    packing=True,                         # pack multiple short samples in one sequence
    dataset_kwargs={"add_special_tokens": False, "append_concat_token": False},
)

# Run training; checkpoint is saved automatically
trainer.train()
# Save final adapter or full model (if not using PEFT)
trainer.save_model()
```

Tips for biomedical data:
- Keep prompts short and focused: patient note + question; end with “Answer:”
- If answers are numeric (e.g., mL/min), use a consistent unit and precision
- Start with a few hundred curated examples, then scale up

## 6) Quick inference

```python
def generate(prompt):
    # Tokenize the prompt and move tensors to the model's device (CPU/GPU)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Disable gradients for faster, memory‑efficient inference
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    # Convert token ids back to text, skipping special tokens
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example: ask for a numeric clinical answer, consistent with training template
user_q = (
    "Patient Note: 16-year-old female with severe hypertension...\n"
    "Question: Compute creatinine clearance (Cockcroft-Gault).\n"
    "Answer:"
)
print(generate(f"You are a clinical calculator assistant.\n\n{user_q}"))
```

## 7) What to try next

- Add evaluation: compare model outputs against ground truth answers
- Expand dataset with more clinical calculators (e.g., BMI, eGFR)
- Use curriculum: start with simple tasks, then harder ones
- Consider GRPO (see RL lecture) if you want to optimize non-differentiable rewards

## Takeaways

- A clean template and consistent units make clinical fine-tuning tractable
- 4-bit loading + LoRA allows training 8B models on a single GPU
- `SFTTrainer` keeps the workflow simple for beginners
