# Supervised Fine-Tuning

This tutorial walks you through a minimal, reliable workflow to fine-tune an open LLM using Hugging Face tools. We use the smallest Llama 3 family model as an example and a simple instruction-style dataset aligned with the format used in the PEFT lecture.

You will learn to:
- Understand the SFT loss function and how it differs from raw pre-training
- Install the right libraries and pick a manageable model
- Prepare a beginner-friendly dataset and template
- Set up a supervised fine-tuning trainer and run training

The guide assumes a biostatistics/biomedical background—no deep ML systems knowledge required.



## The SFT Loss Function

### Starting point: NTP loss

Recall from the [Next-Token Prediction](ntp.md) tutorial that pre-training minimizes:

$$
\mathcal{L}_{\text{NTP}}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})
$$

over all tokens in the corpus. **Every** token—whether a system prompt, user question, or assistant answer—contributes equally to the loss.

### SFT: only learn from the response

In supervised fine-tuning, the training data consists of **input-output pairs**:

$$
\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{N}
$$

where $x^{(i)}$ is the **prompt** (system instruction + user turn) and $ y^{(i)}$ is the **target response** (assistant turn). The goal is to teach the model to produce $ y $given$ x $.

The SFT loss is the NTP loss computed **only on the response tokens**, with the prompt tokens masked out:

$$
\boxed{
\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \frac{1}{|y^{(i)}|} \sum_{t=1}^{|y^{(i)}|} \log P_\theta\!\left(y_t^{(i)} \;\middle|\; x^{(i)}, y_1^{(i)}, \ldots, y_{t-1}^{(i)}\right)
}
$$

**Why mask the prompt?**  
- Prompt tokens are given as context, not generated—penalizing the model for not predicting them would confuse training
- Masking focuses capacity on learning the *style* and *content* of the target response
- It also allows much longer prompts without inflating the loss denominator

### Masking in practice

The `SFTTrainer` implements this via a label mask: positions corresponding to the prompt are set to `-100`, and PyTorch's `cross_entropy` ignores positions with label `-100`.

```
Full sequence:  [SYS] You are a clinical assistant.  [USER] What is the eGFR?  [ASST] 45 mL/min
                ↑________________________ prompt _____________________↑  ↑____ response ____↑
Label mask:         -100    -100    ...      -100       -100   -100  ...   45    mL   /   min
Loss computed:      ✗       ✗                ✗          ✗      ✗           ✓    ✓    ✓    ✓
```

### SFT as maximum likelihood estimation

Maximizing the log-likelihood of responses is equivalent to maximum likelihood estimation (MLE) of the conditional distribution:

$$
\theta^* = \arg\max_\theta \sum_{i=1}^{N} \log P_\theta(y^{(i)} \mid x^{(i)})
$$

Each response $y^{(i)}$ factorizes autoregressively, so:

$$
\log P_\theta(y^{(i)} \mid x^{(i)}) = \sum_{t=1}^{|y^{(i)}|} \log P_\theta\!\left(y_t^{(i)} \mid x^{(i)}, y_{<t}^{(i)}\right)
$$

which is exactly the inner sum in the SFT loss.



## Choose a model

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



## Prepare a simple instruction dataset

We align with the format in the PEFT lecture: OpenAI-style messages with a system instruction, a user prompt, and an assistant answer. This mirrors how you would prepare clinical data (e.g., MedCalc-style questions with ground-truth answers), but here we keep a simple schema for clarity.

Data format (JSONL), each line is one record corresponding to one $(x^{(i)}, y^{(i)})$ pair:

```json
{"messages": [
  {"role": "system",    "content": "You are a clinical calculator assistant."},
  {"role": "user",      "content": "Patient Note: ...\nQuestion: ...\nAnswer:"},
  {"role": "assistant", "content": "95 mL/min"}
]}
```

The **prompt** $x^{(i)}$ consists of the `system` + `user` turns. The **response** $ y^{(i)}$ is the `assistant` turn. Loss is computed only on `assistant` tokens.

Create a tiny demo dataset programmatically (replace with your real data later):

```python
import json

# Minimal demo dataset in OpenAI "messages" format
# Each record = one (x, y) pair; loss is computed on assistant turn only
train_records = [
    {
        "messages": [
            {"role": "system",    "content": "You are a clinical calculator assistant."},
            {"role": "user",      "content": "Patient Note: 16-year-old female with severe hypertension...\nQuestion: Compute creatinine clearance (Cockcroft-Gault).\nAnswer:"},
            {"role": "assistant", "content": "95 mL/min"}  # ← SFT loss computed here
        ]
    },
    {
        "messages": [
            {"role": "system",    "content": "You are a clinical calculator assistant."},
            {"role": "user",      "content": "Patient Note: BMI example.\nQuestion: Height 1.75m, Weight 70kg.\nAnswer:"},
            {"role": "assistant", "content": "22.86"}  # ← SFT loss computed here
        ]
    }
]

# Write as JSONL (one JSON object per line), which HF datasets can read efficiently
with open("train_dataset.json", "w") as f:
    for r in train_records:
        f.write(json.dumps(r) + "\n")
```

**Why this template?**
- A consistent structure simplifies tokenization and training
- The final answer is clear and easy to evaluate later



## Verifying the Masking Manually

Before training, it is instructive to verify that the trainer correctly masks prompt tokens. Here is a minimal check:

```python
from transformers import AutoTokenizer
from trl import setup_chat_format
import torch

model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system",    "content": "You are a clinical calculator assistant."},
    {"role": "user",      "content": "Compute creatinine clearance.\nAnswer:"},
    {"role": "assistant", "content": "95 mL/min"},
]

# Apply chat template; return tensors as token ids
full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]

# Identify where the assistant turn starts
prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
prompt_len = len(prompt_ids)

# Build labels: -100 for prompt positions, actual ids for response positions
labels = torch.full_like(full_ids, -100)
labels[prompt_len:] = full_ids[prompt_len:]

print("Prompt tokens (masked):", full_ids[:prompt_len])
print("Response tokens (loss):", full_ids[prompt_len:])
print("Labels:", labels)
# -100 positions are ignored; loss is only on response tokens
```



## Set up the trainer (Supervised Fine-Tuning)

We use TRL's `SFTTrainer` for simplicity. It natively supports the messages format and PEFT. Under the hood it:

1. Applies the chat template to flatten messages into a single string
2. Tokenizes the full sequence
3. Builds the label mask (sets prompt tokens to `-100`)
4. Computes the SFT loss: $\mathcal{L}_{\text{SFT}} = -\frac{1}{|y|}\sum_{t} \log P_\theta(y_t \mid x, y_{<t})$

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
    gradient_accumulation_steps=8,        # effective batch size = 1 × 8
    gradient_checkpointing=True,          # trade compute for lower memory
    learning_rate=2e-4,                   # typical LR for LoRA fine-tuning
    bf16=True,                            # use bfloat16 on supported GPUs (e.g., A100/4090)
    tf32=True,                            # faster matmul on Ampere+
    logging_steps=10,                     # log every N steps
    save_strategy="epoch",                # save at end of epoch
    report_to="none",                     # set to "tensorboard" if you want TB logs
)

# SFTTrainer handles message formatting, masking prompts, and PEFT integration
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    args=args,
    max_seq_length=2048,       # truncate/pack sequences to this length
    packing=True,              # pack multiple short samples in one sequence
    dataset_kwargs={"add_special_tokens": False, "append_concat_token": False},
)

# Run training; loss logged = SFT loss averaged over response tokens
trainer.train()
trainer.save_model()
```

**Tips for biomedical data:**
- Keep prompts short and focused: patient note + question; end with "Answer:"
- If answers are numeric (e.g., mL/min), use a consistent unit and precision
- Start with a few hundred curated examples, then scale up



## Quick inference

```python
def generate(prompt):
    # Tokenize the prompt and move tensors to the model's device (CPU/GPU)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Disable gradients for faster, memory-efficient inference
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



## What to try next

- Add evaluation: compare model outputs against ground truth answers
- Expand dataset with more clinical calculators (e.g., BMI, eGFR)
- Use curriculum: start with simple tasks, then harder ones
- Consider GRPO (see [RL Fine-Tuning](grpo.md)) if you want to optimize non-differentiable rewards

