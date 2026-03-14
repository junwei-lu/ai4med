# Parameter-Efficient Fine-Tuning (PEFT): LoRA and Quantization

Large Language Models (LLMs) are expensive to fine-tune end-to-end. Parameter-Efficient Fine-Tuning (PEFT) adapts a pre-trained model by training a small number of additional parameters while keeping the original weights frozen. This lecture focuses on two beginner-friendly PEFT techniques widely used in practice:

- LoRA (Low-Rank Adaptation)
- Quantization (8-bit and 4-bit) and QLoRA (LoRA on a quantized base)

These methods enable fine-tuning models like Llama 3 on a single consumer GPU with limited memory. The examples below use the Hugging Face ecosystem: `transformers`, `datasets`, `peft`, `bitsandbytes`, and `trl`.

## Why PEFT?

- Reduce memory and compute costs by training fewer parameters
- Maintain strong performance by reusing powerful base models
- Enable domain adaptation on modest hardware

Typical use cases in biostatistics and biomedical research:

- Adapting a general LLM to clinical language or calculators
- Enforcing structured outputs (e.g., report templates)
- Reducing hallucinations via supervised examples

## Setup

```bash
pip install --upgrade \
  transformers datasets accelerate \
  bitsandbytes \
  trl \
  peft
```

Notes:
- `bitsandbytes` enables 8-bit/4-bit model loading to fit larger models into memory.
- Recent GPUs benefit from bf16; fallback to fp16 if needed.

## Quantization: 8-bit vs 4-bit

Quantization stores model weights in lower precision to reduce memory.

- 8-bit (int8): Good trade-off of speed and stability; widely used for inference and fine-tuning with LoRA.
- 4-bit (int4): Maximum compression; combined with LoRA → QLoRA for efficient fine-tuning on very limited VRAM.

Loading a causal LM (e.g., Llama 3 8B) with quantization:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "meta-llama/Meta-Llama-3-8B"

# Choose either 8-bit OR 4-bit config
bnb_8bit = BitsAndBytesConfig(load_in_8bit=True)
bnb_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_4bit,  # or bnb_8bit
    torch_dtype=torch.bfloat16,
)
```

Tips:
- If you encounter numerical instability on older GPUs, try `torch.float16` compute dtype.
- For chat-tuned models, ensure the tokenizer has correct special tokens and chat template.

## LoRA: Low-Rank Adaptation

LoRA inserts small low-rank adapters into selected linear layers and trains only those adapters. This drastically reduces trainable parameters while keeping the base model frozen.

Key hyperparameters:
- `r` (rank): capacity of adapters (common: 8–64 for small tasks, up to 256 for complex tasks)
- `lora_alpha`: scaling factor (e.g., 16–128)
- `lora_dropout`: regularization (e.g., 0.05–0.1)
- `target_modules`: which modules to adapt ("all-linear" is a robust default for many LLMs)

Minimal LoRA setup with `peft`:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

## QLoRA: LoRA on a 4-bit Base

QLoRA applies LoRA while keeping the base model in 4-bit quantization. This allows fine-tuning larger models (e.g., 8B) on 20–24GB GPUs.

Practical tips:
- Use `nf4` quant type and bf16 compute where possible
- Enable gradient checkpointing to trade compute for memory
- Consider training embeddings and `lm_head` if you add special tokens/chat templates

Example with `trl.SFTTrainer`:

```python
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer, setup_chat_format
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "meta-llama/Meta-Llama-3-8B"

# 4-bit base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)

# Ensure chat format if training on conversations
model, tokenizer = setup_chat_format(model, tokenizer)

# LoRA config typical for QLoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# Example: load a small jsonl dataset in OpenAI messages format
train_data = load_dataset("json", data_files="train_dataset.json", split="train")

args = TrainingArguments(
    output_dir="llama3-8b-qlora-demo",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    tf32=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    args=args,
    peft_config=peft_config,
    max_seq_length=2048,
    packing=True,
    dataset_kwargs={"add_special_tokens": False, "append_concat_token": False},
)

trainer.train()
trainer.save_model()
```

## Merging Adapters (Optional)

QLoRA saves only adapter weights. For simple deployment without PEFT, you can merge adapters into the base model on CPU and save a standalone checkpoint:

```python
from peft import AutoPeftModelForCausalLM

peft_dir = "llama3-8b-qlora-demo"
peft_model = AutoPeftModelForCausalLM.from_pretrained(peft_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
merged = peft_model.merge_and_unload()
merged.save_pretrained(peft_dir, safe_serialization=True, max_shard_size="2GB")
```

## Choosing Settings

- Start with 4-bit QLoRA for 8B models on 24GB VRAM; use 8-bit LoRA if you prefer extra stability
- Increase `r` for harder tasks; increase `lora_alpha` proportionally
- Use packing (`packing=True`) to boost throughput on short examples
- Monitor validation loss and sample outputs; early stop if overfitting

## Takeaways

- Quantization plus LoRA enables practical domain adaptation of LLMs on a single GPU
- QLoRA is a robust default for large models; LoRA+8-bit is a solid alternative
- Hugging Face `trl` + `peft` streamline the workflow from data to training to export


