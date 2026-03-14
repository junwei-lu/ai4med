# Reinforcement Learning Fine-Tuning with GRPO (Clinical Data)

Group Relative Policy Optimization (GRPO) fine-tunes LLMs via reinforcement learning without a separate value model. For each prompt, the policy samples multiple completions, receives rewards, and updates toward higher-reward behaviors.

This lecture covers:
- Setting up Llama 3 (smallest) with PEFT/quantization
- Preparing MedCalc-Bench-v1.0 data and a robust output template
- Writing reward functions (format + correctness)
- Configuring and running the GRPO trainer in `trl`

## 1) Environment setup

```bash
pip install --upgrade transformers datasets accelerate trl peft bitsandbytes torch tensorboard
```

## 2) Load model and tokenizer

Use 4-bit quantization (QLoRA-style) to fit training on limited VRAM; optionally add LoRA.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Meta-Llama-3-8B"

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
```

Optional: add LoRA to further reduce trainable parameters during RL.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

## 3) Data and prompt template (MedCalc-Bench)

We guide the model to produce structured outputs so rewards are easy and reliable.

```python
from datasets import load_dataset

raw_ds = load_dataset("ncbi/MedCalc-Bench-v1.0")
train_ds = raw_ds["train"]

SYSTEM_PROMPT = (
    "You are a clinical calculator assistant. "
    "Provide concise reasoning in <think>...</think> and the final numeric result in <answer>...</answer>."
)

def build_prompt(example):
    patient_note = example.get("Patient Note", "")
    question = example.get("Question", "")
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Patient Note: {patient_note}\n"
        f"Question: {question}\n"
        f"Answer with <think> and <answer> tags."
    )

prompts = [build_prompt(ex) for ex in train_ds]
references = [ex.get("Ground Truth Answer", "") for ex in train_ds]
train_data = [{"prompt": p, "reference": r} for p, r in zip(prompts, references)]
```

## 4) Reward functions

Use simple, deterministic signals first.

```python
import re

def format_reward(completions, **kwargs):
    # Expect both <think>...</think> and <answer>...</answer>
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    return [1.0 if isinstance(c, str) and pattern.search(c) else 0.0 for c in completions]


def exact_answer_reward(completions, references=None, **kwargs):
    out = []
    for c, ref in zip(completions, references or []):
        if not isinstance(c, str):
            out.append(0.0)
            continue
        m = re.search(r"<answer>(.*?)</answer>", c, re.DOTALL)
        pred = m.group(1).strip() if m else ""
        out.append(1.0 if pred == (ref or "").strip() else 0.0)
    return out


def numeric_tolerance_reward(completions, references=None, atol=0.5, **kwargs):
    out = []
    for c, ref in zip(completions, references or []):
        try:
            pred_m = re.search(r"<answer>\s*([+-]?[0-9]*\.?[0-9]+)", c)
            ref_m = re.search(r"([+-]?[0-9]*\.?[0-9]+)", ref)
            if pred_m and ref_m:
                pred_v = float(pred_m.group(1))
                ref_v = float(ref_m.group(1))
                out.append(1.0 if abs(pred_v - ref_v) <= atol else 0.0)
            else:
                out.append(0.0)
        except Exception:
            out.append(0.0)
    return out
```

Wrap correctness rewards so they can read references from samples passed by the trainer.

```python
def reward_wrapper(func):
    def _wrapped(completions, samples, **kwargs):
        refs = [s.get("reference", "") for s in samples]
        return func(completions, references=refs, **kwargs)
    return _wrapped
```

## 5) GRPO trainer configuration

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    output_dir="llama3-medcalc-grpo",
    learning_rate=1e-5,
    gradient_accumulation_steps=8,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    bf16=True,
    max_prompt_length=512,
    max_completion_length=128,
    num_generations=4,
    logging_steps=10,
    save_strategy="epoch",
    report_to=["tensorboard"],
    remove_unused_columns=False,
)

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=config,
    train_dataset=train_data,
    reward_funcs=[format_reward, reward_wrapper(numeric_tolerance_reward)],
)

trainer.train()
trainer.save_model()
```

## 6) Quick inference

```python
def generate_answer(question, patient_note=""):
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Patient Note: {patient_note}\n"
        f"Question: {question}\n"
        f"Answer with <think> and <answer> tags."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return text, (m.group(1).strip() if m else "")
```

## 7) Tips

- Keep rewards sparse and clear; start with one correctness signal
- Constrain outputs with tags to simplify parsing
- Start with `num_generations=2–4`; scale up if compute allows
- Validate on a held-out split by computing rewards without training

## 8) Safety

- Clinical models require rigorous validation before any real use
- Add checks for out-of-range or inconsistent results

## References

- Hugging Face TRL: GRPOTrainer, GRPOConfig (cookbook and docs)
- MedCalc-Bench-v1.0 dataset
