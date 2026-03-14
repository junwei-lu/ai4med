# Reinforcement Learning Fine-Tuning with GRPO

Group Relative Policy Optimization (GRPO) fine-tunes LLMs via reinforcement learning without a separate value model. For each prompt, the policy samples multiple completions, receives rewards, and updates toward higher-reward behaviors.

This lecture covers:
- The mathematical objective of GRPO and how it compares to PPO
- Setting up Llama 3 (smallest) with PEFT/quantization
- Preparing MedCalc-Bench-v1.0 data and a robust output template
- Writing reward functions (format + correctness)
- Configuring and running the GRPO trainer in `trl`

---

## From SFT to RL Fine-Tuning

SFT (see [Supervised Fine-Tuning](sft.md)) trains the model to imitate gold-standard responses via maximum likelihood. It has a key limitation: **you need labeled responses**, and the loss treats all tokens of a response equally regardless of whether the response is actually correct.

RL fine-tuning replaces the supervised signal with a **reward function** $r(y, x)$ that scores a sampled response $ y $ for a given prompt $ x$. This lets you optimize for non-differentiable objectives like numerical accuracy, format compliance, or clinical correctness.

---

## The GRPO Objective

GRPO was introduced in the [DeepSeekMath paper](https://arxiv.org/abs/2402.03300) and popularized for chain-of-thought reasoning tasks. It eliminates the value/critic network required by PPO, making it simpler and more memory-efficient.

### 2.1 Algorithm overview

For each training step:

1. **Sample** a prompt $q \sim \mathcal{Q}$ from the training distribution
2. **Generate** a group of $G$ completions from the current (old) policy:  
   $\{o_1, o_2, \ldots, o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)$
3. **Score** each completion with a reward function: $r_i = r(o_i, q)$
4. **Normalize** rewards within the group to compute advantages
5. **Update** the policy $\pi_\theta$ to increase the probability of high-advantage completions

### 2.2 Advantage computation

Unlike PPO which trains a separate value network $V(s)$ to estimate expected return, GRPO uses the **group mean and standard deviation** as a baseline:

$$
\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\}) + \epsilon}
$$

This is the reward of completion $i$ relative to its peers in the group. A completion that scores above the group average gets a positive advantage; one below average gets a negative advantage. No separate value network is needed.

### 2.3 The GRPO objective (full formula)

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q,\, \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}} \left[
\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
\left\{
\min\!\left( \rho_{i,t}\, \hat{A}_i,\; \text{clip}\!\left(\rho_{i,t}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i \right)
- \beta\, \mathbb{D}_{\text{KL}}\!\left[\pi_\theta \,\|\, \pi_{\text{ref}}\right]
\right\}
\right]
$$

where:

| Symbol | Meaning |
|--------|---------|
| $\rho_{i,t} = \dfrac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$ | Probability ratio: new policy vs. old (importance weight) |
| $\hat{A}_i $ | Group-normalized advantage for completion $ i$ |
| $\epsilon$ | Clipping range (e.g., 0.2); prevents too-large policy updates |
| $\beta$ | KL penalty coefficient; keeps policy close to a reference model |
| $\pi_{\text{ref}}$ | Reference (base) policy, typically the SFT-initialized model |
| $G$ | Group size (number of completions per prompt, e.g., 4–16) |

### 2.4 The clipped surrogate loss

The $\min(\rho\hat{A},\, \text{clip}(\rho, 1-\epsilon, 1+\epsilon)\hat{A})$ term is the PPO clipped surrogate objective:

- When $\hat{A}_i > 0 $ (good completion): the ratio $\rho $ is clipped at $ 1 + \epsilon$, preventing the policy from jumping too aggressively toward this completion
- When $\hat{A}_i < 0 $ (bad completion): the ratio $\rho $ is clipped at $ 1 - \epsilon$, preventing the policy from moving too far away from it in one step

This trust-region mechanism is what makes the training stable.

### 2.5 The KL divergence term

The KL penalty:

$$
\mathbb{D}_{\text{KL}}\!\left[\pi_\theta \,\|\, \pi_{\text{ref}}\right] = \log \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}
$$

prevents the policy from collapsing into reward hacking—producing nonsensical outputs that happen to score high on a simple reward function. A typical value is $\beta = 0.01$.

### 2.6 GRPO vs PPO: key difference

| Feature | PPO | GRPO |
|---------|-----|------|
| Advantage estimate | Requires trained value network $V_\phi(s)$ | Group mean/std normalization |
| Memory | Model + value network + ref model | Model + ref model (no value net) |
| Stability | High (value network reduces variance) | Good (group baseline reduces variance) |
| Simplicity | More complex | Simpler to implement |

GRPO's trick: **the group of completions for the same prompt acts as a natural control group**, giving a low-variance advantage estimate without a separate network.

---

## Connecting the Math to Code

The GRPO formula maps directly to the `GRPOConfig` parameters:

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    # ── Reward / advantage ──────────────────────────────────────
    num_generations=4,          # G: group size; 4 completions per prompt
    # advantages = (r_i - mean) / std  computed automatically

    # ── Clipped surrogate ───────────────────────────────────────
    # epsilon = 0.2 (default)   → clip(ρ, 0.8, 1.2)

    # ── KL penalty ──────────────────────────────────────────────
    # beta = 0.04 (default)     → β * D_KL[π_θ || π_ref]

    # ── Optimization ────────────────────────────────────────────
    learning_rate=1e-5,
    gradient_accumulation_steps=8,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    bf16=True,
    max_prompt_length=512,
    max_completion_length=128,

    # ── Logging ─────────────────────────────────────────────────
    output_dir="llama3-medcalc-grpo",
    logging_steps=10,
    save_strategy="epoch",
    report_to=["tensorboard"],
    remove_unused_columns=False,
)
```

---

## Environment Setup

```bash
pip install --upgrade transformers datasets accelerate trl peft bitsandbytes torch tensorboard
```

---

## Load Model and Tokenizer

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

---

## Data and Prompt Template (MedCalc-Bench)

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

---

## Reward Functions

The reward function $r(o_i, q)$ is the core design choice in GRPO. Use simple, deterministic signals first.

```python
import re

def format_reward(completions, **kwargs):
    """
    r = 1.0 if the completion contains both <think>…</think> and <answer>…</answer>.
    Encourages structured output format.
    """
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    return [1.0 if isinstance(c, str) and pattern.search(c) else 0.0 for c in completions]


def exact_answer_reward(completions, references=None, **kwargs):
    """r = 1.0 if extracted <answer> matches reference exactly."""
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
    """r = 1.0 if extracted numeric answer is within atol of reference."""
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

### How rewards map to advantages

For a group of $G = 4 $ completions with rewards $[1, 0, 1, 0]$:

$$
\text{mean} = 0.5, \quad \text{std} = 0.5
$$

$$
\hat{A} = \left[\frac{1 - 0.5}{0.5}, \frac{0 - 0.5}{0.5}, \frac{1 - 0.5}{0.5}, \frac{0 - 0.5}{0.5}\right] = [1.0, -1.0, 1.0, -1.0]
$$

The policy is updated to increase the probability of completions 1 and 3, and decrease it for completions 2 and 4.

```python
# Demonstrate advantage computation manually
import numpy as np

rewards = np.array([1.0, 0.0, 1.0, 0.0])
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
print("Rewards:    ", rewards)
print("Advantages: ", advantages)
# Rewards:     [1. 0. 1. 0.]
# Advantages:  [ 1. -1.  1. -1.]
```

Wrap correctness rewards so they can read references from samples passed by the trainer:

```python
def reward_wrapper(func):
    def _wrapped(completions, samples, **kwargs):
        refs = [s.get("reference", "") for s in samples]
        return func(completions, references=refs, **kwargs)
    return _wrapped
```

---

## GRPO Trainer Configuration

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
    num_generations=4,      # G: group size (completions per prompt)
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
    # Reward functions are called as r(completions, samples, **kwargs)
    # The trainer automatically combines them: r_total = sum of all reward signals
    reward_funcs=[format_reward, reward_wrapper(numeric_tolerance_reward)],
)

trainer.train()
trainer.save_model()
```

**What the trainer does at each step:**

1. Sample `num_generations=4` completions per prompt using `π_θ_old`
2. Compute rewards: `r_total = format_reward(...) + numeric_tolerance_reward(...)`
3. Compute group advantages: `Â_i = (r_i - mean) / std`
4. For each token in each completion, compute `ρ = π_θ / π_θ_old`
5. Compute clipped surrogate loss + KL penalty
6. Backpropagate and update `θ`

---

## Quick Inference

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

---

## Tips

- Keep rewards sparse and clear; start with one correctness signal
- Constrain outputs with tags to simplify parsing and reward computation
- Start with `num_generations=4`; scale up if compute allows
- Validate on a held-out split by computing rewards without training
- If all completions in a group score identically, advantages are all zero → no learning signal; diversify prompts
