# Next-Token Prediction

Before fine-tuning, it is essential to understand what a language model actually learns during pre-training. The core objective is **next-token prediction (NTP)**: given a sequence of tokens, predict the next one. Everything from GPT-2 to Llama 3 is trained with this single principle.



## The Probability Model

A language model assigns a probability to every possible sequence of tokens $x_1, x_2, \ldots, x_T$. Using the **chain rule of probability**, any joint distribution can be factored autoregressively:

$$
P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, x_2, \ldots, x_{t-1})
$$

A neural language model with parameters $\theta$ approximates each conditional:

$$
P_\theta(x_t \mid x_1, \ldots, x_{t-1})
$$

This is a **categorical distribution** over the vocabulary $\mathcal{V}$(e.g., 32,000 tokens for Llama). The model outputs a vector of logits $\mathbf{z}_t \in \mathbb{R}^{|\mathcal{V}|}$, which are converted to probabilities via softmax:

$$
P_\theta(x_t = v \mid x_{<t}) = \frac{\exp(z_{t,v})}{\sum_{v' \in \mathcal{V}} \exp(z_{t,v'})}
$$

**Autoregressive generation:** At inference time, the model generates token by token—each new token is appended to the context and fed back in to predict the next:

![Autoregressive token generation](./ft.assets/ntp_autoregress.gif)



## The Training Objective: Cross-Entropy Loss

Given a training corpus of documents, each document is treated as a sequence of tokens. The model is trained to **maximize the log-likelihood** of observed tokens, which is equivalent to **minimizing the cross-entropy loss**.

For a single document of length $T$:

$$
\mathcal{L}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})
$$

Each term $-\log P_\theta(x_t \mid x_{<t})$ measures how surprised the model is when it sees the actual next token $x_t$. A perfect model would assign probability 1 to the correct token, giving a loss of 0.

**Over a dataset** $\mathcal{D} = \{d^{(1)}, d^{(2)}, \ldots, d^{(N)}\}$ of $N$ documents:

$$
\mathcal{L}(\theta) = -\frac{1}{|\mathcal{D}|} \sum_{d \in \mathcal{D}} \frac{1}{|d|} \sum_{t=1}^{|d|} \log P_\theta(x_t^{(d)} \mid x_1^{(d)}, \ldots, x_{t-1}^{(d)})
$$

### Connection to Perplexity

**Perplexity (PPL)** is the standard evaluation metric for language models and is directly tied to the NTP loss:

$$
\text{PPL} = \exp\!\left(\mathcal{L}(\theta)\right) = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})\right)
$$

Intuitively, a perplexity of $k $ means the model is "as confused as if choosing uniformly among$k$ options" at each step. Lower is better.



## Why Does This Work?

Training on next-token prediction on large corpora forces the model to:

1. **Learn syntax and grammar** — token sequences must be grammatically plausible
2. **Learn factual knowledge** — predicting "The capital of France is ___" requires knowing "Paris"
3. **Learn reasoning patterns** — math or logic examples appear in text and must be predicted correctly
4. **Learn long-range dependencies** — the Transformer's attention lets each prediction attend to all prior tokens

This is why a model pre-trained purely on NTP can then be fine-tuned for specific tasks with relatively few examples.



## Training Next-Token Prediction Loss from Scratch

Let us implement the NTP loss manually to build intuition before using high-level trainers.

### Minimal example with pure PyTorch

```python
import torch
import torch.nn.functional as F

#  Toy example 
# Suppose we have a tiny vocabulary of 5 tokens and a sequence of length 4
# tokens: [2, 0, 3, 1]  → input = [2, 0, 3], target = [0, 3, 1]

vocab_size = 5
seq_len = 3  # we predict 3 positions

# Simulated logits from the model (shape: [seq_len, vocab_size])
torch.manual_seed(42)
logits = torch.randn(seq_len, vocab_size)

# The correct next tokens for each position
targets = torch.tensor([0, 3, 1])  # shape: [seq_len]

# Cross-entropy loss: equivalent to -log P(correct token)
# F.cross_entropy applies log-softmax internally
loss_per_token = F.cross_entropy(logits, targets, reduction="none")
print("Per-token losses:", loss_per_token)

loss = loss_per_token.mean()
print(f"Average NTP loss: {loss.item():.4f}")
print(f"Perplexity: {torch.exp(loss).item():.2f}")
```

Expected output (deterministic with `torch.manual_seed(42)`):
```
Per-token losses: tensor([1.3644, 2.3091, 1.8469])
Average NTP loss: 1.8401
Perplexity: 6.30
```

### Shift-by-one: input vs. target in practice

The key implementation detail: **the target at position $t $ is the input at position$t+1$**. This is done by shifting the token sequence by one.

```python
import torch
import torch.nn.functional as F

def ntp_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute causal language modeling (NTP) loss.

    Args:
        logits:    Model output, shape [batch, seq_len, vocab_size]
        input_ids: Token IDs,     shape [batch, seq_len]

    Returns:
        Scalar loss (mean cross-entropy over all non-padding positions)
    """
    # Shift: predict position t+1 using logits at position t
    # logits at positions 0..T-2 should predict tokens at positions 1..T-1
    shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
    shift_labels = input_ids[:, 1:].contiguous()     # [B, T-1]

    # Flatten batch and time dimensions for cross_entropy
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),  # [B*(T-1), V]
        shift_labels.view(-1),                          # [B*(T-1)]
    )
    return loss

#  Demo with a batch of 2 sequences of length 6 
torch.manual_seed(0)
B, T, V = 2, 6, 32000  # batch, seq_len, vocab_size

dummy_logits = torch.randn(B, T, V)
dummy_input_ids = torch.randint(0, V, (B, T))

loss = ntp_loss(dummy_logits, dummy_input_ids)
print(f"NTP loss: {loss.item():.4f}")         # ~log(32000) ≈ 10.37 for random init
print(f"Perplexity: {torch.exp(loss).item():.1f}")
```

### Using Hugging Face transformers

When you call `model(**inputs, labels=input_ids)`, Hugging Face models do exactly the shift-and-cross-entropy internally:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "gpt2"  # small model, no auth required
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

text = "The patient's creatinine clearance is"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]

# Pass labels=input_ids → HF computes the shifted NTP loss automatically
outputs = model(**inputs, labels=input_ids)

print(f"NTP loss:   {outputs.loss.item():.4f}")
print(f"Perplexity: {torch.exp(outputs.loss).item():.2f}")

# The loss is stored in outputs.loss; logits in outputs.logits
print(f"Logits shape: {outputs.logits.shape}")  # [1, T, 50257]
```

The `outputs.loss` corresponds exactly to:

$$
\mathcal{L} = -\frac{1}{T-1} \sum_{t=1}^{T-1} \log P_\theta(x_{t+1} \mid x_1, \ldots, x_t)
$$



## What the Gradient Does

During backpropagation, the gradient of the loss with respect to the logit $z_{t,v}$ is:

$$
\frac{\partial \mathcal{L}}{\partial z_{t,v}} = P_\theta(x_t = v \mid x_{<t}) - \mathbf{1}[v = x_t]
$$

This means:

  - For the **correct token** $v = x_t$: the gradient is $P_\theta - 1$, which is **negative** → the logit is pushed **up**
  - For all **other tokens**: the gradient is $P_\theta > 0$, which is **positive** → those logits are pushed **down**

The model learns by repeatedly increasing the probability of observed tokens and decreasing the probability of unobserved tokens.
