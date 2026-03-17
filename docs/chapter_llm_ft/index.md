## Overview

This chapter covers fine-tuning large language models (LLMs) built on the Transformer architecture. We start from the mathematical foundations of how LLMs are trained and progress through parameter-efficient adaptation and reinforcement-learning-based fine-tuning.

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) [Open In Colab](https://colab.research.google.com/github/junwei-lu/ai4med/blob/main/codes/nlp/llm_model2pretrain2ft.ipynb)

## Lectures

- **[Next-Token Prediction](ntp.md)**  
  The core pre-training objective: autoregressive factorization, cross-entropy loss, perplexity, and PyTorch implementation.

- **[Supervised Fine-Tuning (SFT)](sft.md)**  
  The SFT loss function with prompt masking, MLE interpretation, and end-to-end training.

- **[RL Fine-Tuning with GRPO](grpo.md)**  
  The GRPO objective, group-relative advantage, clipped surrogate loss, KL penalty, and reward function design.

- **[Direct Preference Optimization (DPO)](dpo.md)**  
  Aligning models with human preferences directly using preference pairs (chosen vs. rejected) without a reward model.

- **[PEFT: LoRA and QLoRA](peft.md)**  
  Low-rank adaptation math, scaling, parameter savings, and QLoRA with quantized base models.

