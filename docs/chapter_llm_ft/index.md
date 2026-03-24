## Overview

<div style="display: flex; gap: 10px; margin-bottom: 20px;"><a href="https://colab.research.google.com/github/junwei-lu/ai4med/blob/main/codes/nlp/llm_model2pretrain2ft.ipynb" target="_blank" style="display: inline-flex; align-items: center; gap: 8px; padding: 10px 16px; background: linear-gradient(135deg, #1565c0 0%, #42a5f5 100%); color: white; border-radius: 8px; text-decoration: none; font-weight: 600; box-shadow: 0 4px 15px rgba(21, 101, 192, 0.4);"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"/></svg>Open Interactive Notebook in Colab</a></div>

This chapter covers fine-tuning large language models (LLMs) built on the Transformer architecture. We start from the mathematical foundations of how LLMs are trained and progress through parameter-efficient adaptation and reinforcement-learning-based fine-tuning.


![Cover](./ft.assets/llm_train.gif)

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
