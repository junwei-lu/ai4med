# Chapter 13. Foundation Models

This chapter explores the concept of **Foundation Models**—large-scale models trained on vast amounts of data that can be adapted to a wide range of downstream tasks. We will cover both general-purpose foundation models and their specific application to **Genomics** (SNP Foundation Models).

## Building Foundation Models
We start by understanding how to build and pretrain foundation models from scratch.

*   **[Building a Foundation Model](overview.md)**: An overview of the architecture, data requirements, and training objectives.
*   **[Coding Tutorial: Pretraining](coding_build_pretrain.md)**: A step-by-step guide to coding the pretraining loop.
*   **[Fine-Tuning with Hugging Face](finetune.md)**: How to adapt a pretrained model to specific tasks using the Hugging Face ecosystem.

## SNP Foundation Models
We then dive deep into applying these principles to DNA sequences, treating the genome as a language.

*   **[SNP Tokenization & Architecture](snp_tokenization.md)**: How to tokenize DNA (K-mers) and why Encoder-only architectures are preferred.
*   **[SNP Model Training](snp_training.md)**: Pretraining objectives (Masked Language Modeling) and downstream tasks (eQTL prediction).
*   **[SNP Fine-Tuning](snp_finetuning.md)**: Practical tutorial on fine-tuning genomic models using LoRA.
*   **[Advanced Genomic Models](snp_advanced_models.md)**: Exploring long-range models (Enformer) and single-cell foundation models (scGPT).
*   **[Model Interpretation](snp_interpretation.md)**: Using Saliency Maps and attention analysis to interpret model predictions biologically.
