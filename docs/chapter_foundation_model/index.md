# Chapter 13. Genomic Foundation Models

This chapter explores the concept of **Foundation Models**—large-scale models trained on vast amounts of data that can be adapted to a wide range of downstream tasks. We will use the SNP as the example to show you how to build transformer-based foundation models beyond natural language.


![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) [Open In Colab](https://colab.research.google.com/github/junwei-lu/ai4med/blob/main/codes/snp/snp_genomics_workshop.ipynb)

*   **[SNP Tokenization & Architecture](snp_tokenization.md)**: How to tokenize DNA (K-mers) and why Encoder-only architectures are preferred.
*   **[SNP Model Training](snp_training.md)**: Pretraining objectives (Masked Language Modeling) and downstream tasks (eQTL prediction).
*   **[SNP Fine-Tuning](snp_finetuning.md)**: Practical tutorial on fine-tuning genomic models using LoRA.
*   **[Advanced Genomic Models](snp_advanced_models.md)**: Exploring long-range models (Enformer) and single-cell foundation models (scGPT).
*   **[Model Interpretation](snp_interpretation.md)**: Using Saliency Maps and attention analysis to interpret model predictions biologically.
