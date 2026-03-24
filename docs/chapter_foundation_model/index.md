# Genomic Foundation Models

This chapter explores the concept of **Foundation Models**—large-scale models trained on vast amounts of data that can be adapted to a wide range of downstream tasks. We will use the SNP as the example to show you how to build transformer-based foundation models beyond natural language.

<div style="display: flex; gap: 10px; margin-bottom: 20px;"><a href="https://colab.research.google.com/github/junwei-lu/ai4med/blob/main/codes/snp/snp_genomics_workshop.ipynb" target="_blank" style="display: inline-flex; align-items: center; gap: 8px; padding: 10px 16px; background: linear-gradient(135deg, #1565c0 0%, #42a5f5 100%); color: white; border-radius: 8px; text-decoration: none; font-weight: 600; box-shadow: 0 4px 15px rgba(21, 101, 192, 0.4);"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"/></svg>Open Interactive Notebook in Colab</a></div>

*   **[SNP Tokenization & Architecture](snp_tokenization.md)**: How to tokenize DNA (K-mers) and why Encoder-only architectures are preferred.
*   **[SNP Model Training](snp_training.md)**: Pretraining objectives (Masked Language Modeling) and downstream tasks (eQTL prediction).
*   **[SNP Fine-Tuning](snp_finetuning.md)**: Practical tutorial on fine-tuning genomic models using LoRA.
*   **[Advanced Genomic Models](snp_advanced_models.md)**: Exploring long-range models (Enformer) and single-cell foundation models (scGPT).
*   **[Model Interpretation](snp_interpretation.md)**: Using Saliency Maps and attention analysis to interpret model predictions biologically.
