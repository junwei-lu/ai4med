# Self-Supervised Learning

Self-supervised learning is the art of getting a model to teach itself from raw data before we ask it to solve a specific task. Instead of paying humans to label every image, we design a training signal from the data itself and use that to build a strong **encoder**.

Why this matters:

- labels are expensive,
- unlabeled images are everywhere,
- a good encoder can be reused for classification, retrieval, segmentation, and multimodal learning.

<div style="display: flex; gap: 10px; margin-bottom: 20px;"><a href="https://colab.research.google.com/github/junwei-lu/ai4med/blob/main/codes/ssl/ssl_clip_dinov2.ipynb" target="_blank" style="display: inline-flex; align-items: center; gap: 8px; padding: 10px 16px; background: linear-gradient(135deg, #1565c0 0%, #42a5f5 100%); color: white; border-radius: 8px; text-decoration: none; font-weight: 600; box-shadow: 0 4px 15px rgba(21, 101, 192, 0.4);"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"/></svg>Open Interactive Notebook in Colab</a></div>

We will cover:

- **[Why Train an Encoder First?](./encoder_motivation.md)**
- **[Contrastive Learning and CLIP](./contrastive_learning.md)**
- **[DINOv2: Self-Distillation for Universal Visual Features](./dinov2.md)**
