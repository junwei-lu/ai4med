# Self-Supervised Learning

Self-supervised learning is the art of getting a model to teach itself from raw data before we ask it to solve a specific task. Instead of paying humans to label every image, we design a training signal from the data itself and use that to build a strong **encoder**.

Why this matters:

- labels are expensive,
- unlabeled images are everywhere,
- a good encoder can be reused for classification, retrieval, segmentation, and multimodal learning.

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) [Open In Colab](https://colab.research.google.com/github/junwei-lu/ai4med/blob/main/codes/ssl/ssl_clip_dinov2.ipynb)

We will cover:

- **[Why Train an Encoder First?](./encoder_motivation.md)**
- **[Contrastive Learning and CLIP](./contrastive_learning.md)**
- **[DINOv2: Self-Distillation for Universal Visual Features](./dinov2.md)**
