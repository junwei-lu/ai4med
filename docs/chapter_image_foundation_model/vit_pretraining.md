# Pretraining Image Foundation Models

<a href="https://colab.research.google.com/github/junwei-lu/ai4med/blob/main/codes/vit/vit_multimodal_cxr_workshop.ipynb#scrollTo=fff0bc8b" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; background: linear-gradient(135deg, #1565c0 0%, #42a5f5 100%); color: white; border-radius: 6px; text-decoration: none; font-size: 0.85em; font-weight: 600;">▶ Try in Colab</a>

## Why Pretraining Matters So Much

A randomly initialized ViT usually does **not** outperform a well-tuned CNN on modest datasets. The magic happens when we first expose the model to a very large number of images and only later adapt it to a specific task.

That two-stage recipe is the heart of an image foundation model:

1. Learn broad visual features from a large source dataset.
2. Reuse those features for smaller downstream tasks.

This is especially useful in medicine, where labeled datasets are often expensive and small.

## The Original ViT Recipe

The original ViT paper used large-scale supervised pretraining and then fine-tuned on smaller benchmarks.

The basic training target was simple:

- input an image,
- predict its class label,
- repeat across millions of images.

This may sound ordinary, but scale changed everything. Once trained on large enough datasets, ViT learned features that transferred extremely well.

## What the Model Learns During Pretraining

During pretraining, the model gradually learns:

- low-level cues such as color and texture,
- mid-level structures such as edges, parts, and repeated motifs,
- higher-level semantic patterns such as fur, wheels, organs, lesions, or tissue organization.

Early layers are often more local and generic. Later layers usually become more semantic and task-aware.

## Supervised vs Self-Supervised Pretraining

Today, image foundation models are often grouped into two families.

### 1. Supervised Pretraining

This is the original ViT setup.

- Requires labels.
- Works very well when large curated datasets are available.
- Still common for classification backbones.

### 2. Self-Supervised Pretraining

This avoids relying entirely on labels. The model creates a learning signal from the image itself.

Popular strategies include:

- **Masked image modeling**: hide many patches and reconstruct the missing content.
- **Contrastive learning**: make two views of the same image similar and different images dissimilar.
- **Self-distillation**: train a student network to match a teacher network across augmentations.

These approaches are extremely important in domains like medical imaging, remote sensing, and microscopy where labels are limited but unlabeled images are abundant.

## Masked Autoencoders (MAE)

One of the most influential extensions of ViT is the **Masked Autoencoder (MAE)**.

The idea is beginner friendly:

- remove a large fraction of patches, often around 75%,
- run the encoder only on the visible patches,
- ask a lightweight decoder to reconstruct the missing ones.

This forces the encoder to learn meaningful structure instead of memorizing pixels.

Why MAE is attractive:

- it uses unlabeled images,
- it is computationally efficient because the encoder sees only visible patches,
- it transfers well to downstream tasks.

For medical imaging, this is a natural fit because hospitals may have millions of scans but only a small subset with expert labels.

## A Simple Mental Model

If a language model learns by reading many sentences, MAE learns by solving a visual fill-in-the-blanks puzzle:

- "I can only see some patches."
- "Can I infer the missing structure?"

A model that gets good at that task often develops a strong internal understanding of objects, shapes, and spatial organization.

## Template Code: Self-Supervised Patch Masking

The code below shows the *idea* of random patch masking, without implementing a full MAE:

```python
import torch

def random_patch_mask(x, mask_ratio=0.75):
    """
    x: [batch, num_patches, dim]
    returns visible patches and mask indices
    """
    B, N, D = x.shape
    num_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :num_keep]
    x_visible = torch.gather(
        x,
        dim=1,
        index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
    )

    mask = torch.ones(B, N)
    mask[:, :num_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_visible, mask, ids_restore
```

## What Makes Image Pretraining Different from NLP Pretraining

There is an analogy with language models, but also some differences:

- Image patches are less semantically clean than words.
- Spatial structure matters much more.
- Augmentation matters a lot: crops, flips, color jitter, and resizing affect what the model learns.
- Resolution matters because more pixels create more patch tokens.

## Common Pretraining Datasets

Historically, important datasets included:

- **ImageNet-1k**
- **ImageNet-21k**
- **JFT-300M** (internal to Google)

More recent visual encoders often use much larger private or curated web-scale image collections.

In healthcare, institutions often build internal pretraining corpora from pathology tiles, retinal images, radiology frames, or multimodal image-text pairs.

## Summary

Pretraining is what turns ViT from "a Transformer for images" into a **foundation model**.

- With enough data, the model learns broadly useful visual features.
- With self-supervised learning, it can learn from unlabeled image collections.
- With transfer learning, those features become valuable in smaller specialized domains.

## References and Further Reading

- Dosovitskiy et al., [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/forum?id=YicbFdNTTy), *ICLR* 2021.
- He et al., [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), *CVPR* 2022.
- Hugging Face, [Vision Transformer (ViT) documentation](https://huggingface.co/docs/transformers/en/model_doc/vit).
