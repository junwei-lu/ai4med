# Image Foundation Models

This chapter introduces **image foundation models** through the lens of the **Vision Transformer (ViT)**. If language models treat a sentence as a sequence of tokens, ViT does something surprisingly similar for images: it turns an image into a sequence of small patches and lets a Transformer reason over them.
<!-- 
![ViT overview](./vision.assets/vit_figure.png)
> Original ViT overview figure from the Google Research `vision_transformer` repository.

Why this chapter matters for AI in medicine:

- Histopathology slides can be split into patches.
- Retinal images and dermoscopy images benefit from transfer learning.
- Chest X-rays and CT slices often start from pretrained visual encoders.
- Modern multimodal medical systems often build on image encoders that descended from ViT. -->

Lectures:

- [ViT Basics: Patches, Tokens, and Encoders](./vit_basics.md)
- [Pretraining Image Foundation Models](./vit_pretraining.md)
- [Fine-Tuning ViT in Practice](./vit_finetuning.md)
- [Multimodal Foundation Models: Mixing Text and Images](./multimodal_foundation_models.md)
- [Interpreting Vision Transformers](./vit_interpretation.md)
