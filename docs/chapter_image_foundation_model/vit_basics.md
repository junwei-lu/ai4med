# ViT Basics: Patches, Tokens, and Encoders

## Why ViT Was a Big Deal

For a long time, computer vision was dominated by **convolutional neural networks (CNNs)**. CNNs were designed with images in mind: they look at local neighborhoods, share filters across space, and gradually build up larger receptive fields.

The Vision Transformer changed the story by asking a simple question:

> What if we stop hand-designing image-specific operations and instead feed the image to a standard Transformer?

The key trick is to avoid treating every pixel as a token. That would be far too expensive. Instead, ViT groups pixels into **patches**, and each patch becomes one token.

## From Image to Sequence

Suppose the input image has shape $H \times W \times C$ and we use square patches of size $P \times P$.

- Each patch contains $P^2 \cdot C$ numbers.
- The number of patches is

$$
N = \frac{H W}{P^2}
$$

- After flattening each patch, we project it into an embedding vector of dimension $D$.

This gives a token sequence of length $N$, just like a sentence with $N$ words.

### A Concrete Example

For a `224 x 224` RGB image with patch size `16 x 16`:

- `H = 224`, `W = 224`, `C = 3`
- `P = 16`
- Number of patches: `(224 / 16) * (224 / 16) = 14 * 14 = 196`

So ViT turns the image into **196 visual tokens**.

## The Core Architecture

![ViT architecture](./vision.assets/vit_figure.png)
> ViT splits the image into patches, embeds them, adds position information, and feeds them through a standard Transformer encoder.

The usual ViT pipeline is:

1. Split image into non-overlapping patches.
2. Flatten each patch and linearly project it into an embedding.
3. Add a learnable **position embedding** to each patch embedding.
4. Prepend a learnable **class token** (`[CLS]`).
5. Pass the full sequence through a stack of Transformer encoder blocks.
6. Use the final `[CLS]` representation for classification.

## Why Positional Embeddings Matter

A Transformer does not automatically know that one patch came from the top-left corner and another came from the bottom-right. Without extra information, it only sees a bag of tokens.

That is why ViT adds **positional embeddings**. These tell the model where each patch came from, so the sequence still remembers spatial layout.

## How ViT Differs from CNNs

| Idea | CNN | ViT |
|---|---|---|
| Basic unit | Local convolution window | Patch token |
| Built-in spatial bias | Strong | Weak |
| Global context | Built gradually | Available through self-attention |
| Data hunger | Lower | Often higher |
| Transfer from large pretraining | Good | Excellent |

### Intuition

CNNs come with strong assumptions about images. That helps when data is limited.

ViT starts with fewer assumptions. This sounds risky, but it becomes powerful when the model is pretrained on a large image collection. In other words:

!!! note "Key intuition"
    CNNs know more about images before training starts. ViT learns more from data once scale becomes large enough.

## Patch Size Is a Real Design Choice

Patch size controls the trade-off between detail and compute:

- **Smaller patches** keep more local detail, but create longer sequences and more attention cost.
- **Larger patches** are cheaper, but may throw away fine structure.

This matters in medical imaging. Tiny lesions, vessel boundaries, and cellular morphology may disappear if patches are too coarse.

## Minimal Hugging Face Example

The Hugging Face `transformers` library makes it easy to load a pretrained ViT.

```python
import torch
from PIL import Image
import requests
from transformers import AutoImageProcessor, AutoModelForImageClassification

model_name = "google/vit-base-patch16-224"

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

pred_class = logits.argmax(dim=-1).item()
print(model.config.id2label[pred_class])
```

## When ViT Works Especially Well

ViT shines when:

- you have a lot of pretraining data,
- you want to reuse a pretrained encoder across many tasks,
- you care about global context,
- you want an architecture that plugs naturally into multimodal systems.

That last point is especially important. CLIP, DINOv2, Segment Anything, and many medical vision pipelines all build on the same general idea: strong patch-based visual encoders that can be adapted downstream.

## Summary

ViT is not magic. It simply recasts image understanding as a **sequence modeling** problem:

- patches become tokens,
- attention becomes the mechanism for mixing information,
- pretraining becomes the source of general visual knowledge.

Once that idea clicks, the rest of the ecosystem becomes much easier to understand.

## References and Further Reading

- Dosovitskiy et al., [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/forum?id=YicbFdNTTy), *ICLR* 2021.
- Hugging Face, [Vision Transformer (ViT) documentation](https://huggingface.co/docs/transformers/en/model_doc/vit).
- Google Research, [Vision Transformer repository](https://github.com/google-research/vision_transformer).
