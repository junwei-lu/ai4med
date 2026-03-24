# ViT Basics: Patches, Tokens, and Encoders

<a href="https://colab.research.google.com/github/junwei-lu/ai4med/blob/main/codes/vit/vit_multimodal_cxr_workshop.ipynb#scrollTo=57199061" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; background: linear-gradient(135deg, #1565c0 0%, #42a5f5 100%); color: white; border-radius: 6px; text-decoration: none; font-size: 0.85em; font-weight: 600;">▶ Try in Colab</a>


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

## How Patch Tokenization Actually Works in Code

The phrase “split the image into patches” can sound abstract until you look at tensor shapes.

Suppose a batch of images is stored as a PyTorch tensor with shape:

$$
(B, C, H, W)
$$

where:

- $B$ = batch size
- $C$ = number of channels
- $H, W$ = image height and width

For ViT, patch tokenization usually has two steps:

1. **extract non-overlapping patches**
2. **map each flattened patch to an embedding vector**

### Option 1: Explicitly Extract and Flatten Patches

This version is useful for understanding the mechanics.

```python
import torch

# Example batch: 2 RGB images of size 224 x 224
x = torch.randn(2, 3, 224, 224)

patch_size = 16

# Step 1: split image into non-overlapping patches
# Output shape after unfold:
# (B, C, H/P, W/P, P, P)
patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

# Rearrange so each patch becomes one item in the sequence
# (B, H/P, W/P, C, P, P)
patches = patches.permute(0, 2, 3, 1, 4, 5)

# Flatten each patch
# (B, N, P*P*C)
patches = patches.reshape(x.shape[0], -1, patch_size * patch_size * x.shape[1])

print(patches.shape)  # torch.Size([2, 196, 768])
```

Why `768` in the last dimension?

$$
16 \times 16 \times 3 = 768
$$

So before any learned projection, each patch is just a flattened vector of raw pixel values.

!!! note "What this tensor means"
    `torch.Size([2, 196, 768])` means: 2 images, 196 patches per image, and 768 numbers per patch.

### Option 2: The Standard ViT Trick with a Convolution

In real implementations, patch extraction and linear projection are usually merged into **one convolutional layer**.

```python
import torch
import torch.nn as nn

x = torch.randn(2, 3, 224, 224)

patch_size = 16
embed_dim = 768

# kernel_size = stride = patch_size
# This creates non-overlapping patches and projects each one to embed_dim
patch_embed = nn.Conv2d(
    in_channels=3,
    out_channels=embed_dim,
    kernel_size=patch_size,
    stride=patch_size,
)

y = patch_embed(x)
print(y.shape)  # torch.Size([2, 768, 14, 14])

# Flatten spatial grid into a token sequence
tokens = y.flatten(2).transpose(1, 2)
print(tokens.shape)  # torch.Size([2, 196, 768])
```

This is the same token sequence structure as before, but now each patch has already been projected into a learned embedding space of dimension `768`.

That is why many ViT implementations describe patch embedding as a **linear projection of flattened patches**, even though the code often uses `Conv2d`.

Mathematically, these views are equivalent:

- flatten patch, then apply a linear layer
- apply a convolution with `kernel_size = stride = patch_size`

### A Minimal Patch Embedding Module

Here is a compact version of the patch tokenizer you will see in many ViT implementations:

```python
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)              # (B, D, H/P, W/P)
        x = x.flatten(2)              # (B, D, N)
        x = x.transpose(1, 2)         # (B, N, D)
        return x


x = torch.randn(2, 3, 224, 224)
tokenizer = PatchEmbedding(in_channels=3, patch_size=16, embed_dim=768)
tokens = tokenizer(x)

print(tokens.shape)  # torch.Size([2, 196, 768])
```

This output is what the Transformer encoder expects: a batch of token sequences.

### Adding the Class Token and Position Embeddings

Patch tokens alone are not the final input to ViT. We still need to:

1. prepend a learnable class token
2. add positional embeddings

```python
import torch
import torch.nn as nn

B, N, D = 2, 196, 768
tokens = torch.randn(B, N, D)

cls_token = nn.Parameter(torch.randn(1, 1, D))
pos_embed = nn.Parameter(torch.randn(1, N + 1, D))

cls_tokens = cls_token.expand(B, -1, -1)   # (B, 1, D)
x = torch.cat([cls_tokens, tokens], dim=1) # (B, N+1, D)
x = x + pos_embed                           # (B, N+1, D)

print(x.shape)  # torch.Size([2, 197, 768])
```

Now the image has been converted into the exact kind of sequence a Transformer can process.

### Why This Matters in Medical Imaging

Patch tokenization is not just an implementation detail. It determines what information is available to the model from the very beginning.

- If the patch size is too large, tiny abnormalities may be merged into a single coarse token.
- If the patch size is small, subtle local structure is preserved better, but the token sequence becomes longer and attention becomes more expensive.
- If images are grayscale, then `C = 1`, so the raw flattened patch dimension becomes $P^2$ rather than $P^2 \cdot 3$.

For example, for a grayscale MRI slice of size `224 x 224` with patch size `16 x 16`:

$$
16 \times 16 \times 1 = 256
$$

So each raw patch starts as a 256-dimensional vector before projection into the model embedding space.

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
