# Contrastive Learning and CLIP

## The Big Idea

Contrastive learning teaches a model by comparison.

The rule is simple:

- things that belong together should have similar embeddings,
- things that do not belong together should have different embeddings.

That is the whole game.

In the simplest image-only setting, the two matching items are two augmented views of the same image. In CLIP, the matching items are an image and its caption. So CLIP is not a separate idea from contrastive learning. It is a very successful **multimodal implementation** of contrastive learning.

## Start With the Image-Only Version

SimCLR is the standard beginner example.

![SimCLR](./ssl.assets/simclr.png)
> Official SimCLR figure from the project page. Two augmented views of the same image become a positive pair; views from other images act as negatives.

The training recipe is:

1. Take one image $x$.
2. Create two random views $x_i$ and $x_j$ using augmentations.
3. Encode both with the same backbone:

$$
h_i = f_\theta(x_i), \qquad h_j = f_\theta(x_j)
$$

4. Pass them through a projection head:

$$
z_i = g_\phi(h_i), \qquad z_j = g_\phi(h_j)
$$

5. Train so the two views of the same image are close, while views from different images are far apart.

Why add the projection head?

- the loss is applied to $z$,
- the encoder output $h$ is often what we keep for downstream tasks,
- this split usually improves representation quality.

## Why Augmentations Matter

Contrastive learning depends on what we decide should count as "the same example."

If two different crops of the same dog should map nearby, the model must learn the concept of the dog rather than one exact pixel pattern.

Common augmentations include:

- random crop and resize,
- horizontal flip,
- color jitter,
- grayscale conversion,
- blur.

!!! note "Key idea"
    The augmentations define the invariances the encoder is forced to learn.

## The Geometry Intuition

After normalization, you can imagine embeddings as points on a sphere.

For a positive pair:

- pull them together.

For negative pairs:

- push them apart.

Over time, the representation space becomes organized by semantic similarity.

## The Standard Contrastive Loss

The classic SimCLR loss is often called **NT-Xent** or an **InfoNCE-style** loss.

For a positive pair $(i, j)$, define cosine similarity:

$$
\mathrm{sim}(z_i, z_j) = \frac{z_i^\top z_j}{\|z_i\|\|z_j\|}
$$

Then the loss for anchor $i$ is:

$$
\ell_{i,j}
=
- \log
\frac{
\exp(\mathrm{sim}(z_i, z_j)/\tau)
}{
\sum_{k \neq i} \exp(\mathrm{sim}(z_i, z_k)/\tau)
}
$$

where $\tau > 0$ is the temperature.

What this means in plain language:

- reward the correct match,
- compare it against many wrong matches,
- make the correct one win.

### What the Temperature Does

- small $\tau$: sharper competition,
- large $\tau$: softer competition.

## A Minimal SimCLR-Style Template

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def nt_xent_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.T
    sim = sim / temperature

    batch_size = z1.size(0)
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    loss = F.cross_entropy(sim, labels)
    return loss
```

## CLIP Is Contrastive Learning Across Two Modalities

Now replace "two views of the same image" with "an image and the text that describes it."

That is CLIP.

![CLIP diagram](./ssl.assets/clip.png)
> Official CLIP figure from the OpenAI repository. The model jointly learns an image encoder and a text encoder so matching image-text pairs have similar embeddings.

CLIP uses:

- an image encoder $f_\theta(x)$,
- a text encoder $g_\phi(t)$.

For image $x_i$ and caption $t_i$:

$$
v_i = f_\theta(x_i), \qquad u_i = g_\phi(t_i)
$$

These embeddings are normalized:

$$
	ilde{v}_i = \frac{v_i}{\|v_i\|}, \qquad
	ilde{u}_i = \frac{u_i}{\|u_i\|}
$$

Then CLIP computes similarity scores:

$$
s_{ij} = \tilde{v}_i^\top \tilde{u}_j
$$

If image $i$ matches text $i$, then $s_{ii}$ should be high.

## What Changed From SimCLR to CLIP

The training idea stayed the same. The paired views changed.

- SimCLR: image view 1 should match image view 2.
- CLIP: image should match its text.

So you can think of CLIP as a contrastive learner for **modality alignment**.

Instead of learning invariance across crops or colors, it learns alignment across:

- vision,
- language.

That is why CLIP is so useful for multimodal systems.

## The CLIP Loss

Given a batch of $N$ image-text pairs, form the similarity matrix:

$$
S \in \mathbb{R}^{N \times N}, \qquad S_{ij} = \frac{\tilde{v}_i^\top \tilde{u}_j}{\tau}
$$

Then apply cross-entropy in both directions:

$$
\mathcal{L}_{\text{img}}
=
\frac{1}{N}
\sum_{i=1}^N
- \log
\frac{\exp(S_{ii})}{\sum_{j=1}^N \exp(S_{ij})}
$$

$$
\mathcal{L}_{\text{text}}
=
\frac{1}{N}
\sum_{i=1}^N
- \log
\frac{\exp(S_{ii})}{\sum_{j=1}^N \exp(S_{ji})}
$$

$$
\mathcal{L}_{\text{CLIP}} = \frac{1}{2}\left(\mathcal{L}_{\text{img}} + \mathcal{L}_{\text{text}}\right)
$$

This is still a contrastive objective. The only difference is that the competition happens between images and texts in the same batch.

Plain-language interpretation:

- each image should pick its correct caption,
- each caption should pick its correct image.

## Why CLIP Matters

Language is a richer supervision signal than a fixed class ID.

A caption can mention:

- object identity,
- color,
- action,
- style,
- context,
- relationships.

That means the image encoder is pushed to learn concepts that line up with language, not just one narrow label set.

This is why CLIP is often described as learning a **shared embedding space** between images and text.

## Zero-Shot Classification

Once image and text embeddings are aligned, classification becomes a matching problem.

Suppose the candidate labels are:

- "cat"
- "dog"
- "horse"

Write prompts such as:

- "a photo of a cat"
- "a photo of a dog"
- "a photo of a horse"

Encode the prompts with the text encoder, compare them to the image embedding, and choose the most similar one.

This is one reason CLIP felt so powerful. It turned classification into retrieval in a shared embedding space.

## Minimal CLIP Loss Template

```python
import torch
import torch.nn.functional as F


def clip_loss(image_features, text_features, temperature=0.07):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    logits = image_features @ text_features.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i + loss_t)
```

## Training Code

### SimCLR-style image-only training

```python
encoder = backbone.to(device)
projector = ProjectionHead(in_dim=512).to(device)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(projector.parameters()),
    lr=1e-3
)
tau = 0.1

for view1, view2 in train_loader:
    view1, view2 = view1.to(device), view2.to(device)

    z1 = F.normalize(projector(encoder(view1)), dim=-1)
    z2 = F.normalize(projector(encoder(view2)), dim=-1)

    # Similarity matrix over all 2N embeddings
    z = torch.cat([z1, z2])                     # (2B, D)
    sim = z @ z.T / tau                         # (2B, 2B)

    # Mask out self-similarity on the diagonal
    sim.fill_diagonal_(-1e9)

    # Positive labels: view1[i] <-> view2[i]
    B = z1.size(0)
    labels = torch.cat([torch.arange(B) + B,
                        torch.arange(B)]).to(device)

    loss = F.cross_entropy(sim, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### CLIP-style image-text training

```python
tau = 0.07

for images, tokenized_text in train_loader:
    images = images.to(device)
    tokenized_text = tokenized_text.to(device)

    img_emb = F.normalize(image_encoder(images), dim=-1)
    txt_emb = F.normalize(text_encoder(tokenized_text), dim=-1)

    # Cosine similarity matrix scaled by temperature
    logits = img_emb @ txt_emb.T / tau          # (B, B)

    # Diagonal entries are the matching pairs
    labels = torch.arange(logits.size(0), device=device)

    loss = 0.5 * (F.cross_entropy(logits, labels)       # image -> text
                + F.cross_entropy(logits.T, labels))     # text -> image

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### What We Keep After Pretraining

There is one subtle point worth remembering.

- in SimCLR, we usually train on the projection space $z$ but keep encoder features $h$,
- in CLIP, we keep the encoders because the aligned embedding space is itself useful.

Those encoders can then be reused for retrieval, zero-shot classification, search, and multimodal tasks.

### Why This Matters in Medicine

The same contrastive idea extends naturally to medical data.

Examples include:

- chest X-ray + report,
- pathology tile + note,
- retinal image + diagnosis text.

This makes CLIP-style training attractive when paired image-text data are available, because it can build multimodal medical encoders without relying only on handcrafted label taxonomies.



## Summary

Contrastive learning teaches by matching related examples and separating unrelated ones.

- SimCLR does this with two views of the same image.
- CLIP does this with two modalities: image and text.
- The underlying training principle is the same.

So the clean mental model is:

> CLIP is contrastive learning for image-text alignment.

### Strengths

- works without manual labels for every example,
- learns reusable encoders,
- adapts naturally from image-only learning to multimodal alignment,
- supports retrieval and zero-shot prediction.

### Limitations

- performance depends heavily on good augmentations or good pair quality,
- large batches often help,
- captions and reports can be noisy,
- alignment with language does not guarantee clinical correctness.

In medical imaging, this matters because small local findings can be clinically important, and text supervision can still carry bias or ambiguity.

## References and Further Reading

- Chen et al., [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709), *ICML* 2020.
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), *ICML* 2021.
- Official SimCLR page: [SimCLR](https://simclr.github.io/).
- Official SimCLR code: [google-research/simclr](https://github.com/google-research/simclr).
- Official repository: [openai/CLIP](https://github.com/openai/CLIP).
- Hugging Face docs: [CLIP model documentation](https://huggingface.co/docs/transformers/model_doc/clip).
