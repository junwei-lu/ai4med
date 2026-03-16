# Contrastive Learning

## The Core Intuition

Contrastive learning teaches an encoder by comparing examples.

The usual recipe is:

- take one image,
- create two different augmented views of it,
- tell the model these two views should be **close**,
- tell the model views from different images should be **far apart**.

This sounds almost childish, but it turned out to be one of the biggest breakthroughs in self-supervised learning.

![SimCLR](./ssl.assets/simclr.png)
> Official SimCLR figure from the project page. Two augmented views of the same image become a positive pair; views from other images act as negatives.

## SimCLR Pipeline

SimCLR is a clean example of contrastive learning. The steps are:

1. Sample an image $x$.
2. Apply two random augmentations to create $x_i$ and $x_j$.
3. Encode both with the same backbone:

$$
h_i = f_\theta(x_i), \qquad h_j = f_\theta(x_j)
$$

4. Pass them through a projection head:

$$
z_i = g_\phi(h_i), \qquad z_j = g_\phi(h_j)
$$

5. Train with a contrastive loss.

Why the projection head matters:

- the loss is applied on $z$,
- downstream tasks usually use $h$,
- this separation often improves learned representations.

## Why Augmentations Matter So Much

Contrastive learning is really a game of deciding what should count as "the same thing."

If two random crops of the same dog should map nearby, the model must learn dog-ness rather than memorizing one exact pixel layout.

Common augmentations include:

- random crop and resize,
- horizontal flip,
- color jitter,
- grayscale conversion,
- blur.

A good shortcut-free view of contrastive learning is:

!!! note "Key idea"
    The augmentations define the invariances the encoder is forced to learn.

## The Geometry Picture

Think of the encoder output as points on a sphere after normalization.

For a positive pair:

- push them together.

For negatives:

- push them apart.

Over time, the representation space becomes organized by semantic similarity.

## NT-Xent / InfoNCE Loss

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
\sum_{k \neq i}
\exp(\mathrm{sim}(z_i, z_k)/\tau)
}
$$

where $\tau > 0$ is the **temperature**.

### What the Temperature Does

- small $\tau$: sharper competition, stronger emphasis on hard distinctions,
- large $\tau$: softer probabilities.

If you have ever adjusted a softmax temperature, this is the same idea.

## Why the Loss Works

The numerator rewards matching the positive pair.

The denominator asks:

> among all candidates in the batch, can you correctly recognize the matching view?

So the task becomes a kind of self-supervised classification problem where the correct "label" is the paired view.

## Batch Size Matters

In vanilla SimCLR, the other examples in the batch provide negatives. That means:

- bigger batch size,
- more negative examples,
- stronger contrastive signal.

This is one reason early contrastive learning papers often used large batches.

## A Minimal PyTorch Template

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

    z = torch.cat([z1, z2], dim=0)               # [2B, D]
    sim = z @ z.T                                # cosine because already normalized
    sim = sim / temperature

    batch_size = z1.size(0)
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    loss = F.cross_entropy(sim, labels)
    return loss
```

## End-to-End Training Skeleton

```python
encoder = backbone.to(device)
projector = ProjectionHead(in_dim=512).to(device)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(projector.parameters()),
    lr=1e-3
)

for view1, view2 in train_loader:
    view1 = view1.to(device)
    view2 = view2.to(device)

    h1 = encoder(view1)
    h2 = encoder(view2)

    z1 = projector(h1)
    z2 = projector(h2)

    loss = nt_xent_loss(z1, z2, temperature=0.1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## What You Use After Pretraining

Here is a subtle but important point:

- during contrastive pretraining, you optimize the projection space $z$,
- for downstream tasks, you often keep the encoder output $h$.

In practice, people often discard the projection head after training.

## Strengths of Contrastive Learning

- Conceptually simple.
- Works without labels.
- Produces strong transferable encoders.
- Easy to adapt to images, audio, language, or multimodal pairs.

## Limitations

- Often sensitive to augmentations.
- May rely on large batch sizes or memory banks.
- Can overemphasize invariance and suppress fine details if designed poorly.

This last point matters in medical imaging, where tiny local cues may be clinically important.

## Summary

Contrastive learning teaches a model by comparison:

- same underlying example -> close,
- different examples -> far.

That simple geometry reshaped self-supervised learning and set the stage for models like **CLIP**.

## References and Further Reading

- Chen et al., [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709), *ICML* 2020.
- Official SimCLR page: [SimCLR](https://simclr.github.io/).
- Official SimCLR code: [google-research/simclr](https://github.com/google-research/simclr).
