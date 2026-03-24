# DINO: Self-Distillation

<a href="https://colab.research.google.com/github/junwei-lu/ai4med/blob/main/codes/ssl/ssl_clip_dinov2.ipynb#scrollTo=md_19" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; background: linear-gradient(135deg, #1565c0 0%, #42a5f5 100%); color: white; border-radius: 6px; text-decoration: none; font-size: 0.85em; font-weight: 600;">▶ Try in Colab</a>

If contrastive learning says:

> "pull positives together, push negatives apart,"

DINO-style learning says:

> "make a student network match a teacher network across different views."

DINOv2 is one of the strongest examples of this idea at scale. It trains a powerful visual encoder without manual labels and produces features that transfer well to many tasks:

- classification,
- retrieval,
- segmentation,
- depth estimation,
- dense patch matching.

![dino](ssl.assets/dino_collapse.png)


## The Teacher-Student Setup

The key actors are:

- a **student** network,
- a **teacher** network,
- multiple crops or views of the same image.

Both networks process different augmentations of the same image. The student is trained to match the teacher outputs.

The teacher is not trained by backpropagation in the usual way. Instead, it is updated as an **exponential moving average (EMA)** of the student:

$$
\theta_{\text{teacher}}
\leftarrow
\lambda \theta_{\text{teacher}}
+
(1-\lambda)\theta_{\text{student}}
$$

This makes the teacher a slowly moving target, which stabilizes training.

### Multi-Crop Intuition

One image can generate:

- a couple of large global crops,
- several small local crops.

The model learns that these all describe the same underlying scene or object. This encourages invariance across scale, viewpoint, and partial observation.


## DINO Loss

Let:

- $p_t(x)$ be the teacher output distribution,
- $p_s(x)$ be the student output distribution.

DINO uses a cross-entropy-like loss:

$$
\mathcal{L}_{\text{DINO}}
=
- \sum_k p_t^{(k)} \log p_s^{(k)}
$$

where the teacher target is treated as fixed for that step.

## Avoid Collapse

A beginner might worry:

> If the student just copies the teacher, couldn’t both output the same boring constant vector forever?

Yes, that is exactly the collapse problem. From the figure below, we can get the higher loss by just simply predicting the singular probability.

![dino](ssl.assets/dino_sig.png)

DINO-style methods avoid it using a combination of:

- multi-crop views,
- teacher EMA updates,
- centering,
- sharpening,

These ingredients make the teacher signal informative enough to organize the representation space instead of flattening it. 



We have introduced the first two. Let us introduce the later two important stabilizers to avoid collapsing:

### 1. Teacher Centering

The teacher logits are centered before softmax to avoid collapse:

$$
\tilde{q}_t = q_t - c
$$

where $c$ is a running center.

### 2. Sharpening

The teacher distribution uses a lower temperature, which makes it sharper:

$$
p_t = \mathrm{softmax}\left(\frac{\tilde{q}_t}{T_t}\right)
$$

The student uses its own temperature:

$$
p_s = \mathrm{softmax}\left(\frac{q_s}{T_s}\right)
$$

Then the student is trained to match the teacher.



<!-- 
## DINOv2 Training Objective at a High Level

The DINOv2 model card summarizes the training objective as:

- **DINO self-distillation loss with multi-crop**
- **iBOT masked-image modeling loss**
- **KoLeo regularization on class tokens**

For a beginner, it is useful to separate these roles.

### DINO Self-Distillation Loss

Align global semantic representations across views.

### iBOT Patch Loss

Encourage patch-level understanding, not just a good global class token.

### KoLeo Regularization

Encourage feature usage to spread out more evenly in representation space instead of bunching up.

You do not need every detail memorized. The big idea is:

!!! note "Big picture"
    DINOv2 learns both global meaning and local patch structure, then scales this recipe with strong ViTs and high-quality data.

## Simplified Mathematical View

Let the student and teacher outputs on two views of the same image be:

$$
q_s = h_{\theta}(x_s), \qquad q_t = h_{\xi}(x_t)
$$

After centering and temperature scaling:

$$
p_t = \mathrm{softmax}\left(\frac{q_t - c}{T_t}\right), \qquad
p_s = \mathrm{softmax}\left(\frac{q_s}{T_s}\right)
$$

The self-distillation loss is:

$$
\mathcal{L}_{\text{DINO}}
=
- \sum_k p_t^{(k)} \log p_s^{(k)}
$$

If there are multiple views, sum this across view pairs. -->

## DINO Code

Here is a sample code to define DINO loss.

```python
import torch
import torch.nn.functional as F


def dino_loss(student_logits, teacher_logits, center, student_temp=0.1, teacher_temp=0.04):
    student_probs = F.log_softmax(student_logits / student_temp, dim=-1)
    teacher_probs = F.softmax((teacher_logits - center) / teacher_temp, dim=-1)
    loss = -(teacher_probs * student_probs).sum(dim=-1).mean()
    return loss
```

This is simplified, but it captures the central pattern:

- teacher produces a probability target,
- student matches it with cross-entropy,
- center and temperatures help stabilize learning.

### EMA Teacher Update

```python
@torch.no_grad()
def update_teacher(student, teacher, momentum=0.996):
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1 - momentum)
```

This is one of the most elegant ideas in self-supervised learning: the teacher improves by being a moving average of the student rather than a separately trained network.

### Simplified Training Skeleton

```python
for global_views, local_views in train_loader:
    global_views = [v.to(device) for v in global_views]
    local_views = [v.to(device) for v in local_views]

    with torch.no_grad():
        teacher_out_1 = teacher(global_views[0])
        teacher_out_2 = teacher(global_views[1])

    student_views = global_views + local_views
    student_outs = [student(v) for v in student_views]

    loss = 0.0
    for s_out in student_outs:
        loss += dino_loss(s_out, teacher_out_1, center)
        loss += dino_loss(s_out, teacher_out_2, center)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    update_teacher(student, teacher)
```

This hides many engineering details, but the conceptual loop is correct.
<!-- 
## Why DINOv2 Features Are So Useful

DINOv2 features are especially attractive because they work well without much fine-tuning.

In practice, people often use them for:

- k-NN classification,
- linear probing,
- retrieval,
- dense correspondence,
- segmentation or depth heads.

That is a sign of a genuinely strong encoder: not just one benchmark win, but broad transfer.

## DINOv2 and ViTs

DINOv2 is built around Vision Transformers. This is important because:

- patch tokens support dense tasks,
- the class token supports image-level tasks,
- large-scale ViT pretraining produces reusable representations.

The model card also notes variants with **register tokens**, which are extra learned tokens added to stabilize or improve representation behavior.

## Why This Matters in Practice

If CLIP is your tool when you want image-text alignment, DINOv2 is your tool when you want a very strong **general-purpose visual encoder**.

That is why many people think of it as a foundation model for visual features.

## Summary

DINOv2 teaches a student encoder using a slowly moving teacher and multiple views of the same image.

- It does not rely on labels.
- It does not need explicit negative pairs in the SimCLR sense.
- It scales into a strong universal visual backbone.

That makes it one of the clearest examples of how self-supervised learning can train an encoder first and reuse it almost everywhere later. -->

## References and Further Reading

- Caron et al., [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294), 2021.
- Oquab et al., [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193), 2023.
- Official repository: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2).
- Official demo page: [DINOv2](https://dinov2.metademolab.com/).
