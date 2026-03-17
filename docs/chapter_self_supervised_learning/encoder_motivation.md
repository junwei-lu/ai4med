# Encoder Training


An encoder turns a raw input $x$ into a useful representation:

$$
z = f_\theta(x)
$$

Instead of training a new model from scratch for every task, we often train one strong encoder first and reuse it many times.

In practice, an encoder can support:

- representation learning,
- classification or regression,
- retrieval and nearest-neighbor search,
- segmentation or detection with task-specific heads,
- modality alignment such as image-text matching.

This is why encoder pretraining matters so much: once the representation is good, many downstream tasks become easier.

## Why Pretrain an Encoder?

Labels are expensive, but raw data is abundant. In medicine, this gap is especially large: hospitals may store millions of images, while only a small fraction have reliable expert annotations.

Self-supervised learning uses the raw data itself as the training signal. The goal is to learn features that are:

- stable under small irrelevant changes,
- different for genuinely different content,
- useful across many tasks.

That is the basic foundation-model idea for vision: learn general features first, specialize later.


After pretraining, we can freeze the encoder and test it with a small classifier, or fine-tune it for a specific task.

### Why This Matters in Medicine

This is especially useful in medical AI because:

- unlabeled data is common,
- labels are expensive and sometimes noisy,
- a single encoder may be reused across tasks,
- transfer across scanners, hospitals, and modalities matters.

An encoder pretrained on large image collections can later support diagnosis, retrieval, report alignment, or fine-tuning on a small labeled dataset.

## High-Level Idea of Self-Supervised Learning

Self-supervised learning creates supervision from the data itself. Instead of asking for a human label, we build an artificial target from the same sample.

For example, we may take one image, create two augmented views, and ask the model to recognize that they came from the same underlying object. In that sense, self-supervision often means making a sample into its own positive example.

The exact target can vary: match two views, recover masked content, predict a teacher output, or align two modalities. But the common idea is simple: the data teaches the encoder how to represent itself.

## PCA as a Simple Self-Supervised Encoder

Even Principal Component Analysis can be viewed as a simple self-supervised method.

If $X$ is a centered data matrix, then

$$
XX^\top = U D U^\top
$$

is an eigendecomposition of the sample-similarity matrix $XX^\top$. Two samples have a large value in $XX^\top$ when their inner product is large, so this matrix captures which samples are similar.

The top columns of $U$ give a low-dimensional representation that preserves the strongest structure in the data. In that sense, PCA learns a linear encoder from unlabeled data: it uses the similarity structure already present in $X$, without any external labels.

## Self-Supervised Training Methods

A clean way to organize self-supervised training methods is by the target the encoder is asked to match.


### Reconstruction and Masked Modeling Methods

Core idea: hide part of the input and train the encoder to recover what is missing. We have introduced this idea in [masked language model](../chapter_foundation_model/snp_training.md#masked-language-modeling-mlm).

Typical methods: [**MAE**](https://arxiv.org/abs/2111.06377), [**BEiT**](https://arxiv.org/abs/2106.08254), [**iBOT**](https://arxiv.org/abs/2111.07832) (hybrid: masking + self-distillation).

Simple memory aid: recover what was hidden.

![mlm](ssl.assets/mlm_bear.png)

### Contrastive Methods

Core idea: make matching views close and non-matching examples far apart. We will introduce this method later [here](contrastive_learning.md).

This is the classic positive-pair and negative-pair setup.

Typical methods: [**SimCLR**](https://arxiv.org/abs/2002.05709), [**MoCo**](https://arxiv.org/abs/1911.05722), [**CLIP**](https://arxiv.org/abs/2103.00020) (multimodal contrastive, image-text alignment).

Simple memory aid: same image close, different images apart.

![clip](ssl.assets/clip_bear.png)

### Self-Distillation

Core idea: make two views match without using explicit negative pairs.

These methods avoid collapse through asymmetry, such as a teacher-student setup, stop-gradient, or predictor heads.
We will introduce this method later [here](dinov2.md).

Typical methods: [**BYOL**](https://arxiv.org/abs/2006.07733), [**SimSiam**](https://arxiv.org/abs/2011.10566), [**DINO**](https://arxiv.org/abs/2104.14294).

Simple memory aid: match two views without negatives.

![dino](ssl.assets/dino_collapse.png)

### Clustering and Prototype Methods

Core idea: map features to shared prototypes or cluster assignments, then make different views agree on those assignments.

Typical methods: [**SwAV**](https://arxiv.org/abs/2006.09882), [**DeepCluster**](https://arxiv.org/abs/1807.05520).

Simple memory aid: learn through stable pseudo-labels.

### Redundancy-Reduction Methods

Core idea: make views agree while also encouraging different feature dimensions to carry different information.

Typical methods: [**Barlow Twins**](https://arxiv.org/abs/2103.03230), [**VICReg**](https://arxiv.org/abs/2105.04906).

Simple memory aid: match views, but do not let features collapse into copies of each other.


### Hybrid Methods

Some influential methods combine multiple ideas.

- [**DINO**](https://arxiv.org/abs/2104.14294) is mostly self-distillation, but also has prototype-like behavior.
- [**iBOT**](https://arxiv.org/abs/2111.07832) combines masked modeling and self-distillation.
- [**CLIP**](https://arxiv.org/abs/2103.00020) extends contrastive learning to multiple modalities.

The exact boundaries are not always strict, but this taxonomy is a useful mental map.


| Family | Core question | Representative methods |
| --- | --- | --- |
| Contrastive | Which views should be close, and which should be far? | [SimCLR](https://arxiv.org/abs/2002.05709), [MoCo](https://arxiv.org/abs/1911.05722), [CLIP](https://arxiv.org/abs/2103.00020) |
| Non-contrastive / self-distillation | How can two views match without negatives? | [BYOL](https://arxiv.org/abs/2006.07733), [SimSiam](https://arxiv.org/abs/2011.10566), [DINO](https://arxiv.org/abs/2104.14294) |
| Clustering / prototypes | Can two views get the same prototype? | [SwAV](https://arxiv.org/abs/2006.09882), [DeepCluster](https://arxiv.org/abs/1807.05520) |
| Redundancy reduction | Can views match while features stay non-redundant? | [Barlow Twins](https://arxiv.org/abs/2103.03230), [VICReg](https://arxiv.org/abs/2105.04906) |
| Reconstruction / masked modeling | Can the model infer what was hidden? | [MAE](https://arxiv.org/abs/2111.06377), [BEiT](https://arxiv.org/abs/2106.08254) |
| Hybrid | Can we combine masking, distillation, or prototypes? | [iBOT](https://arxiv.org/abs/2111.07832), DINO-style variants |



## References and Further Reading

- Chen et al., [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709), *ICML* 2020.
- He et al., [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722), *CVPR* 2020.
- Grill et al., [Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733), *NeurIPS* 2020.
- Caron et al., [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294), *ICCV* 2021.
- He et al., [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), *CVPR* 2022.
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), *ICML* 2021.
