# Multimodal Foundation models

<a href="https://colab.research.google.com/github/junwei-lu/ai4med/blob/main/codes/vit/vit_multimodal_cxr_workshop.ipynb#scrollTo=48df8023" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; background: linear-gradient(135deg, #1565c0 0%, #42a5f5 100%); color: white; border-radius: 6px; text-decoration: none; font-size: 0.85em; font-weight: 600;">▶ Try in Colab</a>

The previous tutorials introduced ViT as a way to treat images like sequences. The natural next question is: what happens when we want a single model that handles **both** images and text in the same sequence?

This is the goal of **multimodal foundation models**. Instead of building separate systems for each modality and bolting them together, these models learn a unified representation over interleaved text and image content from the start.

The clinical motivation is clear. A pathology report is not just text and not just an image. A radiology workflow involves reading a scan and writing a finding. A clinical decision support system benefits from reasoning jointly over lab values, notes, and medical images. Models that can natively mix modalities are a natural fit for these workflows.

## Late Fusion vs Early Fusion

Before diving into architecture, it helps to understand the two main design philosophies for multimodal models.

### Late Fusion

Late-fusion models process each modality with its own specialized encoder and combine information only at a later stage. Models like Flamingo, LLaVA, and IDEFICS follow this pattern. A pretrained vision encoder produces image features, a pretrained language model processes text, and a bridging module (such as a cross-attention layer or a linear projection) connects them.

Late fusion has an important advantage: you can reuse strong unimodal checkpoints directly. But it also has a limitation. Because the modalities live in separate spaces until late in the pipeline, the model's ability to integrate information across modalities is constrained by the design of the bridging module.

### Early Fusion

Early-fusion models project all modalities into a shared token space from the very beginning and process them together through a single transformer. This is the approach taken by models like Chameleon (Meta FAIR, 2024).

The key idea is straightforward: if we can convert images into discrete tokens, we can interleave them with text tokens and feed the combined sequence into a standard autoregressive transformer. The model never needs to distinguish between modalities at the architectural level. It simply predicts the next token, whether that token represents a word or a patch of an image.

This early-fusion design allows for seamless reasoning across modalities and enables entirely new capabilities, such as generating documents that freely interleave text and images in any order.

![fusion](vision.assets/fusion.png)

| Aspect | Late Fusion (e.g., LLaVA, Flamingo) | Early Fusion (e.g., Chameleon) |
|---|---|---|
| Modality encoders | Separate image and text encoders | Single shared transformer |
| When modalities interact | Late, through a bridging module | From the first layer |
| Can generate images natively | No (needs external generator) | Yes |
| Can leverage pretrained unimodal models | Yes, directly | No, must train from scratch |
| Training stability | Generally easier | Requires special techniques (QK-Norm, z-loss) |
| Interleaved generation | Limited | Native capability |

Neither approach is strictly better. Late-fusion models benefit from strong pretrained components and are easier to train. Early-fusion models offer tighter integration and novel generation capabilities but require more careful optimization.

## Architecture: How Text and Image Tokens Share a Single Transformer

![Chameleon architecture](./vision.assets/mmdal_arch.png)
> **Figure: Early-fusion mixed-modal architecture.** On the left, during pre-training, text prompts and images are tokenized into a single interleaved sequence and fed to a unified autoregressive transformer. On the right, at generation time, the same model produces both text tokens and image tokens, which are then decoded back into pixels by an image de-tokenizer. Text tokens are shown in green; image tokens in blue. *(Source: Chameleon, Meta FAIR 2024)*

The architecture of an early-fusion multimodal model like Chameleon can be understood as three components working together: a tokenizer for each modality, a shared vocabulary, and a single transformer backbone.

### Image Tokenization

To feed images into a text-like transformer, we first need to convert them into discrete tokens. This is done with an **image tokenizer**, typically a learned model based on a VQ-VAE (vector-quantized variational autoencoder).

The image tokenizer works as follows:

- Take a $512 \times 512$ image as input.
- Encode it into 1024 discrete tokens, each drawn from a codebook of size 8192.
- Each token represents a spatial region of the image, analogous to how a BPE token represents a piece of a word.

Special sentinel tokens mark image boundaries. A `<start_image>` token signals that the following tokens are image content, and an `<end_image>` token marks the transition back to text. This lets the model know when it is reading or generating visual content.

### Text Tokenization

Text is tokenized using a standard BPE tokenizer. The key design choice is that the text vocabulary and the image codebook are merged into a single unified vocabulary. In Chameleon, this combined vocabulary has 65,536 entries: the standard BPE subwords plus the 8192 image codebook tokens.

### The Unified Sequence

Once both modalities are tokenized, a training example might look like this:

```
[text tokens] [<start_image>] [image token 1] ... [image token 1024] [<end_image>] [text tokens] ...
```

The transformer sees this as one flat sequence. It does not need separate encoder or decoder branches for each modality. Self-attention operates over all tokens uniformly, allowing image tokens to attend to text tokens and vice versa.

This is a direct generalization of the patch-to-token idea from ViT. In ViT, image patches become tokens for a vision-only transformer. Here, image patches become tokens that live alongside word tokens in a language-model-style transformer.

### Architectural Details

The transformer backbone largely follows the LLaMA-2 design:

- **Normalization**: RMSNorm for layer normalization.
- **Activation**: SwiGLU in the feed-forward layers.
- **Position encoding**: Rotary Positional Embeddings (RoPE).
- **Grouped Query Attention (GQA)** for the larger model variants.

One important deviation from standard LLaMA is the placement of normalization layers. In the 34B-parameter variant, normalization is applied *after* the attention and feed-forward operations (rather than before), following the Swin Transformer convention. This seemingly small change turns out to be important for training stability, as we discuss next.

## Training: Making Early Fusion Work at Scale

Training a model that mixes text and image tokens in a single transformer sounds elegant, but in practice it introduces serious optimization challenges. The Chameleon paper documents several techniques that were essential for stable training.

<!-- ![Training stability ablations](./vision.assets/chameleon_training_stability.png)
> **Figure: Training stability ablations.** (a) Output norm growth over training steps: without QK-Norm or dropout, norms grow uncontrollably, predicting future loss divergence. (b) Training loss with and without QK-Norm on the 7B model — without it, the loss spikes and diverges after ~125k steps. (c) Training loss with and without dropout — dropout provides additional stabilization. *(Source: Chameleon, Meta FAIR 2024)* -->

### The Stability Problem

When a single set of weights is shared across modalities, the different modalities effectively compete for representational capacity. Each modality tends to slowly push its internal norms higher during training. Because the softmax function is translation-invariant ($\text{softmax}(z) = \text{softmax}(z + c)$), this norm growth is invisible at first. But once the norms grow large enough to exceed the effective range of bfloat16 arithmetic, the training loss diverges.

This problem does not appear in text-only training. It is specifically triggered by the mixed-modal setting, and it can surface very late, sometimes after 20--30% of training has completed.

### Query-Key Normalization (QK-Norm)

The first mitigation is **QK-Norm**: applying layer normalization to the query and key vectors inside the attention mechanism before computing attention scores. This directly bounds the magnitude of the dot products entering the softmax, preventing the uncontrolled norm growth that leads to divergence.

### Dropout and Norm Reordering

For the 7B model, adding dropout (rate 0.1) after the attention and feed-forward layers, in combination with QK-Norm, was sufficient for stability. For the larger 34B model, an additional change was needed: reordering the normalization to appear *after* each sublayer (post-norm style) rather than before it. This bounds the norm growth of the feed-forward block, which is particularly important given the multiplicative nature of SwiGLU.

### Z-Loss Regularization

QK-Norm addresses the softmax inside attention, but the final output softmax over the vocabulary can still suffer from logit drift. To address this, a small **z-loss** term is added to the training objective. This regularizes the partition function of the output softmax, keeping logits well-behaved.

The z-loss is defined as $10^{-5} \cdot \log^{2} Z$, where $Z = \sum_{i} e^{z_i}$ is the partition function. This is a lightweight addition to the loss that provides substantial training stability benefits.
<!-- 
![Training loss curves](./vision.assets/chameleon_training_curves.png)
> **Figure: Training loss curves at scale.** (a) Full 600k-step training curves for both the 7B and 34B models over mixed-modal data. (b) Training with image generation disabled does not suffer from instability, confirming the divergence is specific to the mixed-modal setting. (c) For the 34B model, dropout alone does not prevent divergence — norm reordering is additionally required. *(Source: Chameleon, Meta FAIR 2024)* -->

### Pre-Training Data

Pre-training proceeds in two stages over a massive data mixture:

- **Text-only data**: approximately 2.9 trillion tokens from a combination of the LLaMA-2 and CodeLLaMA corpora.
- **Text-image pairs**: 1.4 billion image-text pairs, resized and center-cropped to $512 \times 512$, producing about 1.5 trillion text-image tokens.
- **Interleaved text and image data**: 400 billion tokens of web documents containing naturally interleaved text and images.

The first stage (80% of training) uses this full unsupervised mixture. The second stage (20% of training) lowers the weight of first-stage data by 50% and mixes in higher-quality datasets, including a filtered subset of instruction-tuning sets.

In total, the model sees approximately 9.2 trillion tokens across 2.1 epochs. The 7B model is trained on 1024 NVIDIA A100 GPUs; the 34B model on 3072.

### Alignment via Supervised Fine-Tuning

After pre-training, a lightweight alignment stage uses supervised fine-tuning (SFT) on curated data across several categories: text, code, visual chat, image generation, interleaved text/image generation, and safety. Balancing the modality distribution during SFT is critical. If one modality dominates, the model can learn an unconditional bias toward generating only that modality.

The fine-tuning uses a cosine learning rate schedule starting at 1e-5, batch size 128, sequence length 4096, and a dropout rate of 0.05. The loss is computed only on answer tokens, with prompt tokens masked.
<!-- 
## What Early-Fusion Models Can Do

The payoff of the early-fusion design is a single model that can handle a wide range of tasks without architectural changes:

- **Image captioning**: given an image, generate a text description.
- **Visual question answering**: given an image and a question, produce a text answer.
- **Text-to-image generation**: given a text prompt, generate image tokens that are decoded back into pixels.
- **Interleaved generation**: produce documents that freely alternate between text and generated images.

That last capability is unique to early-fusion models. A late-fusion model would need to call a separate image generator externally. An early-fusion model generates image tokens natively as part of its autoregressive output.

![Interleaved generation example](./vision.assets/chameleon_interleaved_generation.png)
> **Figure: Interleaved image and text generation.** Given the prompt "show me some cool, quirky-looking birds," the model generates a response that interleaves descriptive text with generated images of each bird. The `<img>` markers indicate where the model chose to insert generated images within the text flow. *(Source: Chameleon, Meta FAIR 2024)*

## Template Code: Fine-Tuning a Multimodal Model with Hugging Face

The code below illustrates how to fine-tune a multimodal model for visual question answering using Hugging Face `transformers`. We use a late-fusion model (LLaVA) as the example because it is openly available on the Hub. The training loop structure transfers directly to other multimodal architectures.

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

# -------------------------------------------------------
# 1. Load model and processor
# -------------------------------------------------------
model_name = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_name)
model = LlavaForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# -------------------------------------------------------
# 2. Prepare a visual-question-answering dataset
# -------------------------------------------------------
# For demonstration we use a small public VQA dataset.
# Replace this with your own medical VQA dataset.
dataset = load_dataset("HuggingFaceM4/VQAv2", split="train[:1000]")

def preprocess(example):
    """
    Each example has an image, a question, and an answer.
    We format them into a prompt that the model can learn from.
    """
    image = example["image"].convert("RGB")
    question = example["question"]
    answer = example["multiple_choice_answer"]

    # Build a chat-style prompt
    prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"

    # Tokenize text and image together
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding="max_length",
        max_length=256,
        truncation=True,
    )

    # The labels are the input_ids themselves (causal LM objective).
    # In practice you would mask the prompt portion.
    inputs["labels"] = inputs["input_ids"].clone()

    # Squeeze batch dimension added by return_tensors="pt"
    return {k: v.squeeze(0) for k, v in inputs.items()}

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
dataset.set_format("torch")

# -------------------------------------------------------
# 3. Training arguments
# -------------------------------------------------------
training_args = TrainingArguments(
    output_dir="llava-vqa-demo",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

# -------------------------------------------------------
# 4. Train
# -------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

### What to Adapt for Your Own Project

- **Dataset**: Replace the demo VQA dataset with your own image-question-answer triples. For medical applications, this might be radiology QA, pathology-based questions, or clinical image interpretation tasks.
- **Prompt format**: Different models expect different prompt templates. Always check the model card.
- **Loss masking**: For a production fine-tune, you should mask the prompt tokens so the loss is only computed on the answer tokens. This prevents the model from wasting capacity memorizing the question.
- **Image resolution**: Some multimodal models accept variable resolutions. Check whether the processor handles this or whether you need to resize manually.
- **LoRA or QLoRA**: For large models, parameter-efficient fine-tuning methods can reduce memory requirements significantly while preserving most of the performance. -->


## Medical Applications

Multimodal models are a natural fit for many clinical workflows:

- **Radiology report generation**: the model reads a chest X-ray and produces a structured text finding.
- **Pathology consultation**: given a whole-slide image and a clinical question, the model reasons over tissue regions and text jointly.
- **Clinical decision support**: integrating imaging, lab values, and clinical notes in a unified sequence.
- **Medical education**: generating illustrated explanations that interleave text and relevant images.

The early-fusion approach is especially interesting for tasks that require tight coupling between visual and textual reasoning, such as explaining *why* a particular region of an image supports a diagnosis.

## Summary

Multimodal foundation models extend the ViT insight one step further:

- ViT showed that images can be treated as sequences of patch tokens.
- Multimodal models show that images and text can be **interleaved** in a single token sequence.
- Early-fusion architectures process both modalities through one transformer, enabling native mixed-modal reasoning and generation.
- Training stability requires careful attention to normalization (QK-Norm), regularization (z-loss, dropout), and data balancing.
- Hugging Face provides practical tooling for fine-tuning multimodal models on custom datasets.

The field is moving quickly. As these models scale and improve, the boundary between vision models and language models continues to blur, opening up new possibilities for integrated AI systems in medicine and beyond.

## References and Further Reading

- Chameleon Team, [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818), *arXiv* 2024.
- Alayrac et al., [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2205.01068), *NeurIPS* 2022.
- Liu et al., [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485), *NeurIPS* 2023.
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020), *ICML* 2021.
- Hugging Face, [LLaVA documentation](https://huggingface.co/docs/transformers/en/model_doc/llava).
