# Fine-Tuning ViT in Practice

## The Typical Workflow

Once a ViT encoder is pretrained, adapting it to a new task is usually straightforward:

1. Load a pretrained checkpoint.
2. Replace or attach a task-specific head.
3. Preprocess images to the expected resolution and normalization.
4. Fine-tune on your labeled dataset.

In real projects, most performance gains come from choosing the right data pipeline, augmentation, and regularization rather than rewriting the architecture.

## A Concrete Example Image

![Example image](./vision.assets/example_cat.jpeg)
> Small example image from the Hugging Face documentation assets. In practice, you would replace this with histology, X-ray, fundus, or dermoscopy images.

## Linear Probe vs Full Fine-Tuning

There are two common ways to adapt ViT.

### Linear Probe

Freeze the encoder and train only a small classifier on top.

Use this when:

- you want a quick baseline,
- your dataset is very small,
- you want to test representation quality.

### Full Fine-Tuning

Update most or all of the backbone weights.

Use this when:

- the new domain is far from natural images,
- you have enough data,
- you need the best possible task performance.

For medical imaging, full fine-tuning often beats linear probing because the texture and acquisition patterns can differ substantially from ImageNet.

## Hugging Face Example

```python
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)

model_name = "google/vit-base-patch16-224"

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=2,
    ignore_mismatched_sizes=True,
)

dataset = load_dataset("beans")

def transform(example):
    image = example["image"].convert("RGB")
    inputs = processor(image, return_tensors="pt")
    example["pixel_values"] = inputs["pixel_values"][0]
    return example

dataset = dataset.with_transform(transform)

args = TrainingArguments(
    output_dir="vit-beans-demo",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()
```

## Practical Tips That Matter

### 1. Match the Expected Input Pipeline

Use the checkpoint's image processor whenever possible. It handles:

- resizing,
- center crop or rescale,
- normalization,
- RGB conversion.

Small preprocessing mismatches can hurt performance more than people expect.

### 2. Consider Higher Resolution

A common trick is:

- pretrain at `224 x 224`,
- fine-tune at `384 x 384`.

This gives the model more visual detail, though it also increases memory and attention cost.

### 3. Watch Sequence Length

If patch size stays fixed, higher image resolution means more patches:

- `224 x 224` with patch size 16 gives `196` patches,
- `384 x 384` with patch size 16 gives `576` patches.

Because attention scales roughly quadratically with sequence length, compute can jump quickly.

### 4. Do Not Ignore Class Imbalance

In medicine, rare disease categories may be severely underrepresented. Use:

- weighted loss,
- balanced sampling,
- AUROC or AUPRC,
- patient-level splits instead of image-level leakage.

## A Good Beginner Strategy

If you are new to ViT fine-tuning, use this recipe:

1. Start with a pretrained `vit-base-patch16-224`.
2. Run a linear probe.
3. Then unfreeze the full encoder.
4. Add moderate augmentation.
5. Track both accuracy and calibration.

This is fast, interpretable, and usually more informative than jumping straight into large experiments.

## Why Fine-Tuning Can Fail

Common failure modes include:

- too little data for the chosen model size,
- overly aggressive augmentation,
- poor label quality,
- resolution mismatch,
- evaluating only image-level metrics when patient-level metrics matter.

## Summary

Fine-tuning ViT is often easier than training from scratch:

- the pretrained encoder already knows many useful visual patterns,
- you only need to adapt it to your target labels,
- good preprocessing and evaluation design matter as much as the optimizer.

## References and Further Reading

- Hugging Face, [Vision Transformer (ViT) documentation](https://huggingface.co/docs/transformers/en/model_doc/vit).
- Hugging Face model card, [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224).
- Dosovitskiy et al., [An Image is Worth 16x16 Words](https://openreview.net/forum?id=YicbFdNTTy), *ICLR* 2021.
