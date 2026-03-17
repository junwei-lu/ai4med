# Stable Diffusion
Stable Diffusion feels a bit like hiring a very patient painter who starts from television static and, one denoising step at a time, turns it into "a corgi wearing a lab coat in watercolor style." The surprising part is that the model does not paint directly in pixel space. Instead, it works in a compressed **latent space**, which is the main trick that makes high-resolution diffusion practical.

This lecture explains the mathematical principle behind Stable Diffusion, its architecture, the main training losses, and code templates for loading data, training, and sampling.


Classic diffusion models are powerful, but pixel-space generation is expensive. If you try to denoise a large image directly, memory and compute costs rise quickly. Stable Diffusion solves this by pushing the diffusion process into a smaller latent representation:

$$
x \in \mathbb{R}^{H \times W \times 3}
\quad \longrightarrow \quad
z = \mathcal{E}(x) \in \mathbb{R}^{h \times w \times c},
\quad h \ll H,\; w \ll W
$$

Instead of learning to denoise full images, the model learns to denoise the latent code $z$. After sampling, a decoder maps the cleaned latent back to pixels:

$$
\hat{x} = \mathcal{D}(z_0)
$$

This is why Stable Diffusion can make detailed images without needing the compute budget of a small moon mission.

![Stable Diffusion architecture](generative.assets/stable_diffusion.png)


## The Core Idea in One Pipeline

Stable Diffusion has three main learned components:

1. **VAE**: compresses an image into a latent and decodes the latent back to pixels.
2. **Text encoder**: turns a prompt into token embeddings.
3. **U-Net denoiser**: predicts the noise inside a noisy latent while attending to the text.


## Mathematics of Stable-Diffusion

For self-consistency, we assemble all math procedures of stable diffusion here.

### 1. Compress Images into Latent Space

Let $\mathcal{E}$ be the VAE encoder and $\mathcal{D}$ be the decoder. For an input image $x$:

$$
z_0 \sim q_{\phi}(z \mid x), \qquad \hat{x} = \mathcal{D}_{\phi}(z_0)
$$

The latent $z_0$ is much smaller than $x$, but should still preserve semantics such as shape, layout, and texture.

### 2. Forward Diffusion in Latent Space

Stable Diffusion uses a DDPM-style forward process, but on latents instead of pixels:

$$
z_t = \sqrt{\bar{\alpha}_t}\, z_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon,
\qquad \epsilon \sim \mathcal{N}(0, I)
$$

where:

- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$
- $t \in \{1, \dots, T\}$

At large $t$, $z_t$ looks nearly Gaussian.

### 3. Learn the Reverse Process

The U-Net is trained to predict the noise that was added:

$$
\epsilon_{\theta}(z_t, t, c)
$$

where $c$ is the conditioning signal, usually the prompt embedding from the text encoder.

If the model can estimate $\epsilon$, then the sampler can iteratively walk from noisy latent $z_T$ back to a clean latent $z_0$.

### 4. Text Conditioning via Cross-Attention

The text prompt is tokenized and encoded into a sequence of hidden states:

$$
c = \text{TextEncoder}(\text{prompt})
$$

Inside the U-Net, cross-attention lets each spatial feature location attend to prompt tokens. Intuitively, the model can decide when to care about "red", "microscope", or "oil painting" instead of smearing all words into one vector soup.

## Architecture Details

### VAE: the compression engine

The VAE has two jobs:

- **Encoder**: map image $x$ to latent $z$
- **Decoder**: map latent $z$ back to image space

Why this matters:

- Diffusion becomes cheaper because the spatial grid is much smaller.
- The denoiser can spend more capacity on semantic structure instead of raw pixel bookkeeping.

In the original latent diffusion setup, a $512 \times 512$ RGB image is compressed to a latent with much smaller spatial size, often around `4 x 64 x 64`.

### Text Encoder: prompt to embeddings

Stable Diffusion v1 commonly uses a frozen CLIP text encoder. The prompt is converted into token embeddings, usually padded or truncated to a fixed token length.

Why freeze it?

- It already knows a lot about image-text alignment.
- Training becomes cheaper and more stable.
- The diffusion model only needs to learn how to use the embeddings, not reinvent language understanding from scratch.

### U-Net: the denoising workhorse

The U-Net takes:

- noisy latent $z_t$
- timestep embedding $t$
- text conditioning $c$

and outputs a prediction of the injected noise.

The U-Net usually contains:

- convolutional residual blocks
- downsampling and upsampling paths
- skip connections
- self-attention and cross-attention blocks
- timestep embeddings injected into residual blocks

The down path captures large context, the bottleneck mixes global information, and the up path restores spatial detail. Skip connections help preserve useful structure while denoising.

### Scheduler: the reverse-time navigator

The scheduler is not the neural network itself. It defines how to move from $z_t$ to $z_{t-1}$ after the U-Net predicts noise. Different schedulers such as DDPM, DDIM, Euler, and DPM-Solver trade off speed, stochasticity, and sample quality.

This is why changing the scheduler can alter image style and sharpness even when the learned model weights stay the same.

## Training Losses

Stable Diffusion is really a two-stage story: first learn the latent space, then learn diffusion inside it.

### Stage 1: VAE loss

The autoencoder is trained to reconstruct images while regularizing the latent distribution:

$$
\mathcal{L}_{\text{VAE}}
= \lambda_{\text{rec}} \, \|x - \hat{x}\|_1
+ \lambda_{\text{perc}} \, \mathcal{L}_{\text{perceptual}}(x, \hat{x})
+ \beta \, D_{\mathrm{KL}}\!\left(q_{\phi}(z \mid x)\,\|\,\mathcal{N}(0,I)\right)
$$

Interpretation:

- reconstruction loss keeps the decoded image faithful
- perceptual loss preserves visual quality better than raw pixels alone
- KL regularization keeps the latent space smooth enough to sample and denoise

### Stage 2: latent diffusion loss

Once the VAE is trained, encode the image into a latent $z_0$ and train the denoiser with:

$$
\mathcal{L}_{\text{diffusion}}
=
\mathbb{E}_{z_0, \epsilon, t, c}
\left[
\left\|
\epsilon - \epsilon_{\theta}(z_t, t, c)
\right\|_2^2
\right]
$$

where

$$
z_t = \sqrt{\bar{\alpha}_t}\, z_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon
$$

This is the standard noise-prediction objective. Some later models use $x_0$-prediction or $v$-prediction, but the intuition stays the same: teach the denoiser how to undo corruption at any noise level.

### Classifier-Free Guidance loss trick

To make prompts matter more, Stable Diffusion uses **classifier-free guidance**. During training, some prompts are dropped and replaced with an empty condition:

$$
c =
\begin{cases}
\varnothing, & \text{with probability } p_{\text{drop}} \\
\text{TextEncoder}(\text{prompt}), & \text{otherwise}
\end{cases}
$$

The model therefore learns both conditional and unconditional denoising in one network.

At sampling time, combine the two predictions:

$$
\hat{\epsilon}
= \epsilon_{\theta}(z_t, t, \varnothing)
+ s \left[
\epsilon_{\theta}(z_t, t, c)
- \epsilon_{\theta}(z_t, t, \varnothing)
\right]
$$

where $s$ is the guidance scale.

Rule of thumb:

- small `guidance_scale`: more diversity, weaker prompt adherence
- medium `guidance_scale` such as `5-8`: good balance
- very large `guidance_scale`: prompt-following becomes stronger, but artifacts can appear

## Training Recipe

If you want to fine-tune Stable Diffusion on a captioned dataset, the standard workflow is:

1. Load image-caption pairs.
2. Freeze the VAE and text encoder.
3. Encode images into latents with the VAE.
4. Tokenize captions and obtain text embeddings.
5. Add random noise to latents at random timesteps.
6. Train the U-Net to predict that noise.
7. Sample from the trained checkpoint with a scheduler.

This is already enough for a strong first fine-tuning baseline.


### Data loading

The Hugging Face `datasets` library can load a folder of images and captions with one call. No custom Dataset class needed.

Organize your folder like this:

```
my_data/
  train/
    image_001.png
    image_002.png
    ...
  metadata.csv          # columns: file_name, text
```

Then load and preprocess:

```python
from datasets import load_dataset
from torchvision import transforms

# One line to load the whole dataset.
dataset = load_dataset("imagefolder", data_dir="my_data", split="train")

# Define the image transform (resize, crop, normalize to [-1, 1]).
transform = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def preprocess(examples):
    """Apply transform to images and tokenize captions."""
    examples["pixel_values"] = [transform(img.convert("RGB")) for img in examples["image"]]
    examples["input_ids"] = tokenizer(
        examples["text"],
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids
    return examples


dataset.set_transform(preprocess)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
```

### Model loading

Load everything at once through `StableDiffusionPipeline`, then pull out the parts you need. This avoids importing and loading four separate classes.

```python
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"

# One call downloads the VAE, text encoder, U-Net, tokenizer, and scheduler.
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

# Pull out individual components for training.
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
vae = pipe.vae
unet = pipe.unet

# Use DDPM scheduler for training (the pipeline default may differ).
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Freeze everything except the U-Net (the part we want to fine-tune).
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.train()
```

### Training by `diffusers` 

The [`diffusers` library](https://huggingface.co/docs/diffusers/index) ships an official fine-tuning script that handles the training loop, mixed precision, multi-GPU, logging, and checkpointing for you. Just point it at your data:

```bash
pip install accelerate diffusers transformers datasets

accelerate launch diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="my_data"  \
  --resolution=512 \
  --train_batch_size=4 \
  --learning_rate=1e-5 \
  --max_train_steps=5000 \
  --output_dir="sd-finetuned"
```

That single command does everything: data loading, VAE encoding, noise scheduling, U-Net training, and checkpoint saving.


### Training: under the hood

If you want to understand what the script above is doing, here is the core loop stripped to its essentials. Each numbered comment marks one stage of the latent-diffusion training step.

```python
import torch
import torch.nn.functional as F

optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5, weight_decay=1e-2)

for batch in dataloader:
    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)

    with torch.no_grad():
        # ① Encode image → latent.
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # ② Get text embeddings from captions.
        encoder_hidden_states = text_encoder(input_ids)[0]

    # ③ Add random noise at a random timestep.
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],), device=device,
    ).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # ④ U-Net predicts the noise that was added.
    noise_pred = unet(noisy_latents, timesteps,
                      encoder_hidden_states=encoder_hidden_states).sample

    # ⑤ MSE between predicted and actual noise → backprop.
    loss = F.mse_loss(noise_pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"loss={loss.item():.4f}")
```

**Practical notes**

- Fine-tuning the whole model is expensive; many workflows train only the U-Net or LoRA adapters.
- Mixed precision and gradient accumulation are usually necessary on limited GPUs.
- Caption quality matters a lot. If captions are vague, the model learns vague associations.
- For domain adaptation, keep prompts close to the visual content you actually want the model to learn.
- To train with classifier-free guidance, randomly replace some captions with empty strings so the model also learns unconditional denoising. The math is explained in the Classifier-Free Guidance section above.

###  Sampling

The easiest way to sample is through a high-level pipeline:

```python
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "a glowing jellyfish floating in a glass laboratory, cinematic lighting"
negative_prompt = "blurry, low quality, distorted anatomy"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    height=512,
    width=512,
).images[0]

image.save("stable_diffusion_sample.png")
```
<!-- 
### Watching the denoising process

If you want to watch how the image forms step by step, use the pipeline's built-in `callback_on_step_end` instead of writing your own sampling loop. The callback receives the in-progress latents at every step:

```python
from diffusers.image_processor import VaeImageProcessor
from IPython.display import display

processor = VaeImageProcessor()
snapshots = []   # collect intermediate images here


def save_snapshot(pipe, step, timestep, callback_kwargs):
    """Decode the current latent and save a snapshot every 5 steps."""
    if step % 5 == 0:
        latents = callback_kwargs["latents"]
        with torch.no_grad():
            image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        snapshots.append(processor.postprocess(image, output_type="pil")[0])
    return callback_kwargs


image = pipe(
    "a watercolor painting of a fox reading a medical textbook",
    num_inference_steps=30,
    guidance_scale=7.5,
    callback_on_step_end=save_snapshot,
).images[0]

# snapshots now contains images from steps 0, 5, 10, 15, 20, 25.
for i, snap in enumerate(snapshots):
    display(snap)
``` -->
