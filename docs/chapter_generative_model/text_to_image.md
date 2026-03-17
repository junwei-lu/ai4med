# Conditional Generation


## Contextual DDPM



We can generalize the DDPM model to generate the conditional distribution $p(x |c)$ where $c$ could be even be a text prompt.

We can just simple add the context $c$ to the input of the noise predictor $\varepsilon_\theta(x_t, t, c)$ and train the loss function as:

$$
\mathbb{E}_{(x_0, c) \sim \text{Data}, t \sim \text{Uniform}\{1 ,\ldots, T\}, \varepsilon \sim \mathcal{N}(0, I)} \| \varepsilon - \varepsilon_\theta(\bar{\alpha}_t x_0 + \bar{\beta}_t \varepsilon, t, c)\|^2 
$$

More technologies like the [CLIP](https://arxiv.org/abs/2103.00020) can be used to improve the quality of the generated images. Please refer to the [OpenAI DALL-E paper](https://arxiv.org/pdf/2204.06125) for more details.

![dalle](generative.assets/dalle.png)



### Classifier-Free Guidance








For conditional probabiltiy generation, there is a trade-off between the fidelity and mode-coverage (diversity) of the generated images. In order to tune the trade-off, we can use the **classifier-free guidance** to sample using a linear combination of conditional and unconditional samples:

$$
\tilde{\varepsilon}_\theta(x_t, t, c) = \epsilon_{\theta}(z_t, t, \varnothing)
+ s \left[
\epsilon_{\theta}(z_t, t, c)
- \epsilon_{\theta}(z_t, t, \varnothing)
\right]
$$

where $\varepsilon_\theta(x_t, t)$ is an unconditional noise predictor. Usually, we will use the same network for both conditional and unconditional cases. For unconditional case, we will use a null token $\varnothing$ as the context $c$ and fit $\varepsilon_\theta(x_t, t) = \varepsilon_\theta(x_t, t, \varnothing)$.

![CFG](generative.assets/classifier_free.png)


Classifier-Free Guidance works by training the model to handle both **conditional** (with text) and **unconditional** (without text) generation.

During **training**, we randomly "drop" the text caption about 10% of the time (replacing it with an empty string `""`). This forces the model to learn how to generate images even without instructions.

During **inference** (generation), we perform two forward passes for every step:
1.  **Unconditional pass**: Predict noise given the empty string ($\varnothing$).
2.  **Conditional pass**: Predict noise given the actual text prompt.

We then combine these predictions to push the image away from the "generic" result and towards the "text-specific" result.

The training process can be summarized as follows.

**Input**: $p_{uncond}$: probability of unconditional training

**Repeat**:

  1. Sample data with conditioning from the dataset: $(x, c) \sim p(x, c)$
  2. Randomly discard conditioning to train unconditionally: $c \leftarrow \emptyset$ with probability $p_{uncond}$
  3. Sample log SNR value: $\lambda \sim p(\lambda)$
  4. Sample Gaussian noise: $\epsilon \sim \mathcal{N}(0, I)$
  5. Corrupt data to the sampled log SNR value: $z_\lambda = \alpha_\lambda x + \sigma_\lambda \epsilon$
  6. Take gradient step on $\nabla_\theta \|\epsilon_\theta(z_\lambda, c) - \epsilon\|^2$

**Until** converged


We will then use $\tilde{\varepsilon}_\theta(x_t, t, c)$ to sample from the model.

When $w$ increases from 0 to $\infty$, the generated images will become less fidelity and more diversity.

This guide introduces the concepts behind modern text-to-image generation models (like Stable Diffusion) and walks through building a small-scale version from scratch. We will cover the core architecture, the mathematics of guidance, and the code structure needed to train your own model.

## Example: Text-to-Image Generation

At a high level, a text-to-image model learns to transform random noise into coherent images that match a text description. It does this through a process called **diffusion**, where the model learns to iteratively remove noise from an image.

To make this process "conditional" (i.e., controlled by text), we need two key components working together:

1.  **Text Understanding**: A way to convert text into numbers (vectors) that the computer can understand.
2.  **Image Generation**: A neural network that can "attend" to these text vectors while generating the image.

### The Architecture

We will build a model that generates 64x64 Pokemon images from captions. The architecture consists of three main parts:

1.  **Text Encoder (Frozen CLIP)**: Converts text prompts like "a blue dragon" into rich feature vectors. We use a pre-trained model and keep it "frozen" (we don't train it), allowing us to leverage its existing knowledge of language.
2.  **U-Net (The Denoiser)**: The core neural network that predicts noise. It takes a noisy image and the text features as input.
3.  **Cross-Attention**: The mechanism inside the U-Net that allows the image generation process to "look at" specific parts of the text description (e.g., attending to the word "blue" when generating the body).




<!-- 
### The Formula

The final predicted noise $\hat{\epsilon}$ is calculated as:

$$
\hat{\epsilon} = \epsilon_\theta(x_t, \varnothing) + s \cdot (\epsilon_\theta(x_t, \text{text}) - \epsilon_\theta(x_t, \varnothing))
$$

Where:
*   $\epsilon_\theta(x_t, \varnothing)$ is the noise predicted without text (unconditional).
*   $\epsilon_\theta(x_t, \text{text})$ is the noise predicted with text (conditional).
*   $s$ is the **Guidance Scale**.

### Understanding the Guidance Scale ($s$)
The scale $s$ controls how strongly the model listens to the text:
*   **$s = 1$**: No guidance. The model generates what it thinks is likely, often ignoring specific details.
*   **$s \approx 3-7$**: The "Sweet Spot". The model follows the text well while maintaining good image quality.
*   **$s > 10$**: Strong guidance. The image will strictly follow the prompt but might look "fried" or have artifacts because we are pushing the values too far. -->

## Code for Classifier-Free Guidance

Here is a breakdown of the code structure for building this system using PyTorch and the Hugging Face `diffusers` library.

### 1. The Dataset
We need pairs of (image, text). The **Pokemon BLIP Captions** dataset is perfect for this.

```python
# Load dataset
dataset = load_dataset("lambdalabs/naruto-blip-captions")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to 64x64
    transforms.ToTensor(),        # Convert to tensor [0, 1]
    transforms.Normalize([0.5], [0.5]) # Normalize to [-1, 1]
])
```

### 2. The Text Encoder (CLIP)
We use a pre-trained CLIP model to encode text.

```python
# Load CLIP (frozen)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Freeze weights (we don't train this part)
text_encoder.requires_grad_(False)
```

### 3. The U-Net Model
We use `UNet2DConditionModel`, which supports cross-attention.

```python
unet = UNet2DConditionModel(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    cross_attention_dim=512, # Must match CLIP output size
    down_block_types=(
        "DownBlock2D",          # Standard convolution
        "CrossAttnDownBlock2D", # Convolution + Text Attention
        "CrossAttnDownBlock2D",
    ),
    up_block_types=(
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D",
    ),
)
```

### 4. The Training Loop
The training loop involves:
1.  Encoding the text (with random dropout for CFG).
2.  Adding noise to the image (Forward Diffusion).
3.  Predicting the noise using the U-Net.
4.  Calculating Loss (MSE between actual noise and predicted noise).

```python
# CFG Dropout Probability (e.g., 10%)
cfg_prob = 0.1

for images, captions in dataloader:
    # 1. Text Encoding with Dropout
    # Randomly replace some captions with ""
    captions = [c if random.random() > cfg_prob else "" for c in captions]
    text_embeddings = text_encoder(captions)

    # 2. Add Noise
    noise = torch.randn_like(images)
    timesteps = torch.randint(0, 1000, (batch_size,))
    noisy_images = scheduler.add_noise(images, noise, timesteps)

    # 3. Predict Noise
    noise_pred = unet(noisy_images, timesteps, encoder_hidden_states=text_embeddings).sample

    # 4. Compute Loss & Backpropagate
    loss = F.mse_loss(noise_pred, noise)
    loss.backward()
    optimizer.step()
```

### 5. Inference (Sampling)
To generate an image, we start with random noise and iteratively remove it, using the CFG formula at each step.

```python
# Start with random noise
latents = torch.randn(1, 3, 64, 64)

# Loop backwards from T to 0
for t in scheduler.timesteps:
    # Predict noise for both empty "" and prompt "text"
    noise_uncond = unet(latents, t, encoder_hidden_states=empty_embeds).sample
    noise_cond = unet(latents, t, encoder_hidden_states=text_embeds).sample

    # Apply CFG Formula
    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    # Update latents (remove noise)
    latents = scheduler.step(noise_pred, t, latents).prev_sample
```

## Summary

By combining a powerful pre-trained text encoder (CLIP) with a trainable image generator (U-Net) and using the mathematical trick of Classifier-Free Guidance, we can build a system that generates specific images from text descriptions.

This "mini" model uses the exact same principles as massive state-of-the-art models like Stable Diffusion, just on a smaller scale (pixel space instead of latent space, smaller resolution).
