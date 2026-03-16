# Generative Model

This should be a 3 hour course on generative models. Your job is to create a Jupyter notebook that covers the following topics:

- Incorporate the ddpm and flow match code from the ddpm.ipynb and flow_match.ipynb files

- Add more materials on the code for the conditional generation of both diffusion and flow models. 

- Follow: pokemon-blip-captions + 64x64 pixel-space conditional DDPM + frozen text encoder + small cross-attention U-Net + classifier-free guidance. Add as detailed explanations for each step as possible.

Primary choice: Pokémon BLIP captions

Use reach-vb/pokemon-blip-captions or the equivalent Lambda dataset mirror. It is the best teaching dataset because:
	•	captions are simple and visually grounded
	•	the images are stylized, so a small model can still produce outputs that look “reasonable”
	•	prompt-response relationships are obvious: color, creature type, eyes, pose, mood
	•	students can see conditioning work after relatively little training  ￼

What not to use as the main training set

Do not use COCO as the main day-3 training set. It is too broad for a small scratch DDPM to show good prompt-following in one course. COCO’s scale and diversity are much better suited to pretrained or much larger systems.  ￼

Best model design for teaching

I would teach a small conditional DDPM in pixel space:
	•	images: 64x64 RGB
	•	noise process: DDPMScheduler
	•	denoiser: UNet2DConditionModel
	•	text side: frozen pretrained text encoder
	•	conditioning method: cross-attention
	•	sampling: classifier-free guidance at inference

This is the cleanest educational design because Hugging Face Diffusers already exposes exactly these building blocks: UNet2DConditionModel is the standard conditional U-Net used in diffusion systems, and DDPMScheduler handles the forward process of adding noise to clean samples at a selected timestep.  ￼

Why pixel-space DDPM instead of latent diffusion

For teaching, pixel-space DDPM is better than Stable-Diffusion-style latent diffusion.

Stable Diffusion uses a VAE, a frozen CLIP text encoder, a conditional U-Net, and a scheduler; the model card explains that images are first encoded into latents and the U-Net learns to predict the added noise in latent space using text features through cross-attention. That is powerful, but it adds too many moving parts for a first classroom build.  ￼

So for day 3, I would simplify:
	•	skip the VAE
	•	work directly on 64x64 pixels
	•	keep only the essential DDPM loop
	•	keep text conditioning through cross-attention

Students then see the core logic without getting lost in latent-space machinery.

Best text encoder choice

Use a frozen pretrained text encoder, not one trained from scratch.

Stable Diffusion uses a frozen CLIP text encoder with a tokenizer, and that is the right conceptual precedent for your class.  ￼

For teaching, I would choose one of these two:

Option A: frozen CLIP text encoder

Best if you want the strongest prompt semantics.

Option B: a smaller frozen text encoder

Best if Colab memory is tight.

For the Pokémon dataset, either can work because the domain is narrow. But educationally, I would still explain that using a frozen pretrained text encoder helps because the U-Net can learn image generation while borrowing an already useful text representation. That mirrors how modern text-conditioned diffusion systems are commonly built.  ￼

Best U-Net size

Keep the U-Net small enough that students can understand it.

Suggested classroom config:
	•	sample_size=64
	•	in_channels=3
	•	out_channels=3
	•	3 down blocks / 3 up blocks
	•	channel widths around 64, 128, 256
	•	cross-attention only in the deeper blocks
	•	1 transformer layer per cross-attention block

The reason this is the right place to simplify is that UNet2DConditionModel is the main denoiser and the core component students need to understand. Diffusers documents it as one of the most important parts of the system because it produces same-size outputs while conditioning on external information.  ￼

Use classifier-free guidance, but teach it lightly

Yes, include classifier-free guidance. It makes prompt following visibly stronger.

Diffusers’ Stable Diffusion docs state that guidance_scale > 1 activates classifier-free guidance and that larger guidance scales make images more closely linked to the prompt, often at some cost to image quality. Stable Diffusion v1.5 was also trained with 10% text-conditioning dropout specifically to improve classifier-free guidance sampling.  ￼

For class:
	•	during training, randomly drop text conditioning about 10% of the time
	•	during inference, compare guidance_scale=1, 3, and 6







