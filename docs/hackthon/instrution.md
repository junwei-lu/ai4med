# Prompt-to-Protein: CLIP-Style Protein Design in One Afternoon

## Goal

Given a text prompt such as:
- "a metal transporter"
- "a kinase-like signaling protein"
- "a small enzyme involved in redox chemistry"

You build a small system that:
1. Embeds protein sequences.
2. Aligns proteins with text.
3. Generates a candidate protein latent for a target prompt.
4. Retrieves or refines a sequence.
5. Predicts and visualizes its structure.

## Dataset 

- Download the dataset from [Protein2Text-QA](https://huggingface.co/datasets/tumorailab/Protein2Text-QA)

## Model

- Protein encoder: [ProtT5](https://huggingface.co/Rostlab/prot_t5_xl_uniref50)
- Text encoder: [MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## Stages

### Stage 1: Build the Paired Dataset 

From Protein2Text-QA, create pairs like:
- Protein: amino-acid sequence
- Text: short answer/function description

To simplify:
- Keep only the first answer sentence
- Filter to sequences shorter than a max length (for example, 256 or 512 aa)
- Optionally cluster or deduplicate very similar text descriptions





### Stage 2: Protein Encoder Training

Start from ProtT5 and train a lightweight protein representation model.

Output:
- 256-d or 384-d protein embedding

Get a working AA sequence -> latent -> AA sequence path.


### Stage 3: CLIP-Style Protein-Text Alignment

Use:
- protein branch: ProtT5 encoder + projection head
- text branch: MiniLM + projection head
- loss: standard InfoNCE / contrastive loss

Evaluation:
- Protein-to-text retrieval
- Text-to-protein retrieval
- Recall@1 or simple top-k hit rate

Good prompts:
- "metal transporter"
- "probable kinase"
- "serine endoprotease"

The `uniprot_sentences` and Protein2Text-QA previews show these kinds of short function descriptions associated with amino-acid sequences.

### Stage 4: Diffusion-Style Latent Generator


Train a small conditional latent DDPM over the protein embedding space.

### Stage 5: Decode Latent to Sequence

Use retrieval decoding:
- Sample a latent from diffusion
- Find nearest protein embeddings in the training set
- Choose the nearest protein as the candidate
- Optionally mutate a few residues around low-conservation positions

This gives you a valid sequence immediately.

#### Slightly more ambitious option

Add a tiny sequence decoder or use a pretrained protein generator afterward, but keep that optional.

For the core hackathon, retrieval-as-decoder is the right compromise.

### Stage 6: Fold and Visualize

Take the final sequence and run it through a protein folding model.

- Predicted 3D structure