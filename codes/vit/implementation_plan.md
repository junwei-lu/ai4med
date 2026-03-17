# Revised ViT + Multimodal Workshop Notebook

Create a new notebook `vit_multimodal_cxr_workshop.ipynb` based on the existing `vit_mae_mri_workshop.ipynb`, with three major changes: (1) switch to chest X-ray data, (2) reorganize sections and add model architecture visualization, (3) add a new multimodal section with Janus-Pro-1B.

## Proposed Changes

### [NEW] [vit_multimodal_cxr_workshop.ipynb](file:///Users/junweilu/Dropbox/Teach/AI_bootcamp/ai4med/codes/vit/vit_multimodal_cxr_workshop.ipynb)

New notebook with the following structure:

#### Part I: ViT-MAE on Chest X-Rays (Sections 0–7)

| Section | Title | Notes |
|---------|-------|-------|
| 0 | **Setup** | Install deps including `janus` package; same structure as original |
| 1 | **Load Chest X-Ray Images** | Use `itsanmolgupta/mimic-cxr-dataset` instead of ROCOv2. Load a subset (~500 train, ~100 val). Display sample gallery. |
| 2 | **Load Pretrained ViT-MAE** | Moved from original Section 3. Load `facebook/vit-mae-base`. **Add**: (a) print full model architecture, (b) visualize how an image is split into patches (grid overlay + flattened patch sequence), (c) show how patches become input embeddings. |
| 3 | **How MAE Masking Works** | Moved from original Section 2 (comes after model loading now). Same masking visualization but on chest X-ray. |
| 4 | **Reconstructions Before Training** | Same as original Section 4, adapted for CXR. |
| 5 | **Fine-Tune MAE on CXR** | Same as original Section 5, adapted data references. |
| 6 | **Reconstructions After Training** | Same as original Section 6. |
| 7 | **Before vs After Comparison** | Same as original Section 7. |

#### Part II: Multimodal Foundation Model (Sections 8–14)

| Section | Title | Notes |
|---------|-------|-------|
| 8 | **Intro to Multimodal Models** | Brief markdown explaining unified multimodal understanding + generation. Introduce Janus-Pro architecture (decoupled visual encoding). |
| 9 | **Load Janus-Pro-1B & Visualize Mixed-Modal Input** | Load `deepseek-ai/Janus-Pro-1B` using `janus` package. Visualize how the model tokenizes and embeds an image + text prompt together into a single sequence. Show the mixed-modal token sequence. |
| 10 | **Image Captioning** | Given a CXR image, generate a text description using Janus-Pro understanding mode. |
| 11 | **Visual Question Answering** | Given a CXR image + a clinical question, produce a text answer. |
| 12 | **Text-to-Image Generation** | Given a text prompt describing a chest X-ray, generate image tokens decoded back to pixels. |
| 13 | **Interleaved Generation** | Produce a document that alternates between text and generated images — e.g., a mini radiology report with inline generated X-ray views. |
| 14 | **Quick Fine-Tuning on CXR** | Use HuggingFace `Trainer` / PEFT (LoRA) to fine-tune Janus-Pro-1B on the CXR dataset for improved captioning. Show before/after comparison. |

#### Section 15: Summary

Updated summary table covering both ViT-MAE and multimodal sections.

---

## Key Design Decisions

> [!IMPORTANT]
> - **Janus-Pro-1B requires the `janus` package** from `git+https://github.com/deepseek-ai/Janus.git`. This will be installed in the setup cell.
> - **The model uses `trust_remote_code=True`** — necessary for the custom architecture.
> - For the **Janus-Pro-1B conversation format**, the role names use `<|User|>` and `<|Assistant|>` tags.
> - **Fine-tuning** in Section 14 will use LoRA via PEFT for efficiency (only a few trainable parameters on the 1B model).
> - **Interleaved generation** (Section 13) is a demonstration of the concept — we'll generate text, then generate an image from the text, showing how the model can alternate modalities.
> - The notebook is designed for **Google Colab with a T4 GPU** (16 GB VRAM), which should comfortably handle the 1B model in bfloat16.

## Verification Plan

### Manual Verification
- Open the notebook in Jupyter/Colab and verify all cells are syntactically correct
- Verify the notebook structure matches the planned section order
- Verify the `itsanmolgupta/mimic-cxr-dataset` is used for ViT-MAE training (not ROCOv2)
- Confirm Section 2 contains model architecture printing and patch visualization
- Confirm Sections 8–14 contain the multimodal content with Janus-Pro-1B
- The user should run the notebook on Colab with a T4 GPU to validate all cells execute correctly
