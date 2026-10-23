# Diffusion Model Exploration

A hands-on walkthrough of diffusion models in three parts: visualizing the forward noising process on custom images, comparing DDIM sampling quality/speed at different step counts, and using Stable Diffusion for text-guided image-to-image editing.

## Part 1 — Forward diffusion

A from-scratch implementation of the forward diffusion process (`ForwardDiffusion`): a linear beta schedule (`beta_start=1e-4`, `beta_end=0.02`) over 1000 timesteps, with closed-form sampling of `x_t` from `x_0` via the cumulative product of `alphas`. Applied to a handful of sample images (`image1.jpg`–`image4.jpg`) at timesteps `t = 0, 10, 50, 100, 500`, measuring the mean-squared error between the original and noised image as a function of `t` to quantify how quickly structure is destroyed.

## Part 2 — DDIM sampling speed/quality tradeoff

Uses a pretrained [`google/ddpm-cifar10-32`](https://huggingface.co/google/ddpm-cifar10-32) UNet with a Hugging Face `DDIMScheduler` to generate images from a *fixed* noise seed at 100, 50, and 10 inference steps, comparing output quality (visual + MSE against the 100-step result) and confirming DDIM's deterministic sampling property (two runs with identical settings produce (near-)identical outputs).

## Part 3 — Text-guided image editing with Stable Diffusion

Uses [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) (`StableDiffusionPipeline` for text-to-image, `StableDiffusionImg2ImgPipeline` for image-to-image) to generate a base image from a prompt, then edit it toward a second prompt (e.g. "a photo of a cat on a couch" → "a photo of a dog on a couch") at `strength=0.75`.

## Dataset

Four sample images (`image1.jpg`–`image4.jpg`) used as inputs to the forward diffusion experiment in Part 1. Parts 2 and 3 generate their own images at runtime via the pretrained models above — no external dataset required.

## Running

```bash
pip install torch torchvision diffusers transformers matplotlib pillow numpy tqdm
jupyter notebook diffusion_model_exploration.ipynb
```

A CUDA GPU is strongly recommended for Parts 2 and 3 (UNet/Stable Diffusion inference). Part 1 runs fine on CPU. Note: the notebook was originally run on Google Colab (`google.colab.files` import, `--break-system-packages` pip flag) — drop the Colab-specific import if running locally and just place the sample images alongside the notebook.

## Files

- `diffusion_model_exploration.ipynb` — all three parts
- `image1.jpg`–`image4.jpg` — sample input images for the forward diffusion experiment
