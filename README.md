# PrivateSeal

Code for NeurIPS 2026 Paper "PrivateSeal: Low-Sensitivity Latent Directions for Diffusion-Resilient User-Specific Watermarking" by Yiheng Chen, Zhong Ji, Zhihao Li, Boyu Wang.



## Abstract

Diffusion-driven image editing and regeneration are becoming increasingly widespread. This creates an urgent need for watermarking methods that can embed imperceptible yet verifiable signals into existing images for ownership attribution and provenance tracking. Existing methods fail to simultaneously satisfy three practical requirements: robustness against diffusion-based transformations, non-iterative verification, and scalable support for per-user key assignment at low overhead. To address these challenges, we propose PrivateSeal, a watermarking framework for pre-existing images under diffusion-based transformations. The perturbation is encouraged to lie along low-sensitivity latent directions, so that the embedded signal is less likely to be suppressed during diffusion-based editing or regeneration. During verification, the embedded message is reliably recovered via a simple latent-space projection using the corresponding key. This design allows a platform to assign independent projection keys to different users, accounts, or images without retraining or modifying the verifier, while maintaining non-iterative verification. Extensive experiments on the W-Bench benchmark show that PrivateSeal achieves competitive robustness against diffusion-based regeneration and editing, with additional cross-model and cross-dataset evaluations further validating its strong transferability.

## Requirements

- Set up a conda environment and install some prerequisite packages.


```bash
conda create -n privateseal python=3.12 -y
conda activate privateseal

pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

## Configuration required before running

### 1. Hugging Face cache and endpoint

```python
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
```

Remove `HF_ENDPOINT` if the mirror is unnecessary, and set `HF_HOME` to a writable cache directory.

### 2. Stable Diffusion model

```python
model_id = "runwayml/stable-diffusion-v1-5"
```

You may replace this with a compatible local Diffusers directory or another accessible SD1.5 repository.

### 3. Key directory

```python
KeyPATH = "/path/to/your/keys"
```

The script expects a file named `global_watermark_key_30.pt` in that directory, you may use `generate_private_key.py` to generate it.

### 4. Dataset

```python
dataset_path = "/path/to/images/*.png"
```

### 5. Experiment size

```python
N_loop = 1000
batch_size = 4
```

Set `N_loop` to no more than the number of input images. The current loop assumes that every scheduled batch is non-empty.

## Run

After configuring the paths:

```bash
python PrivateSeal_core.py
```

## W-Bench evaluation

Our diffusion-attack evaluation follows the pipelines from [VINE/W-Bench](https://github.com/Shilin-LU/VINE). W-Bench is a separate project with its own license and dependencies; it is not bundled into `PrivateSeal_core.py`.

## Responsible use

PrivateSeal is intended for research on provenance, ownership verification, and responsible media tracking. Users are responsible for complying with applicable laws, platform policies, dataset licenses, and model licenses. Do not use watermarking to misrepresent authorship or ownership.

## Citation

If you use this code, please cite the PrivateSeal paper. Replace the author and venue fields below with the final publication metadata before release:

```bibtex
@inproceedings{
chen2026privateseal,
title={PrivateSeal: Low-Sensitivity Latent Directions for Diffusion-Resilient User-Specific Watermarking},
author={Yiheng Chen and Zhong Ji and Zhihao Li and Boyu Wang},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026}
}
```