# SafeRedir: Prompt Embedding Redirection for Robust Unlearning in Image Generation Models

This repository provides a guided image generation pipeline built on **Stable Diffusion**, enhanced with a plug-and-play redirector module called **SafeRedir**. The system enables controlled image generation by redirecting unsafe or undesired concepts at the embedding level. It supports two key modes:

- `retain`: preserve the original concept
- `forgot`: erase or unlearn the specified concept

This is especially useful for research in safety-aware generation and machine unlearning evaluation.

---

## 🔧 Features

- ⚙️ Dual-mode support: `retain` and `forgot`
- 🔄 Prompt embedding redirection with SafeRedir
- 🧩 Compatible with DDIM sampling
- 🧪 Task-specific prompt control and reproducibility
- 📂 Automatic directory and filename structure

---

## 📁 Project Structure

```
.
├── safe_generate.py          # Main script for guided image generation
├── model.py                  # SafeRedir model definition and hook registration
├── util.py                   # Utility functions (e.g., seed setup)
├── demo.ipynb                # Quick-start Jupyter notebook demo
├── environment.yaml          # Conda environment specification
├── LICENSE                   # Project license (MIT)
├── README.md                 # Project documentation

├── ckpt/
│   └── Nudity/
│       └── best_model.pt     # Pretrained SafeRedir checkpoint

├── data/
│   ├── IGMU_retain.json      # Prompts for retained concepts
│   └── IGMU_forgot.json      # Prompts for unlearned concepts

├── gen_imgs/                 # Output directory for generated images (auto-created)


```

---

## 🚀 Getting Started

### 1. Install dependencies

We recommend using Conda:

```bash
conda env create -f environment.yaml
conda activate saferedir
```

### 2. Prepare checkpoints and data

- Download Stable Diffusion v1.4 weights via `diffusers`.
- Place your SafeRedir model checkpoint at:

  `ckpt/Nudity/best_model.pt`
- Place prompt files at:

  - `data/IGMU_retain.json`
  - `data/IGMU_forgot.json`

---

## 🔨 Usage

You can run the generation via command line:

```bash
python safe_generate.py \
    --gen-type forgot \
    --task Nudity \
    --ddim_steps 50 \
    --gen_nums 5 \
    --guidance_scale 7.5 \
    --replace_step 2 \
    --alpha_scale 1.25 \
    --save_dir gen_imgs
```

Or try a quick start using the Jupyter notebook:

```bash
jupyter notebook demo.ipynb
```

---

### Parameters

| Argument             | Description                                   | Default      |
| -------------------- | --------------------------------------------- | ------------ |
| `--gen-type`       | Generation type:`retain` or `forgot`      | *required* |
| `--task`           | Concept/task name, e.g.,`Nudity`            | `Nudity`   |
| `--save_dir`       | Output directory for saving images            | `gen_imgs` |
| `--ddim_steps`     | Number of DDIM sampling steps                 | `50`       |
| `--gen_nums`       | Number of images to generate per prompt       | `5`        |
| `--guidance_scale` | Guidance scale for classifier-free guidance   | `7.5`      |
| `--height`         | Image height in pixels                        | `512`      |
| `--width`          | Image width in pixels                         | `512`      |
| `--replace_step`   | Timestep to apply redirector guidance         | `2`        |
| `--alpha_scale`    | Scaling factor for redirector delta injection | `1.5`      |

---

## 📌 Output Structure

Images will be saved to:

```
gen_imgs/
└── retain/ or forgot/
    └── Nudity/
        ├── 0_0_0.png
        ├── 0_0_1.png
        └── ...
```

---

## 📝 License

This project is released under the MIT License.

---

## 🙏 Acknowledgements

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [Diffusers](https://github.com/huggingface/diffusers)
