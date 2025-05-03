Modification of [SUPIR](https://github.com/Fanghua-Yu/SUPIR) repository.

- Removed the LLaVA implementation. 
- Added safetensors support. 
- Updated dependencies. 





---
## 🔧 Dependencies and Installation

1. Clone repo
    ```bash
    git clone https://github.com/yushan777/SUPIR.git
    cd SUPIR
    ```

2. Install dependent packages
    ```bash
        python3 -m venv venv
        source venv/bin/activate
        pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
        pip install -r requirements.txt
    ```

3. Download Models

SUPIR Models: 

(FP16)
* `SUPIR-v0Q`: [FP16](https://huggingface.co/yushan777/SUPIR/resolve/main/SUPIR/SUPIR-v0Q_fp16.safetensors)
* `SUPIR-v0F`: [FP16](https://huggingface.co/yushan777/SUPIR/resolve/main/SUPIR/SUPIR-v0F_fp16.safetensors)  
Place in `models/SUPIR/`
(FP32)
* `SUPIR-v0Q`: [FP32](https://huggingface.co/yushan777/SUPIR/resolve/main/SUPIR/SUPIR-v0Q_fp32.safetensors)
* `SUPIR-v0F`: [FP32](https://huggingface.co/yushan777/SUPIR/resolve/main/SUPIR/SUPIR-v0F_fp32.safetensors)  
Place in `models/SUPIR/`

#### Dependent Models
* [CLIP Encoder-1](https://huggingface.co/yushan777/SUPIR/resolve/main/CLIP1/clip-vit-large-patch14/safetensors/clip-vit-large-patch14.safetensors)  
  Place in `models/CLIP1`
* [CLIP Encoder-2](https://huggingface.co/yushan777/SUPIR/resolve/main/CLIP2/CLIP-ViT-bigG-14-laion2B-39B-b160k/safetensors/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors)  
  Place in `models/CLIP2`
* [Juggernaut-XL_v9_RunDiffusionPhoto_v2](https://huggingface.co/yushan777/SUPIR/resolve/main/SDXL/juggernautXL_v9Rundiffusionphoto2.safetensors)  
  Place in `models/SDXL`



SUPIR-v0Q : Default training settings with paper. High generalization and high image quality in most cases.
SUPIR-v0F : Training with light degradation settings. Stage1 encoder of `SUPIR-v0F` remains more details when facing light degradations.

4. Edit Custom Path for Checkpoints
    ```
    * [options/SUPIR_v0.yaml] --> SDXL_CKPT, SUPIR_CKPT_Q, SUPIR_CKPT_F. CLIP1, CLIP2
    ```
---

## ⚡ Quick Inference
### Val Dataset
RealPhoto60: [Baidu Netdisk](https://pan.baidu.com/s/1CJKsPGtyfs8QEVCQ97voBA?pwd=aocg), [Google Drive](https://drive.google.com/drive/folders/1yELzm5SvAi9e7kPcO_jPp2XkTs4vK6aR?usp=sharing)

### Usage

```bash
python3 test.py [options]
python3 test.py --img_path 'input/bottle.png' --save_dir ./output --SUPIR_sign Q --upscale 2 --use_tile_vae --loading_half_params
```

### Required Arguments

* `--img_path`
  Path to the input image.

* `--save_dir`
  Directory to save the output.

### Optional Options

* `--upscale`
  Upsampling ratio for the input. Default: `1`

* `--SUPIR_sign`
  Model type. Options: `['F', 'Q']`. Default: `'Q'`

* `--seed`
  Random seed for reproducibility. Default: `1234`

* `--min_size`
  Minimum output resolution. Default: `1024`

* `--num_samples`
  Number of images to generate per input. Default: `1`

### Prompt Guidance

* `--a_prompt`
  Additional positive prompt (appended to input caption).
  Default:

  ```
  Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, 
  hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, 
  extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.
  ```

* `--n_prompt`
  Fixed negative prompt.
  Default:

  ```
  painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, 
  3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, 
  frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth
  ```

### Diffusion Sampling Parameters

* `--edm_steps`
  Number of diffusion steps. Default: `50`

* `--s_churn`
  Adds random noise to encourage variation.
  Default: `5`

  * `0`: No noise (deterministic)
  * `1–5`: Mild/moderate (typical use)
  * `6–10+`: Strong (rare)

* `--s_noise`
  Scales churn noise strength. Default: `1.003`

  * Slightly < 1: More stable
  * Slightly > 1: More variation

* `--s_cfg`
  Prompt guidance strength. Default: `7.5`

  * `1.0`: Weak (ignores prompt)
  * `7.5`: Strong (follows prompt closely)

### Image Structure Guidance

* `--s_stage1`
  Early-stage restoration strength.
  Default: `-1` (disabled). Typical values: `1–6`

* `--s_stage2`
  Structural guidance from input image. Default: `1.0`

  * `0.0`: Disabled
  * `0.1–0.5`: Light
  * `0.6–1.0`: Balanced (default)
  * `1.1–1.5+`: Very strong

### Color Correction

* `--color_fix_type`
  Color adjustment method. Default: `'Wavelet'`
  Options: `['None', 'AdaIn', 'Wavelet']`

### Precision Settings

* `--ae_dtype`
  Autoencoder precision. Default: `'bf16'`
  Options: `['fp32', 'bf16']`

* `--diff_dtype`
  Diffusion model precision. Default: `'fp16'`
  Options: `['fp32', 'fp16', 'bf16']`

---

### Linear Scheduling (for dynamic scaling during sampling)

* `--linear_CFG`
  Enable linear scheduling for prompt guidance (`--s_cfg`). Default: `True`

* `--spt_linear_CFG`
  Start value of `--s_cfg` when linear scheduling is enabled. Default: `4.0`

* `--linear_s_stage2`
  Enable linear scheduling for structure guidance (`--s_stage2`). Default: `False`

* `--spt_linear_s_stage2`
  Start value of `--s_stage2` if `--linear_s_stage2` is enabled. Default: `0.0`

**Purpose:**
Linear scheduling gradually adjusts values during sampling based on noise level (`sigma`).

* Prompt guidance (`CFG`) moves from `--spt_linear_CFG` (start value) → `--s_cfg` (end value)
* Structure guidance (`Control scale`) moves from `--spt_linear_s_stage2` (start value) → `--s_stage2` (end value)

See below for mappings to Kijai's SUPIR custom nodes
---

### ComfyUI Mapping (Kijai's Node)

| ComfyUI Parameter     | Equivalent CLI Option   |
| --------------------- | ----------------------- |
| `cfg_scale_start`     | `--spt_linear_CFG`      |
| `cfg_scale_end`       | `--s_cfg`               |
| `control_scale_start` | `--spt_linear_s_stage2` |
| `control_scale_end`   | `--s_stage2`            |
| `restore_cfg`         | `--s_stage1`            |                       

* Setting both start and end to the same value disables linear scheduling.
* Setting different values enables dynamic scheduling.

#### Example Behaviors:

* `control_scale_start=1.0`, `control_scale_end=0.0`:
  Strong guidance early, fading out for prompt-driven details later.

* `control_scale_start=0.0`, `control_scale_end=1.0`:
  No early structural bias, gradually increasing structure influence.

---

### Python Script


