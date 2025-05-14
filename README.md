# A Customized Version of the Original [SUPIR](https://github.com/Fanghua-Yu/SUPIR) Project
# 🚧 Work in Progress

- Removed the heavy LLaVA implementation. 
- Added safetensors support. 
- Updated dependencies. 
- Replaced SoftMax with SDPA for default attention.
- Removed `use_linear_control_scale (linear_s_stage2)` and `use_linear_cfg_scale (linear_CFG)` arguments.  
   - Uses the start and end scale values to determine whether linear scaling will be used/have effect or not.
- Renamed arguments to make settings a bit more intuitive (more alignment with kijai's SUPIR ComfyUI custom nodes)
- Added `--skip_denoise_stage` argument to skip the initial denoising stage that softens the input image to smooth out image or compression artefacts. You might want to do this if your image is already high quality. 

---
## 🔧 Dependencies and Installation

1. Clone repo
    ```bash
    git clone https://github.com/yushan777/SUPIR.git
    cd SUPIR
    ```
## Vast.ai Install (will automatically download models)
    ```bash
    chmod +x install_vastai.sh
    ./install_vastai.sh
    ```


## Manual Local Install
1. Install dependent packages
    ```bash
        python3 -m venv venv
        source venv/bin/activate
        pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu126
        # pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
        pip install -r requirements.txt
    ```
    
2. Download Models

---

#### SUPIR Models

Download and place the model files in the `models/SUPIR/` directory.

**FP16 Versions**

* [`SUPIR-v0Q (FP16)`](https://huggingface.co/yushan777/SUPIR/resolve/main/SUPIR/SUPIR-v0Q_fp16.safetensors)
* [`SUPIR-v0F (FP16)`](https://huggingface.co/yushan777/SUPIR/resolve/main/SUPIR/SUPIR-v0F_fp16.safetensors)

**FP32 Versions**

* [`SUPIR-v0Q (FP32)`](https://huggingface.co/yushan777/SUPIR/resolve/main/SUPIR/SUPIR-v0Q_fp32.safetensors)
* [`SUPIR-v0F (FP32)`](https://huggingface.co/yushan777/SUPIR/resolve/main/SUPIR/SUPIR-v0F_fp32.safetensors)

There are two SUPIR model variants: v0Q and v0F. 

* **SUPIR-v0Q**
The v0Q model (Quality) is trained on a wide range of degradations, making it robust and effective across varied real-world scenarios. However, this broad generalization comes at a cost—when applied to images with only mild degradation, v0Q might overcompensate, hallucinate or alter details that are already mostly intact. This behavior stems from its training bias toward assuming significant visual damage. 

* **SUPIR-v0F**
In contrast, the v0F model (Fidelity) is specifically trained on lighter degradation patterns. Its Stage1 encoder is tuned to better preserve fine details and structure, resulting in restorations that are more faithful to the input when the degradation is minimal. As a result, v0F is the preferred choice for high-fidelity restoration where subtle preservation is more critical than aggressive enhancement.


#### Dependent Models
* [CLIP Encoder-1](https://huggingface.co/yushan777/SUPIR/resolve/main/CLIP1/clip-vit-large-patch14/safetensors/clip-vit-large-patch14.safetensors)  
  Place in `models/CLIP1`
* [CLIP Encoder-2](https://huggingface.co/yushan777/SUPIR/resolve/main/CLIP2/CLIP-ViT-bigG-14-laion2B-39B-b160k/safetensors/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors)  
  Place in `models/CLIP2`
* [Juggernaut-XL_v9_RunDiffusionPhoto_v2](https://huggingface.co/yushan777/SUPIR/resolve/main/SDXL/juggernautXL_v9Rundiffusionphoto2.safetensors)  
  Place in `models/SDXL`

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
# for gradio test
python3 run_supir_gradio.py --listen

# for cli test
python3 run_supir.py [options]
python3 run_supir.py --img_path 'input/bottle.png' --save_dir ./output --SUPIR_sign Q --upscale 2 --use_tile_vae --loading_half_params

python3 run_supir.py \
--img_path 'input/woman-low-res.jpg' \
--save_dir ./output \
--SUPIR_sign Q \
--upscale 2 \
--seed 1234567891 \
--img_caption 'A woman has dark brown eyes, dark curly hair wearing a dark scarf on her head. She is wearing a black shirt on with a pattern on it. The wall behind her is brown and green.' \
--edm_steps=50 \
--s_churn=5 \
--cfg_scale_start=2.0 \
--cfg_scale_end=4.0 \
--control_scale_start=0.9 \
--control_scale_end=0.9 \
--loading_half_params \
--use_tile_vae

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

* `--skip_denoise_stage`
  Skips the VAE Denoiser Stage.  
  Default: `'False'`
  The denoise stage softens the input image to smooth out artefacts that is typical of low quality or low resolution images in order to prevent SUPIR from processing them as details. If your input image is already of high quality then you can enable this. 

* `--sampler_mode`
  Sampler choice",  Options: `['TiledRestoreEDMSampler', 'RestoreEDMSampler']`. Default: `'TiledRestoreEDMSampler'`

* `--seed`
  Random seed for reproducibility. Default: `1234`

* `--min_size`
  Minimum output resolution. Default: `1024`

* `--num_samples`
  Number of images to generate per input. Default: `1`

### Prompt Guidance

* `--img_caption`
  Specific caption for the input image.
  Default: `''` 
  This caption is combined with `--a_prompt`.

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

* `--cfg_scale_start` 
  Prompt guidance strength start
  Default: `2.0`

* `--cfg_scale_end`
  Prompt guidance strength end. 
  Default: `4`

  * `1.0`: Weak (ignores prompt)
  * `7.5`: Strong (follows prompt closely)

  If `--cfg_scale_start` and `--cfg_scale_end` have the same value, no scaling occurs - 
  When these values differ, then linear scheduling is applied, 
  starting from `--cfg_scale_start` and gradually changing to `--cfg_scale_end` during processing.

  `--cfg_scale_start` doesn't have to be lower than `--cfg_scale_end`.  
  They can start strong and finish weaker
   - More creative freedom early in the process while enforcing prompt adherence later
  or start weaker and finish strong
   - Establish initial structure from the prompt while allowing more freedom for details later
  
### Image Structure Guidance

* `--restoration_scale`
  Early-stage restoration strength .
  Default: `-1` (disabled). 
  Typical values: `1–6`

* `--control_scale_start`
  Structural guidance from input image. Default: `1.0`

* `--control_scale_end` 
  Structural guidance from input image. Default: `1.0`

  * `0.0`: Disabled
  * `0.1–0.5`: Light
  * `0.6–1.0`: Balanced (default)
  * `1.1–1.5+`: Very strong

  If `--control_scale_start` and `--control_scale_end` have the same value, no scaling occurs - 
  That single value throughout the process. When these values differ, then linear scheduling is applied, 
  starting from `--control_scale_start` and gradually changing to `--control_scale_end` during processing.

  `--control_scale_start` doesn't have to be lower than `--control_scale_end`.  
  They can start strong and finish weaker
   - Starts with stronger adherence to the structure of the original image
   - Gradually gives the model more freedom to deviate and enhance details
  or start weaker and finish strong
   - Starts with more creative freedom
   - Gradually increases control to ensure the final result maintains key structural elements

### Color Correction

* `--color_fix_type`
  Color adjustment method. Default: `'Wavelet'`
  Options: `['None', 'AdaIn', 'Wavelet']`

### Precision/Performance Settings

* `--loading_half_params`
  loads the SUPIR model weights in half precision (FP16). 
  Default: `False`
  Reduces VRAM usage and increases speed at the cost of slight precision loss.

* `--diff_dtype`  
  Precision to use for the diffusion model only (e.g., UNet).  
  Allows overriding the default precision **independently of model-wide settings**, unless `--loading_half_params` is set.  
  If `--loading_half_params` is enabled, this setting will have no effect.  
  Default: `'fp16'`  
  Options: `['fp32', 'fp16', 'bf16']`

* `--ae_dtype`
  Autoencoder precision. Default: `'bf16'`
  Options: `['fp32', 'bf16']`

* `--use_tile_vae`
  Enables tile-based encoding/decoding for memory efficiency with large images
  Default: `False`

* `--encoder_tile_size` / `--decoder_tile_size`  
  Tile sizes (in pixels) used when use_tile_vae is enabled.
  Encoder defaults to 512, decoder to 64.



