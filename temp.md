| **Argument** | **Description** |
|--------------|-----------------|
| `--img_path` | Path to the input image. **(required)** |
| `--save_dir` | Directory to save the output. |
| `--SUPIR_sign` | Model type. Options: `['F', 'Q']`<br>Default: `'Q'`<br>Q model (Quality) Trained on diverse, heavy degradations, making it robust for real-world damage. However, it may overcorrect or hallucinate when used on lightly degraded images due to its bias toward severe restoration.<br>F model (Fidelity) Optimized for mild degradations, preserving fine details and structure. Ideal for high-fidelity tasks where subtle restoration is preferred over aggressive enhancement. |
| `--skip_denoise_stage` | Skips the VAE Denoiser Stage. Default: `'False'`<br>The denoise stage softens the input image to smooth out artefacts that is typical of low quality or low resolution images in order to prevent SUPIR from processing them as details. If your input image is already of high quality then you can enable this. |
| `--sampler_mode` | Sampler choice. Options: `['TiledRestoreEDMSampler', 'RestoreEDMSampler']`<br>Default: `'TiledRestoreEDMSampler' (uses less VRAM)` |
| `--seed` | Random seed for reproducibility. Default: `1234` |
| `--upscale` | Upsampling ratio for the input.<br>The higher the scale factor, the slower the process.<br>Default: `2` |
| `--min_size` | Minimum output resolution. Default: `1024` |
| `--num_samples` | Number of images to generate per input. Default: `1` |
| `--img_caption` | Specific caption for the input image.<br>Default: `''`<br>This caption is combined with `--a_prompt`. |
| `--a_prompt` | Additional positive prompt (appended to input caption).<br>Default:<br>```Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.``` |
| `--n_prompt` | Negative prompt.<br>Default:<br>```painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth``` |
| `--edm_steps` | Number of diffusion steps. Default: `50` |
| `--s_churn` | Adds random noise to encourage variation. Default: `5`<br>`0`: No noise (deterministic)<br>`1–5`: Mild/moderate<br>`6–10+`: Strong |
| `--s_noise` | Scales churn noise strength. Default: `1.003`<br>Slightly < 1: More stable<br>Slightly > 1: More variation |
| `--cfg_scale_start` | Prompt guidance strength start.<br>Default: `2.0` |
| `--cfg_scale_end` | Prompt guidance strength end.<br>Default: `4`<br>`1.0`: Weak (ignores prompt)<br>`7.5`: Strong (follows prompt closely)<br>If `--cfg_scale_start` and `--cfg_scale_end` have the same value, no scaling occurs. When these values differ, linear scheduling is applied from start to end. They can also be reversed for creative strategies. |
| `--control_scale_start` | Structural guidance from input image, start strength. Default: `0.9` |
| `--control_scale_end` | Structural guidance from input image, end strength. Default: `0.9`<br>`0.0`: Disabled<br>`0.1–0.5`: Light<br>`0.6–1.0`: Balanced (default)<br>`1.1–1.5+`: Very strong<br>Same value = fixed. Different values = scheduled. |
| `--restoration_scale` | Early-stage restoration strength.<br>Works as an additional guidance mechanism beyond the control scale.<br>Targets fine details and textures.<br>Default: `-1` (disabled).<br>Typical values: `1–6` |
| `--color_fix_type` | Color adjustment method. Default: `'Wavelet'`<br>Options: `['None', 'AdaIn', 'Wavelet']` |
| `--loading_half_params` | Loads the SUPIR model weights in half precision (FP16).<br>Default: `False`<br>Reduces VRAM usage and increases speed at the cost of slight precision loss. |
| `--diff_dtype` | Precision to use for the diffusion model only.<br>Allows overriding default precision independently, unless `--loading_half_params` is set.<br>Default: `'fp16'`<br>Options: `['fp32', 'fp16', 'bf16']` |
| `--ae_dtype` | Autoencoder precision.<br>Default: `'bf16'`<br>Options: `['fp32', 'bf16']` |
| `--use_tile_vae` | Enables tile-based encoding/decoding for memory efficiency with large images.<br>Default: `False` |
| `--encoder_tile_size` / `--decoder_tile_size` | Tile sizes (in pixels) used when `use_tile_vae` is enabled.<br>Encoder defaults to 512, decoder to 64. |
| `--sampler_tile_size` | Tile size for `TiledRestoreEDMSampler`.<br>This is the size of each tile that the image is divided into during tiled sampling.<br>Example: `tile_size` of 128 → image is split into 128×128 pixel tiles. |
| `--sampler_tile_stride` | Tile stride for `TiledRestoreEDMSampler`.<br>Controls overlap between tiles during sampling.<br>Smaller `tile_stride` = more overlap, better blending, more compute.<br>Larger `tile_stride` = less overlap, faster, may cause seams.<br>`Overlap = tile_size - tile_stride`<br>Examples:<br>- tile_size = 128, stride = 64 → 64 px overlap. |
