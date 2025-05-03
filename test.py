import torch.cuda
import argparse
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
from PIL import Image
# from llava.llava_agent import LLavaAgent # Removed LLaVA
# from CKPT_PTH import LLAVA_MODEL_PATH # Removed LLaVA
import os
from torch.nn.functional import interpolate

if torch.cuda.device_count() >= 1: # Adjusted device logic
    SUPIR_device = 'cuda:0'
    # LLaVA_device = 'cuda:1' if torch.cuda.device_count() >= 2 else 'cuda:0' # Removed LLaVA
else:
    raise ValueError('Currently support CUDA only.')

# hyparams here
parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, required=True, help="Path to the input image file") # Changed from img_dir
parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the output image") # Made save_dir required for clarity
parser.add_argument("--upscale", type=int, default=1)
parser.add_argument("--SUPIR_sign", type=str, default='Q', choices=['F', 'Q'])
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--min_size", type=int, default=1024)
parser.add_argument("--edm_steps", type=int, default=50)
parser.add_argument("--s_stage1", type=int, default=-1)
parser.add_argument("--s_churn", type=int, default=5)
parser.add_argument("--s_noise", type=float, default=1.003)
parser.add_argument("--s_cfg", type=float, default=7.5)
parser.add_argument("--s_stage2", type=float, default=1.)
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--a_prompt", type=str,
                    default='Cinematic, High Contrast, highly detailed, taken using a Canon EOS R '
                            'camera, hyper detailed photo - realistic maximum detail, 32k, Color '
                            'Grading, ultra HD, extreme meticulous detailing, skin pore detailing, '
                            'hyper sharpness, perfect without deformations.')
parser.add_argument("--n_prompt", type=str,
                    default='painting, oil painting, illustration, drawing, art, sketch, oil painting, '
                            'cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, '
                            'worst quality, low quality, frames, watermark, signature, jpeg artifacts, '
                            'deformed, lowres, over-smooth')
parser.add_argument("--color_fix_type", type=str, default='Wavelet', choices=["None", "AdaIn", "Wavelet"])
parser.add_argument("--linear_CFG", action='store_true', default=True)
parser.add_argument("--linear_s_stage2", action='store_true', default=False)
parser.add_argument("--spt_linear_CFG", type=float, default=4.0)
parser.add_argument("--spt_linear_s_stage2", type=float, default=0.)
parser.add_argument("--ae_dtype", type=str, default="bf16", choices=['fp32', 'bf16'])
parser.add_argument("--diff_dtype", type=str, default="fp16", choices=['fp32', 'fp16', 'bf16'])
# parser.add_argument("--no_llava", action='store_true', default=False) # Removed LLaVA
parser.add_argument("--loading_half_params", action='store_true', default=False)
parser.add_argument("--use_tile_vae", action='store_true', default=False)
parser.add_argument("--encoder_tile_size", type=int, default=512)
parser.add_argument("--decoder_tile_size", type=int, default=64)
# parser.add_argument("--load_8bit_llava", action='store_true', default=False) # Removed LLaVA
args = parser.parse_args()
print(args)
# use_llava = not args.no_llava # Removed LLaVA

# load SUPIR
model = create_SUPIR_model('options/SUPIR_v0.yaml', SUPIR_sign=args.SUPIR_sign)
if args.loading_half_params:
    model = model.half()
if args.use_tile_vae:
    model.init_tile_vae(encoder_tile_size=args.encoder_tile_size, decoder_tile_size=args.decoder_tile_size)
model.ae_dtype = convert_dtype(args.ae_dtype)
model.model.dtype = convert_dtype(args.diff_dtype)
model = model.to(SUPIR_device)
# load LLaVA # Removed LLaVA
# if use_llava:
#     llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=LLaVA_device, load_8bit=args.load_8bit_llava, load_4bit=False)
# else:
#     llava_agent = None
llava_agent = None # Ensure llava_agent is defined as None

os.makedirs(args.save_dir, exist_ok=True)

# Removed directory iteration loop
# for img_pth in os.listdir(args.img_dir):
img_path = args.img_path
img_name = os.path.splitext(os.path.basename(img_path))[0] # Get base filename without extension

try:
    LQ_ips = Image.open(img_path)
except FileNotFoundError:
    print(f"Error: Input image not found at {img_path}")
    exit()
except Exception as e:
    print(f"Error opening image {img_path}: {e}")
    exit()

LQ_img, h0, w0 = PIL2Tensor(LQ_ips, upsacle=args.upscale, min_size=args.min_size)
LQ_img = LQ_img.unsqueeze(0).to(SUPIR_device)[:, :3, :, :]

# Removed LLaVA pre-denoise step
# step 1: Pre-denoise for LLaVA, resize to 512 
# LQ_img_512, h1, w1 = PIL2Tensor(LQ_ips, upsacle=args.upscale, min_size=args.min_size, fix_resize=512)
# LQ_img_512 = LQ_img_512.unsqueeze(0).to(SUPIR_device)[:, :3, :, :]
# clean_imgs = model.batchify_denoise(LQ_img_512)
# clean_PIL_img = Tensor2PIL(clean_imgs[0], h1, w1)

 # Removed LLaVA caption generation
# step 2: LLaVA
# if use_llava:
#     captions = llava_agent.gen_image_caption([clean_PIL_img])
# else:
#     captions = ['']
captions = [''] # Use empty caption as LLaVA is removed
print(f"Using empty caption: {captions}")

# step 3: Diffusion Process
samples = model.batchify_sample(LQ_img, captions, 
                                num_steps=args.edm_steps, 
                                restoration_scale=args.s_stage1, 
                                s_churn=args.s_churn,
                                s_noise=args.s_noise, 
                                cfg_scale=args.s_cfg, 
                                control_scale=args.s_stage2, 
                                seed=args.seed,
                                num_samples=args.num_samples, 
                                p_p=args.a_prompt, 
                                n_p=args.n_prompt, 
                                color_fix_type=args.color_fix_type,
                                use_linear_CFG=args.linear_CFG, 
                                use_linear_control_scale=args.linear_s_stage2,
                                cfg_scale_start=args.spt_linear_CFG, 
                                control_scale_start=args.spt_linear_s_stage2)

# save
output_base_name = f"{img_name}_SUPIR" # Construct a base name for output
for _i, sample in enumerate(samples):
    output_filename = f'{output_base_name}_{_i}.png' if args.num_samples > 1 else f'{output_base_name}.png'
    save_path = os.path.join(args.save_dir, output_filename)
    Tensor2PIL(sample, h0, w0).save(save_path)
    print(f"Saved output image to: {save_path}")

"""
Usage: 
-- python test.py [options] 

--img_dir                Input folder.
--save_dir               Output folder.
--upscale                Upsampling ratio of given inputs. Default: 1
--SUPIR_sign             Model selection. Default: 'Q'; Options: ['F', 'Q']
--seed                   Random seed. Default: 1234
--min_size               Minimum resolution of output images. Default: 1024
--edm_steps              Number of steps for EDM Sampling Scheduler. Default: 50
--s_stage1               Control Strength of Stage1. Default: -1 (negative means invalid)
--s_churn                Original hy-param of EDM. Default: 5
--s_noise                Original hy-param of EDM. Default: 1.003
--s_cfg                  Classifier-free guidance scale for prompts. Default: 7.5
--s_stage2               Control Strength of Stage2. Default: 1.0
--num_samples            Number of samples for each input. Default: 1
--a_prompt               Additive positive prompt for all inputs. 
    Default: 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, 
    hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme
     meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.'
--n_prompt               Fixed negative prompt for all inputs. 
    Default: 'painting, oil painting, illustration, drawing, art, sketch, oil painting, 
    cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, 
    low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth'
--color_fix_type         Color Fixing Type. Default: 'Wavelet'; Options: ['None', 'AdaIn', 'Wavelet']
--linear_CFG             Linearly (with sigma) increase CFG from 'spt_linear_CFG' to s_cfg. Default: True
--linear_s_stage2        Linearly (with sigma) increase s_stage2 from 'spt_linear_s_stage2' to s_stage2. Default: False
--spt_linear_CFG         Start point of linearly increasing CFG. Default: 4.0
--spt_linear_s_stage2    Start point of linearly increasing s_stage2. Default: 0.0
--ae_dtype               Inference data type of AutoEncoder. Default: 'bf16'; Options: ['fp32', 'bf16']
--diff_dtype             Inference data type of Diffusion. Default: 'fp16'; Options: ['fp32', 'fp16', 'bf16']


--s_stage1:
Controls restore_cfg (also called this in kijai's node) parameter, a restoration guidance feature in the diffusion process. 
It’s used only in the early steps when the image is still very noisy. If enabled (with a positive number), it pulls the 
generated image closer to the structure of the original low-quality input. The higher the value, the stronger the pull 
towards the input image's structure. By default (--s_stage1 = -1), and is turned off.
Typical values: 1-6

--s_churn:
Adds extra noise ("churn") during diffusion steps to boost image diversity and prevent the model from getting stuck.
At most steps, it adds random noise to the image and slightly increases the noise level before denoising. 
This is known as "stochastic churning."
Default value is 5
0	   : No extra noise (deterministic)	Good for repeatability
1-5	   : Mild to moderate churning	Most common, default = 5
6-10   : Stronger effect	Good for short sampling schedules
10-20+ : Very strong (mostly capped)	Diminishing returns; rare use

--s_noise:
Fine-tunes the strength of the extra noise added during churning (when --s_churn > 0).
It scales the random noise used in churning steps.
1.0	Standard noise	Neutral baseline
0.99-0.999	Slightly reduced noise	More stable, less stochastic
1.001-1.01	Slightly increased noise	Default is 1.003; enhances variation subtly
Default is 1.003

--s_cfg
Controls how strongly the model follows your text prompt during generatio
It blends the model's unprompted and prompted predictions.
Higher values make the output closer to your prompt, lower values allow more freedom and variation.
1.0	Very weak guidance	Often ignores prompt
2.0-6.0	Light to moderate guidance	More freedom, less strict
7.0-10.0	Strong guidance (default = 7.5)	Follows prompt well
10-15	Very strong guidance	May reduce variation or add artifacts
> 15	Extremely strong (not recommended)	Can cause overfitting to prompt or image artifacts

--s_stage2:
Controls how strongly the model follows the structure of the input image, using a control network (like ControlNet or GLVControl)
Higher values = more faithful to the input image
Lower values = more freedom to invent new structure based on the text prompt
0.0	Disables control network	Only text prompt influences the output
0.1-0.5	Weak structural guidance	Allows creativity, looser structure
0.6-1.0	Moderate to strong guidance (default = 1.0)	Good balance between structure and prompt
1.1-1.5	Very strong structural guidance	Closely follows input image
> 1.5	Extremely rigid	May cause artifacts, ignores prompt styling

--num_samples:
Sets how many images are generated per input in a single run.
Lets you create multiple variations of the output from the same input and prompt — great for picking the best result.
> 1 variability will depend on seed, s_churn

--a_prompt:
an additional positive prompt that is appended to the main positive prompt (formerly generated by llava captioning) but now currently an empty caption.

--n_prompt:
Fixed negative prompt for all inputs

--color_fix_type:
Controls how the colors of the final image are adjusted after generation, using the input image as a reference.
The diffusion process can shift colors — this option helps match the output's colors back to the original image.
None	No color correction — output may differ in tone or hue
AdaIn	Uses Adaptive Instance Normalization to transfer color stats (mean/var)
Wavelet	(Default) Uses a wavelet-based method to better preserve color fidelity


--linear_CFG
--linear_s_stage2
--spt_linear_CFG
--spt_linear_s_stage2

These four arguments control optional linear scheduling for the prompt guidance (CFG) and the input image guidance (Control scale) during the sampling process.

--linear_CFG (Default: True)
Function: This flag enables linear scheduling for the CFG scale (prompt guidance strength) by default.
Behavior: When enabled, the CFG scale changes linearly throughout the sampling steps, depending on the noise level (sigma). It starts at the value set by --spt_linear_CFG and moves towards the final value set by --s_cfg.
Default Effect: With defaults (--spt_linear_CFG 4.0 and --s_cfg 7.5), the prompt guidance starts weaker (4.0) when noise is high (early steps) and increases linearly to the final strength (7.5) as noise decreases (later steps).
Purpose: This strategy applies weaker prompt guidance initially, allowing for more exploration, and strengthens it towards the end to refine the image according to the prompt.

--spt_linear_CFG (Default: 4.0)
Function: Sets the starting point value for the CFG scale when --linear_CFG is enabled.
Usage: This is the CFG value used at the beginning of the sampling process (highest noise). With the default settings, it's 4.0.

--linear_s_stage2 (Default: False)
Function: This flag disables linear scheduling for the Control scale (input image guidance strength, --s_stage2) by default.
Behavior: If enabled (by adding --linear_s_stage2 to the command), the Control scale would change linearly during sampling, starting from the value set by --spt_linear_s_stage2 and moving towards the final value set by --s_stage2.
Purpose: Allows for dynamic adjustment of the input image's structural influence during generation.

--spt_linear_s_stage2 (Default: 0.0)
Function: Sets the starting point value for the Control scale only if --linear_s_stage2 is enabled.
Usage: If you enable linear scheduling for the Control scale, you would set this value. For example, to decrease control strength from 1.2 to 1.0, you'd use --linear_s_stage2 --spt_linear_s_stage2 1.2 --s_stage2 1.0. To increase it from 0.8 to 1.0, you'd use --linear_s_stage2 --spt_linear_s_stage2 0.8 --s_stage2 1.0. The default 0.0 is only relevant if --linear_s_stage2 is active.

In summary (based on defaults): By default, only the CFG scale uses linear scheduling (--linear_CFG True). 
It starts at 4.0 (--spt_linear_CFG) and increases to 7.5 (--s_cfg) during sampling. 
The Control scale remains constant at 1.0 (--s_stage2) because --linear_s_stage2 is False.

In Kijai's ComfyUI custom nodes for SUPIR, this is how they map:

* `cfg_scale_start` (ComfyUI) corresponds to `--spt_linear_CFG` (test.py)
  This sets the initial value for the CFG scale (prompt guidance) at the beginning of sampling (high noise).

* `cfg_scale_end` (ComfyUI) corresponds to `--s_cfg` (test.py)
  This sets the final value for the CFG scale at the end of sampling (low noise).

* `control_scale_start` (ComfyUI) corresponds to `--spt_linear_s_stage2` (test.py)
  This sets the initial value for the Control scale (input image guidance) at the beginning of sampling.

* `control_scale_end` (ComfyUI) corresponds to `--s_stage2` (test.py)
  This sets the final value for the Control scale at the end of sampling.

  ==================================
The ComfyUI node version of SUPIR simplifies the interface. Instead of separate flags (`--linear_CFG`, `--linear_s_stage2`) 
to enable or disable linear scheduling and separate arguments for start and end points, it directly asks 
for the `start` and `end` values for both scales.

If you set `cfg_scale_start` equal to `cfg_scale_end` in ComfyUI, it's equivalent to using a constant CFG scale in the script 
(`--linear_CFG False` or just setting `--s_cfg`).

If you set `cfg_scale_start` different from `cfg_scale_end`, it's equivalent to enabling linear CFG scheduling in the script 
(`--linear_CFG True`) and using `--spt_linear_CFG` for the start and `--s_cfg` for the end.

The same logic applies to `control_scale_start` and `control_scale_end`, 
which correspond to `--linear_s_stage2`, `--spt_linear_s_stage2`, and `--s_stage2`.

so if control_scale_start and control_scale_end are 0, then it is effectively off.

If either value is greater than zero then it is on. 

==================================
If you set `control_scale_start = 1.0` and `control_scale_end = 0.0` in the ComfyUI node:

The Control scale (input image guidance strength) will start at 1.0 at the beginning of the sampling process (when noise is high).

Then, as the sampling progresses and the noise level decreases, the Control scale will linearly decrease step-by-step.

By the end of the sampling process (when noise is lowest), the Control scale will have reached 0.0.

Effect:This means the structural guidance from the input image is strongest at the beginning, helping to lock in the overall composition, and then its influence gradually fades out, potentially allowing the text prompt (CFG guidance) or the base model's tendencies more freedom to influence the finer details and textures in the later stages of generation.

==================================
If you set control_scale_start = 0.0 and control_scale_end = 1.0 in the ComfyUI node:

The Control scale (input image guidance strength) will start at 0.0 at the beginning of the sampling process (when noise is high).
As sampling progresses and the noise decreases, the Control scale will increase linearly step-by-step.
By the end of the sampling process (when noise is lowest), the Control scale will reach 1.0.

Effect: This means the input image has no structural influence at the start, allowing the prompt (or base model) more freedom to define the overall composition and style. As the sampling continues, the structure from the input image gradually becomes more influential, helping refine and align the final output with the source image in the later stages.
"""