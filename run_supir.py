import torch.cuda
import argparse
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
from PIL import Image
import os
from torch.nn.functional import interpolate
import time 

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True, help="Path to the input image file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the output image")
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--SUPIR_sign", type=str, default='Q', choices=['F', 'Q'])
    parser.add_argument("--sampler_mode", type=str, default='TiledRestoreEDMSampler', choices=['TiledRestoreEDMSampler', 'RestoreEDMSampler'])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min_size", type=int, default=1024)
    parser.add_argument("--edm_steps", type=int, default=50)
    parser.add_argument("--restoration_scale", type=int, default=-1) #renamed from s_stage1
    parser.add_argument("--s_churn", type=int, default=5)
    parser.add_argument("--s_noise", type=float, default=1.003)
    parser.add_argument("--cfg_scale_end", type=float, default=4.0) #renamed from s_cfg
    parser.add_argument("--control_scale_end", type=float, default=1.0) #renamed from s_stage2
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--img_caption", type=str, default='', help="Specific caption for the input image.")
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
    parser.add_argument("--cfg_scale_start", type=float, default=2.0) #renamed from spt_linear_CFG
    parser.add_argument("--control_scale_start", type=float, default=0.9) #renamed from spt_linear_s_stage2

    parser.add_argument("--loading_half_params", action='store_true', default=False) # load model weights in fp16
    parser.add_argument("--ae_dtype", type=str, default="bf16", choices=['fp32', 'bf16'])
    parser.add_argument("--diff_dtype", type=str, default="fp16", choices=['fp32', 'fp16', 'bf16'])
    
    parser.add_argument("--use_tile_vae", action='store_true', default=False)
    parser.add_argument("--encoder_tile_size", type=int, default=512)
    parser.add_argument("--decoder_tile_size", type=int, default=64)

    parser.add_argument("--skip_denoise_stage", action='store_true', default=False)
    

    return parser.parse_args()

# =====================================================================
def setup_model(args, device):
    # Load SUPIR model
    if args.sampler_mode == "TiledRestoreEDMSampler":
        config = "options/SUPIR_v0_tiled.yaml"
    else:
        config = "options/SUPIR_v0.yaml"

    model = create_SUPIR_model(config, SUPIR_sign=args.SUPIR_sign)

    if args.loading_half_params:
        model = model.half()
    if args.use_tile_vae:
        model.init_tile_vae(encoder_tile_size=args.encoder_tile_size, decoder_tile_size=args.decoder_tile_size)
    
    model.ae_dtype = convert_dtype(args.ae_dtype)
    model.model.dtype = convert_dtype(args.diff_dtype)
    model = model.to(device)
    
    return model

# =====================================================================
def process_image(model, args, device):
    img_path = args.img_path
    img_name = os.path.splitext(os.path.basename(img_path))[0]  # Get base filename without extension
    
    try:
        LQ_ips = Image.open(img_path)
    except FileNotFoundError:
        print(f"Error: Input image not found at {img_path}")
        return None
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return None
    
    # update the input image
    LQ_img, h0, w0 = PIL2Tensor(LQ_ips, upsacle=args.upscale, min_size=args.min_size)
    LQ_img = LQ_img.unsqueeze(0).to(device)[:, :3, :, :]
    
    # Image caption(s)
    # captions = [args.img_caption]
    
    # Diffusion Process
    # batchify_sample() is in SUPIR/models/SUPIR_model.py
    samples = model.batchify_sample(LQ_img, args.img_caption, 
                                    num_steps=args.edm_steps, 
                                    restoration_scale=args.restoration_scale, 
                                    s_churn=args.s_churn,
                                    s_noise=args.s_noise,
                                    cfg_scale_start=args.cfg_scale_start,                                     
                                    cfg_scale_end=args.cfg_scale_end, 
                                    control_scale_start=args.control_scale_start,
                                    control_scale_end=args.control_scale_end, 
                                    seed=args.seed,
                                    num_samples=args.num_samples, 
                                    p_p=args.a_prompt, 
                                    n_p=args.n_prompt, 
                                    color_fix_type=args.color_fix_type,
                                    skip_denoise_stage=args.skip_denoise_stage)
    
    return samples, h0, w0, img_name

# =====================================================================
def save_results(samples, h0, w0, img_name, save_dir):
    output_base_name = f"{img_name}_SUPIR"  # Construct a base name for output
    saved_paths = []
    
    for _i, sample in enumerate(samples):
        # Determine initial filename
        base_filename = f'{output_base_name}_{_i}' if len(samples) > 1 else output_base_name
        extension = '.png'
        output_filename = f'{base_filename}{extension}'
        save_path = os.path.join(save_dir, output_filename)
        
        # Check if file exists and append index if necessary
        counter = 1
        while os.path.exists(save_path):
            output_filename = f'{base_filename}_{counter}{extension}'
            save_path = os.path.join(save_dir, output_filename)
            counter += 1
        
        # Save the image
        Tensor2PIL(sample, h0, w0).save(save_path)
        saved_paths.append(save_path)
        print(f"Saved output image to: {save_path}")
    
    return saved_paths

# =====================================================================
def format_elapsed_time(seconds):
    formatted_time = f"Process time: {seconds:.4f} seconds."
    
    # If over 60 seconds, add hours:minutes:seconds format in parentheses
    if seconds >= 60:
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        time_str = f"{hours:02d}:{minutes:02d}:{secs:02d}"
        formatted_time += f" ({time_str})"
    
    return formatted_time

# =====================================================================
def main():

    # Check for CUDA availability
    if torch.cuda.device_count() >= 1:
        device = 'cuda:0'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("CUDA not available. Falling back to CPU. Warning: This will be significantly slower.")

    # for tracking execution time
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    print(args)
    
    # Create output directory if not exist
    os.makedirs(args.save_dir, exist_ok=True)
        
    # Setup SUPIR model
    SUPIR_model = setup_model(args, device)
    
    # Process the image
    result = process_image(SUPIR_model, args, device)
    if result is not None:
        samples, h0, w0, img_name = result
        
        # Save results
        save_results(samples, h0, w0, img_name, args.save_dir)
    
    # Print elapsed time
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    print(format_elapsed_time(elapsed_seconds))

# =====================================================================
if __name__ == "__main__":
    main()