import torch
import gradio as gr
import argparse
import os
import random
import tempfile
import glob  # Added import
from PIL import Image
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype

# Check for CUDA availability
if torch.cuda.device_count() >= 1:
    SUPIR_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')

# Create SUPIR model with default settings
def load_model(supir_sign='Q', loading_half_params=False, use_tile_vae=False, 
               encoder_tile_size=512, decoder_tile_size=64, 
               ae_dtype="bf16", diff_dtype="fp16"):
    model = create_SUPIR_model('options/SUPIR_v0.yaml', SUPIR_sign=supir_sign)
    if loading_half_params:
        model = model.half()
    if use_tile_vae:
        model.init_tile_vae(encoder_tile_size=encoder_tile_size, decoder_tile_size=decoder_tile_size)
    model.ae_dtype = convert_dtype(ae_dtype)
    model.model.dtype = convert_dtype(diff_dtype)
    model = model.to(SUPIR_device)
    return model

# Function to generate random seed
def generate_random_seed():
    return random.randint(1, 2147483647)

# Process the image with SUPIR
def process_image(input_image, upscale, supir_sign, seed, min_size, edm_steps, 
                 s_stage1, s_churn, s_noise, s_cfg, s_stage2, 
                 img_caption, a_prompt, n_prompt, color_fix_type, 
                 linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2,
                 loading_half_params, use_tile_vae, encoder_tile_size, decoder_tile_size,
                 ae_dtype, diff_dtype):
    
    # Load model with specified precision and performance settings
    global model
    # Reload model if precision settings changed or not loaded yet
    if ('model' not in globals() or
        'current_settings' not in globals() or
        current_settings['supir_sign'] != supir_sign or
        current_settings['loading_half_params'] != loading_half_params or
        current_settings['use_tile_vae'] != use_tile_vae or
        current_settings['encoder_tile_size'] != encoder_tile_size or
        current_settings['decoder_tile_size'] != decoder_tile_size or
        current_settings['ae_dtype'] != ae_dtype or
        current_settings['diff_dtype'] != diff_dtype):
        
        model = load_model(
            supir_sign=supir_sign, 
            loading_half_params=loading_half_params,
            use_tile_vae=use_tile_vae,
            encoder_tile_size=encoder_tile_size,
            decoder_tile_size=decoder_tile_size,
            ae_dtype=ae_dtype,
            diff_dtype=diff_dtype
        )
        
        # Store current settings for future comparison
        globals()['current_settings'] = {
            'supir_sign': supir_sign,
            'loading_half_params': loading_half_params,
            'use_tile_vae': use_tile_vae,
            'encoder_tile_size': encoder_tile_size,
            'decoder_tile_size': decoder_tile_size,
            'ae_dtype': ae_dtype,
            'diff_dtype': diff_dtype
        }
    
    # Convert to PIL if needed
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)
    
    # Process image
    LQ_img, h0, w0 = PIL2Tensor(input_image, upsacle=upscale, min_size=min_size)
    LQ_img = LQ_img.unsqueeze(0).to(SUPIR_device)[:, :3, :, :]
    
    # Use the provided caption
    captions = [img_caption]
    
    # Run diffusion process
    samples = model.batchify_sample(LQ_img, captions, 
                                    num_steps=edm_steps, 
                                    restoration_scale=s_stage1, 
                                    s_churn=s_churn,
                                    s_noise=s_noise, 
                                    cfg_scale=s_cfg, 
                                    control_scale=s_stage2, 
                                    seed=seed,
                                    num_samples=1,  # Always 1 for UI 
                                    p_p=a_prompt, 
                                    n_p=n_prompt, 
                                    color_fix_type=color_fix_type,
                                    use_linear_CFG=linear_CFG, 
                                    use_linear_control_scale=linear_s_stage2,
                                    cfg_scale_start=spt_linear_CFG, 
                                    control_scale_start=spt_linear_s_stage2)
    
    # Convert result to PIL image
    result_img = Tensor2PIL(samples[0], h0, w0)

    # --- Save the image before returning ---
    try:
        # Ensure the output directory exists
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # Find existing gradio files in the output directory to determine the next index
        existing_files = glob.glob(os.path.join(output_dir, "gradio_*.png"))
        max_index = -1
        for f in existing_files:
            try:
                # Extract index from filename like "gradio_123.png"
                index_str = f.replace("gradio_", "").replace(".png", "")
                index = int(index_str)
                if index > max_index:
                    max_index = index
            except ValueError:
                # Ignore files that don't match the pattern
                pass

        next_index = max_index + 1
        # Construct the full save path within the output directory
        save_path = os.path.join(output_dir, f"gradio_{next_index}.png")
        result_img.save(save_path)
        print(f"Saved generated image to: {save_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
    # --- End saving logic ---

    return result_img

# Function to update tile VAE visibility
def update_tile_vae_visibility(use_tile):
    return gr.update(visible=use_tile)

# Default prompts
default_positive_prompt = 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.'
default_negative_prompt = 'painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth'

# Create the Gradio UI
def create_ui():
    with gr.Blocks(title="SUPIR Image Restoration") as demo:
        gr.Markdown("# SUPIR Image Restoration Shit")
        gr.Markdown("Upload an image to enhance/detail/restore it using the SUPIR model")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil", height=800)
                run_button = gr.Button("Enhance Image")
            
            with gr.Column():
                output_image = gr.Image(label="Enhanced Image", height=800)
        
        with gr.Accordion("Precision & Performance Settings", open=True):
            with gr.Row():
                loading_half_params = gr.Checkbox(value=True, label="Load Half Precision Parameters (use if <=24GB VRAM)")                
            
            with gr.Row():
                ae_dtype = gr.Radio(choices=["fp32", "bf16"], value="bf16", label="Autoencoder Data Type")
                diff_dtype = gr.Radio(choices=["fp32", "fp16", "bf16"], value="fp16", label="Diffusion Model Data Type")
            
            with gr.Row():
                use_tile_vae = gr.Checkbox(value=True, label="Use Tile VAE (use if <=24GB VRAM)")

            # Create a visible container for tile VAE settings
            with gr.Column(visible=False) as tile_vae_settings:
                encoder_tile_size = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Encoder Tile Size")
                decoder_tile_size = gr.Slider(minimum=32, maximum=128, value=64, step=8, label="Decoder Tile Size")
            
            # Connect the checkbox to update visibility
            use_tile_vae.change(
                fn=update_tile_vae_visibility,
                inputs=[use_tile_vae],
                outputs=[tile_vae_settings]
            )
        
        with gr.Accordion("Basic Settings", open=True):
            upscale = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Upscale Factor")
            supir_sign = gr.Radio(choices=["F", "Q"], value="Q", label="SUPIR Model : SUPIR-v0Q is robust to heavy degradations but may overcorrect clean images, while SUPIR-v0F is tuned for lighter degradations, making it better suited for high-fidelity restoration of already high-quality inputs.")
            
            # Replace seed slider with text input and add random button
            with gr.Row():
                seed = gr.Number(value=1234567891, precision=0, label="Seed")
                random_seed_button = gr.Button("🎲 Random", size="sm")
            
            min_size = gr.Slider(minimum=512, maximum=2048, value=1024, step=64, label="Minimum Size")
            img_caption = gr.Textbox(label="Image Caption (optional)", placeholder="Leave empty for no caption")
            color_fix_type = gr.Radio(choices=["None", "AdaIn", "Wavelet"], value="Wavelet", label="Color Fix Type")
        
        with gr.Accordion("Advanced Settings", open=False):
            edm_steps = gr.Slider(minimum=10, maximum=100, value=50, step=1, label="EDM Steps")
            s_stage1 = gr.Slider(minimum=-1, maximum=10, value=-1, step=1, label="Stage 1 Scale (kijai restore_cfg)")
            s_churn = gr.Slider(minimum=0, maximum=20, value=5, step=1, label="Churn")
            s_noise = gr.Slider(minimum=1.0, maximum=2.0, value=1.003, step=0.001, label="Noise")
            s_cfg = gr.Slider(minimum=1.0, maximum=15.0, value=4.0, step=0.1, label="CFG Scale (kijai cfg_scale_end)")
            s_stage2 = gr.Slider(minimum=0.0, maximum=2.0, value=0.9, step=0.1, label="Stage 2 Scale (kijai control_scale_end)")
            
            with gr.Row():
                linear_CFG = gr.Checkbox(value=True, label="Linear CFG")
                spt_linear_CFG = gr.Slider(minimum=0.0, maximum=10.0, value=2.0, step=0.1, label="Start Linear CFG (kijai cfg_scale_start)")
            
            with gr.Row():
                linear_s_stage2 = gr.Checkbox(value=False, label="Linear Stage 2")
                spt_linear_s_stage2 = gr.Slider(minimum=0.0, maximum=2.0, value=0.9, step=0.1, label="Start Linear Stage 2 (kijai control_scale_start)")

        with gr.Accordion("(Additional) Prompt Settings - these are appended to the main caption", open=False):
            a_prompt = gr.Textbox(value=default_positive_prompt, label="Positive Prompt")
            n_prompt = gr.Textbox(value=default_negative_prompt, label="Negative Prompt")
        
        # Random seed button functionality
        random_seed_button.click(
            fn=generate_random_seed,
            inputs=[],
            outputs=[seed]
        )
        
        # Connect the run button to the process_image function
        run_button.click(
            fn=process_image,
            inputs=[
                input_image, upscale, supir_sign, seed, min_size, edm_steps,
                s_stage1, s_churn, s_noise, s_cfg, s_stage2,
                img_caption, a_prompt, n_prompt, color_fix_type,
                linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2,
                loading_half_params, use_tile_vae, encoder_tile_size, decoder_tile_size,
                ae_dtype, diff_dtype
            ],
            outputs=output_image
        )
        
        gr.Markdown("## Instructions")
        gr.Markdown("""
        1. Upload an image you want to enhance
        2. Adjust the settings as needed (default settings are sufficient for most uses)
        3. Click 'Enhance Image' to process
        4. The enhanced image will appear on the right
        
        Note: Processing may take some time depending on your GPU and image size.
        """)
        
        gr.Markdown("## Precision & Performance Tips")
        gr.Markdown("""
        - **Loading Half Precision**: Reduces memory usage but may slightly reduce quality
        - **Tile VAE**: Useful for processing large images with limited VRAM
        - **Data Types**: 
          - fp32: Best quality, highest memory usage
          - bf16: Good balance of quality and memory (not supported on all GPUs)
          - fp16: Fastest, lowest memory usage
        """)
    
    return demo

# Parse command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUPIR Gradio Interface")
    parser.add_argument("--listen", action="store_true", help="Make the interface accessible on the network")
    parser.add_argument("--share", action="store_true", help="Create a shareable link")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Server name/IP to bind to")
    parser.add_argument("--server-port", type=int, default=3000, help="Port to use")
    args = parser.parse_args()
    
    # Create and launch the interface
    demo = create_ui()
    demo.launch(
        # if --listen arg is passed then bind to 0.0.0.0, otherwise default to localhost (127.0.0.1) only
        server_name=args.server_name if args.listen else None,
        server_port=args.server_port,
        share=args.share
    )
