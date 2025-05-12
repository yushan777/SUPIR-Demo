import torch
import gradio as gr
import argparse
import os
import random
import tempfile
import glob  # Added import
from PIL import Image
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
import time
from Y7.colored_print import color, style

# Check for CUDA availability
if torch.cuda.device_count() >= 1:
    SUPIR_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')

# Create SUPIR model with specified settings
def load_model(config_path, supir_sign='Q', loading_half_params=False, use_tile_vae=False,
               encoder_tile_size=512, decoder_tile_size=64,
               ae_dtype="bf16", diff_dtype="fp16"):
    print(f"Loading SUPIR model from config: {config_path}")
    model = create_SUPIR_model(config_path, SUPIR_sign=supir_sign)
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
def process_image(input_image, 
                  upscale, 
                  supir_sign, 
                  seed,  
                  edm_steps, 
                  restoration_scale, 
                  s_churn, 
                  s_noise, 
                  cfg_scale_end, 
                  control_scale_end,
                  img_caption, 
                  a_prompt, 
                  n_prompt,
                  cfg_scale_start, 
                  control_scale_start,
                  config_path, 
                  loading_half_params, 
                  use_tile_vae, 
                  encoder_tile_size, 
                  decoder_tile_size,
                  ae_dtype, 
                  diff_dtype, 
                  skip_denoise):

    start_time = time.time()

    # Load model with specified precision and performance settings
    global model
    # Reload model if precision settings changed or not loaded yet
    if ('model' not in globals() or
        'current_settings' not in globals() or
        current_settings['config_path'] != config_path or
        current_settings['supir_sign'] != supir_sign or
        current_settings['loading_half_params'] != loading_half_params or
        current_settings['use_tile_vae'] != use_tile_vae or
        current_settings['encoder_tile_size'] != encoder_tile_size or
        current_settings['decoder_tile_size'] != decoder_tile_size or
        current_settings['ae_dtype'] != ae_dtype or
        current_settings['diff_dtype'] != diff_dtype):

        model = load_model(
            config_path=config_path,
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
            'config_path': config_path,
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
    LQ_img, h0, w0 = PIL2Tensor(input_image, upsacle=upscale, min_size=1024)
    LQ_img = LQ_img.unsqueeze(0).to(SUPIR_device)[:, :3, :, :]
    
    # Use the provided caption
    # captions = [img_caption]    

    # Run diffusion process
    samples = model.batchify_sample(LQ_img, img_caption, 
                                    num_steps=edm_steps, 
                                    restoration_scale=restoration_scale, 
                                    s_churn=s_churn,
                                    s_noise=s_noise, 
                                    cfg_scale_start=cfg_scale_start,
                                    cfg_scale_end=cfg_scale_end, 
                                    control_scale_start=control_scale_start,
                                    control_scale_end=control_scale_end, 
                                    seed=seed,
                                    num_samples=1,  # Always 1 for UI 
                                    p_p=a_prompt, 
                                    n_p=n_prompt, 
                                    color_fix_type="Wavelet",
                                    skip_denoise_stage=skip_denoise)
    
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
        save_path = os.path.join(output_dir, f"gradio_{supir_sign}_{next_index}.png")
        result_img.save(save_path)
        print(f"Saved generated image to: {save_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
    # --- End saving logic ---

    end_time = time.time()
    SUPIR_Process_Time = end_time - start_time
    print(f"SUPIR Processing executed in {SUPIR_Process_Time:.2f} seconds.", color.GREEN)

    return result_img

# Function to update tile VAE visibility
def update_tile_vae_visibility(use_tile):
    return gr.update(visible=use_tile)

# Default prompts
default_positive_prompt = 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.'
default_negative_prompt = 'painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth'


# =================================================================================
# GRADIO UI SHIT
# =================================================================================
def create_ui():
    with gr.Blocks(title="SUPIR Image Restoration") as demo:
        # Two-column main layout
        with gr.Row():
            # Left sidebar for all controls
            with gr.Column(scale=1, min_width=380):
                # gr.Markdown("## Controls")
                gr.Markdown("# SUPIR Image Restoration")

                # Image caption (Optional)
                img_caption = gr.Textbox(
                    label="Image Caption", 
                    placeholder="Describe the image to be restored / upscaled",
                    lines=2
                )
                
                with gr.Row():
                        seed = gr.Number(value=1234567891, precision=0, label="Seed")
                        upscale = gr.Dropdown(
                            choices=[1, 2, 3, 4], 
                            value=2, 
                            label="Upscale",
                            interactive=True
                        )                

                        skip_denoise_stage = gr.Checkbox(value=False, label="Skip Denoise Stage", info="Use if input image is already clean and high quality.")
                
                # Basic settings as dropdowns for space efficiency
                # with gr.Group():
                #     with gr.Row():

                        # skip_denoise_stage = gr.Checkbox(value=False, label="Skip Stage 1 Denoise")

                run_button = gr.Button("▶️ Process SUPIR", variant="primary", size="lg")
                        
                # Compact tabbed interface for advanced settings
                with gr.Tabs():
                    # Model Settings Tab
                    with gr.TabItem("Model"):
                        supir_sign = gr.Dropdown(
                            choices=["Q", "F"], 
                            value="Q", 
                            label="Model Type"
                        )                        
                        config_path = gr.Dropdown(
                            choices=[
                                ("Standard (High VRAM)", 'options/SUPIR_v0.yaml'),
                                ("Tiled (Low VRAM)", 'options/SUPIR_v0_tiled.yaml')
                            ],
                            value=('options/SUPIR_v0_tiled.yaml'),
                            label="Sampler"
                        )
                        
                        with gr.Row():
                            loading_half_params = gr.Checkbox(value=True, label="Half Precision")
                            use_tile_vae = gr.Checkbox(value=True, label="Tile VAE")
                        
                        # Compact data type selection
                        with gr.Row():
                            ae_dtype = gr.Dropdown(
                                choices=["fp32", "bf16"], 
                                value="bf16", 
                                label="AE Type"
                            )
                            diff_dtype = gr.Dropdown(
                                choices=["fp32", "fp16", "bf16"], 
                                value="fp16", 
                                label="Diff Type"
                            )
                        
                        # Tile settings in collapsible group
                        with gr.Accordion("Tile Settings", open=False) as tile_vae_settings:
                            encoder_tile_size = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Encoder Tile")
                            decoder_tile_size = gr.Slider(minimum=32, maximum=128, value=64, step=8, label="Decoder Tile")
                    
                    # Diffusion Tab  
                    with gr.TabItem("Diffusion"):
                        # Use compact sliders with labels above
                        edm_steps = gr.Slider(minimum=10, maximum=100, value=50, step=1, label="Sampler Steps")
                        
                        # Group similar controls
                        with gr.Group():
                            gr.Markdown("##### Noise Settings")
                            with gr.Row():
                                s_churn = gr.Slider(minimum=0, maximum=20, value=5, step=1, label="Churn")
                                s_noise = gr.Slider(minimum=1.0, maximum=2.0, value=1.003, step=0.001, label="Noise")
                        
                        with gr.Group():
                            gr.Markdown("##### Guidance")
                            with gr.Row():
                                cfg_scale_start = gr.Slider(minimum=0.0, maximum=10.0, value=2.0, step=0.1, label="CFG Scale Start")        
                                cfg_scale_end = gr.Slider(minimum=1.0, maximum=15.0, value=4.0, step=0.1, label="CFG Scale End")
                            
                            with gr.Row():
                                control_scale_start = gr.Slider(minimum=0.0, maximum=2.0, value=0.9, step=0.1, label="Control Scale Start")
                                control_scale_end = gr.Slider(minimum=0.0, maximum=2.0, value=0.9, step=0.1, label="Control Scale End")
                            
                            restoration_scale = gr.Slider(minimum=-1, maximum=10, value=-1, step=1, label="Restoration Scale(-1=Off)")
                
                    # Prompts Tab
                    with gr.TabItem("Prompts"):
                        a_prompt = gr.Textbox(value=default_positive_prompt, lines=4, label="Additional Positive Prompt (appended to main caption)")
                        n_prompt = gr.Textbox(value=default_negative_prompt, lines=4, label="Negative Prompt")
                
                # # Quick preset buttons
                # with gr.Group():
                #     gr.Markdown("### Quick Presets")
                #     with gr.Row():
                #         preset_detail = gr.Button("Detail Enhancement", size="sm")
                #         preset_restore = gr.Button("Photo Restoration", size="sm") 
                #         preset_upscale = gr.Button("Upscale 4x", size="sm")
            
            # Right main area for images and results
            with gr.Column(scale=2):
                # App title and description
                with gr.Row():
                    pass
                    # gr.Markdown("# SUPIR Image Restoration")
                
                # Before/After with tabs
                with gr.Tabs():
                    with gr.TabItem("Input Image"):
                        input_image = gr.Image(
                            label="Upload an image to enhance", 
                            type="pil", 
                            height=600
                        )
                        gr.Markdown("Upload an image to enhance or restore")
                    
                    with gr.TabItem("Enhanced Result"):
                        output_image = gr.Image(
                            label="Enhanced Image", 
                            height=600
                        )
                        with gr.Row():
                            compare_btn = gr.Button("Compare with Original", size="sm")
                
                # Help accordion at bottom
                with gr.Accordion("Help & Tips", open=False):
                    gr.Markdown("""
                    ## Quick Start
                    1. Upload an image on the Input tab
                    2. Optionally describe what to enhance
                    3. Click ENHANCE button
                    4. View result in the Enhanced Result tab
                    
                    ## Performance Tips
                    - For faster processing: Lower steps (20-30), use fp16 precision
                    - For better quality: Higher steps (50+), use fp32 precision
                    - For large images: Enable Tile VAE in the Model tab
                    """)

        # Connect UI functions
        use_tile_vae.change(
            fn=update_tile_vae_visibility,
            inputs=[use_tile_vae],
            outputs=[tile_vae_settings]
        )
        
        # random_seed_button.click(
        #     fn=generate_random_seed,
        #     inputs=[],
        #     outputs=[seed]
        # )
        
        # # Connect preset buttons
        # def set_detail_preset():
        #     return gr.update(value="Detail enhancement, clear and sharp"), gr.update(value=50), gr.update(value="Q")
        
        # def set_restore_preset():
        #     return gr.update(value="Restore damaged photo, fix artifacts"), gr.update(value=70), gr.update(value="Q")
            
        # def set_upscale_preset():
        #     return gr.update(value="High quality upscale"), gr.update(value=4), gr.update(value="F")
        
        # preset_detail.click(fn=set_detail_preset, inputs=[], outputs=[img_caption, edm_steps, supir_sign])
        # preset_restore.click(fn=set_restore_preset, inputs=[], outputs=[img_caption, edm_steps, supir_sign])
        # preset_upscale.click(fn=set_upscale_preset, inputs=[], outputs=[img_caption, upscale, supir_sign])
        
        # !>>>> Added input validation wrapper
        def validate_and_process(input_image, *args):
            if input_image is None:
                return gr.update(value="Please upload an image first.")
            try:
                return process_image(input_image, *args)
            except Exception as e:
                return gr.update(value=f"Error processing image: {str(e)}")
        
        # !>>>> Connected run button with validation
        run_button.click(
            fn=validate_and_process,
            inputs=[
                input_image, upscale, supir_sign, seed, edm_steps,
                restoration_scale, s_churn, s_noise, cfg_scale_end, control_scale_end,
                img_caption, a_prompt, n_prompt, 
                cfg_scale_start, control_scale_start,
                config_path, loading_half_params, use_tile_vae, encoder_tile_size, decoder_tile_size,
                ae_dtype, diff_dtype, skip_denoise_stage
            ],
            outputs=output_image
        )
    
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
