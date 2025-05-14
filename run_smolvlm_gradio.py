import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForImageTextToText
import gradio as gr
from Y7.colored_print import color, style
import os
import time
from threading import Thread
from transformers.generation.streamers import TextIteratorStreamer
# from huggingface_hub import snapshot_download
from smolvlm.verify_download_model import hash_file_partial, check_model_files, download_model_from_HF

# macOS shit, just in case some pytorch ops are not supported on mps yes, fallback to cpu
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ================================================
# DEFAULT PARAM VALUESS
MAX_NEW_TOKENS = 512
REP_PENALTY = 1.2

# TOP P SAMPLING VALUES
DO_SAMPLING = False
TOP_P = 0.8
TEMP = 0.4

# Define caption style prompts
STYLE_PROMPTS = {
    "Brief and concise": "Caption this image with a short and concise description of the subject.",
    "Moderately detailed": "Caption this image with a moderately detailed description of the subject and environment.",
    "Highly detailed": "Caption this image with a highly detailed and lengthy description of the subject and environment."
}

# Default prompts (the positive is appended to the main caption (whether it exists or not))
default_positive_prompt = 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.'
default_negative_prompt = 'painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth'

# list of just the keys for gradio dropdown
CAPTION_STYLE_OPTIONS = list(STYLE_PROMPTS.keys())

# ====================================================================
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"

# ====================================================================
def load_model(model_path):
    device = get_device()
    print(f"Using {device} device")
    

    # FIRST CHECK IF MODEL EXISTS IN LOCAL DIRECTORY (./models/)
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Attention fallback order 
    attention_fallback = [
        "flash_attention_2",  # Best performance if available
        "sdpa",              # Good default in PyTorch 2.0+
        "xformers",          # Good alternative, memory efficient
        "eager",             # Reliable fallback
        None                 # Absolute fallback
    ]
    
    # Try each attention implementation
    for impl in attention_fallback:
        try:
            if impl is not None:
                model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    _attn_implementation=impl
                ).to(device)
                print(f"✓ Loaded with {impl} attention", color.GREEN)
            else:
                model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16
                ).to(device)
                print("✓ Loaded with no attention specified", color.GREEN)

            return processor, model, device
        
        except ImportError as e:
            if impl == "flash_attention_2" and "flash_attn" in str(e):
                print(f"  flash_attention_2 not available (package not installed)", color.YELLOW)
            else:
                print(f"  Failed with {impl}: {e}", color.RED)
            continue
        except Exception as e:
            print(f"  Failed with {impl}: {e}", color.RED)
            continue
    
    # If we get here, all attempts failed
    raise Exception("Failed to load model with any attention implementation")


# ====================================================================
def generate_caption_streaming(
    image,
    caption_style,
    max_new_tokens=MAX_NEW_TOKENS,
    repetition_penalty=REP_PENALTY,
    do_sample=DO_SAMPLING,
    temperature=TEMP,
    top_p=TOP_P
):
    """Streaming version of caption generation"""
    # Check if image is provided, if not, quit and show msg
    if image is None:
        msg = "Please upload an image first to generate a caption."
        yield msg
        return
    
    start_time = time.time()
    
 
    prompt_text = STYLE_PROMPTS.get(caption_style, "Caption this image.")

    # construct multi-modal input msg
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        },
    ]

    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs_data = processor(text=prompt, images=[image], return_tensors="pt")
    inputs_data = inputs_data.to(DEVICE)

    # Setup streamer
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)

    # Prepare generation arguments
    generation_args = {
        **inputs_data, 
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_args["temperature"] = temperature
        generation_args["top_p"] = top_p

    # Run generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_args)
    thread.start()

    # Yield generated text as it comes in
    generated_text_so_far = ""
    is_first_chunk = True # used for stripping away leading space from first chunk (below)
    
    for new_text_chunk in streamer:
        # Strip leading space from the first chunk only
        if is_first_chunk:
            new_text_chunk = new_text_chunk.lstrip()
            is_first_chunk = False
        
        generated_text_so_far += new_text_chunk
        yield generated_text_so_far


    thread.join()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Optional: print final caption to console for logging
    if generated_text_so_far:
        print(f"Generated caption: '{generated_text_so_far.strip()}'", color.GREEN)
    print(f"Execution_time = {execution_time:.2f} seconds.", color.BRIGHT_BLUE)

# ====================================================================
def generate_caption_non_streaming(
    image,
    caption_style,
    max_new_tokens=MAX_NEW_TOKENS,
    repetition_penalty=REP_PENALTY,
    do_sample=DO_SAMPLING,
    temperature=TEMP,
    top_p=TOP_P
):
    """Non-streaming version of caption generation"""
    # Check if image is provided, if not, quit and show msg
    if image is None:
        msg = "Please upload an image first to generate a caption."
        return msg
    
    start_time = time.time()
        
    prompt_text = STYLE_PROMPTS.get(caption_style, "Caption this image.")

    # construct multi-modal input msg
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        },
    ]

    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(DEVICE)

    # Generate args
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
    }
    # only include temp and top p if do sample
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    generated_ids = model.generate(
        **inputs,
        **generation_kwargs
    )

    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    # Get only the assistant's response
    full_output = generated_texts[0]
    
    if "Assistant:" in full_output:
        response_only = full_output.split("Assistant: ")[-1].strip()
    else:
        response_only = full_output.strip()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Generated caption: '{response_only}'", color.GREEN)
    print(f"Execution_time = {execution_time:.2f} seconds.", color.BRIGHT_BLUE)
    
    return response_only

# ====================================================================
def process_edited_caption(additional_text):
    print(additional_text)

# ====================================================================
# ====================================================================
# GRADIO UI SHIT
# ====================================================================
# ====================================================================
def launch_gradio(use_stream, listen_on_network, port=None):

    # Create custom theme (unchanged from your original code)
    custom_theme = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#faf8fc",
            c100="#f3edf7",
            c200="#e7dbe9",
            c300="#d9c7dc",
            c400="#c9b3ce",
            c500="#7d539a",   # main color
            c600="#7d539a",
            c700="#68447f",
            c800="#533666",
            c900="#3f2850",
            c950="#2a1b36"
        )
    ).set(
        button_primary_background_fill="#7d539a",
        button_primary_background_fill_hover="#68447f",
        button_primary_text_color="white",
        block_label_text_color="#1f2937",
        input_border_color="#e5e7eb",
    )

    model_name = os.path.basename(MODEL_PATH)
    mode = "Streaming" if use_stream else "Non-streaming"

    # Create Gradio interface
    with gr.Blocks(title="Image Captioner", theme=custom_theme,  
                css="""           
                        /* outermost wrapper of the entire Gradio app */         
                        .gradio-container {
                            max-width: 100% !important;
                            margin: 0 auto !important;
                        }
                        /* main content area within the Gradio container */
                        .main {
                            max-width: 1200px !important;
                            margin: 0 auto !important;
                        }
                        /* Individual columns */
                        .fixed-width-column {
                            width: 600px !important;
                            flex: none !important;
                        }
                        /* Custom color for the editable text box */
                        #text_box textarea {
                            /*  color: #2563eb !important;  text color */
                            font-family: 'monospace', monospace !important; 
                            font-size: 12px !important; 
                        }           
                        /* Make dropdown menus taller */
                        .gradio-dropdown .choices {
                            max-height: 300px !important;  /* Adjust this value as needed */
                            overflow-y: auto;
                        }       
                        /* Force scrollbar to always be present */
                        html {
                            overflow-y: scroll !important;
                        }
                        
                        /* For modern browsers that support it, we can make the scrollbar not take up space */
                        @supports (scrollbar-gutter: stable) {
                            html {
                                scrollbar-gutter: stable !important;
                            }
                        }
                        
                        /* Alternative approach for browsers that don't support scrollbar-gutter */
                        body {
                            padding-right: calc(100vw - 100%) !important; /* This adds padding equal to the scrollbar width */
                        }                                                                                                      
                    """) as demo:   
        
        gr.Markdown("# Image Captioner : SmolVLM-Instruct")    
        gr.Markdown(f"**Model**: {model_name} | **Mode**: {mode}")        
        
        # Create tabs
        with gr.Tabs() as tabs:
            # ==============================================================================================
            # TAB 1 - INPUT IMAGE + SMOLVLM
            # ==============================================================================================
            with gr.TabItem("Input Image / Caption Generator"):
                gr.Markdown("Upload an image and generate a caption (optional)")
                
                with gr.Row():
                    # ================================================
                    # COL 1
                    with gr.Column(elem_classes=["fixed-width-column"]):
                        input_image = gr.Image(type="pil", label="Input Image", height=640)
                                                
                        submit_btn = gr.Button("Generate Caption", variant="primary")
                        
                    # ================================================
                    # COL 2                    
                    with gr.Column(elem_classes=["fixed-width-column"]):
                        with gr.Accordion("Settings", open=True):
                            with gr.Row():
                                caption_style = gr.Dropdown(
                                    choices=CAPTION_STYLE_OPTIONS,
                                    value=CAPTION_STYLE_OPTIONS[1] if len(CAPTION_STYLE_OPTIONS) > 1 else CAPTION_STYLE_OPTIONS[0] if CAPTION_STYLE_OPTIONS else "Moderately detailed",
                                    label="Caption Style"
                                )
                            gr.Markdown("Sampler Settings") 
                            with gr.Row():
                                max_tokens = gr.Slider(minimum=50, maximum=1024, value=MAX_NEW_TOKENS, step=1, label="Max New Tokens")
                                rep_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=REP_PENALTY, step=0.1, label="Repetition Penalty")

                            gr.Markdown("""    
                                        `Max New Tokens`: Controls the maximum length of the generated caption   
                                        `Repetition Penalty`: Higher values discourage repetition in the text                                 
                                        """)
                            
                            # Group the sampling-related controls together
                            with gr.Group():
                                do_sample_checkbox = gr.Checkbox(value=DO_SAMPLING, label="Do Sample")
                                with gr.Row():
                                    temperature_slider = gr.Slider(minimum=0.1, maximum=1.0, value=TEMP, step=0.1, label="Temperature")
                                    top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=TOP_P, step=0.1, label="Top P")

                            gr.Markdown("""    
                                        `Do Sample`: Enabled: uses Top P sampling for more diverse outputs. Disabled: use greedy mode (deterministic)  
                                        `Temperature`: Higher values (>1.0) = output more random, lower values = more deterministic  
                                        `Top P`: Higher values (0.8-0.95): More variability, more diverse outputs, Lower values (0.1-0.5): Less variability, more consistent outputs  
                                        """)
                
                with gr.Row():
                    with gr.Column():
                        output_text = gr.Textbox(label="Generated Caption", lines=5, interactive=True, elem_id="text_box", info="you can edit the caption here before proceeding")
                
                        # Add the Process button under the second column
                        # process_btn = gr.Button("Continue", variant="primary")
            
            # ==============================================================================================
            # TAB 2 - SUPIR
            # ==============================================================================================
            with gr.TabItem("SUPIR"):
                # gr.Markdown("Restore/Enhance/Upscale")
                
                # -------------------------------------------------
                # ROW - OUTER
                # -------------------------------------------------
                with gr.Row():
                                        
                    # -------------------------------------------------
                    # COL 1
                    # -------------------------------------------------
                    # Left column content
                    with gr.Column(elem_classes=["fixed-width-column"]):

                        with gr.Row():
                            # supir_sign renamed to supir_model
                            supir_model = gr.Dropdown(
                                choices=["Q", "F"], 
                                value="Q", 
                                label="Model Type"
                            )  

                            # sampler type[RestoreEDMSampler, TiledRestoreEDMSampler] 
                            # internally returns the config_path to the correct config (yaml)
                            sampler_config_path = gr.Dropdown(
                                choices=[
                                    ("RestoreEDMSampler (Higher VRAM)", 'options/SUPIR_v0.yaml'),
                                    ("TiledRestoreEDMSampler (Lower VRAM)", 'options/SUPIR_v0_tiled.yaml')
                                ],
                                value=('options/SUPIR_v0_tiled.yaml'),
                                label="Sampler Type"
                            )

                        with gr.Row():
                                seed = gr.Number(value=1234567891, precision=0, label="Seed", interactive=True)
                                upscale = gr.Dropdown(choices=[1, 2, 3, 4], value=2, label="Upscale", interactive=True)      
                                skip_denoise_stage = gr.Checkbox(value=False, label="Skip Denoise Stage", info="Use if input image is already clean and high quality.")

                        with gr.Group():
                            with gr.Row():
                                loading_half_params = gr.Checkbox(value=True, label="Load Model in Half Precision (fp16)")
                            with gr.Row():
                                ae_dtype = gr.Dropdown(
                                    choices=["fp32", "bf16"], 
                                    value="bf16", 
                                    label="AE dType"
                                )
                                diff_dtype = gr.Dropdown(
                                    choices=["fp32", "fp16", "bf16"], 
                                    value="fp16", 
                                    label="Diffusion dType"
                                )

                        # Tile settings in collapsible group
                        with gr.Group() as tile_vae_settings:
                            with gr.Row():
                                use_tile_vae = gr.Checkbox(value=True, label="Use Tile VAE")
                                # The AE processes the input image in tiles of specified size instead of the full image at once
                                encoder_tile_size = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Encoder Tile Size")
                                # The AE reconstructs the final image by stitching together outputs from smaller tile segments
                                decoder_tile_size = gr.Slider(minimum=32, maximum=128, value=64, step=8, label="Decoder Tile Size")
                                                    
                        # sample_image = gr.Image(type="pil", label="Sample Image Input", height=400)
                        
                        # sample_dropdown = gr.Dropdown(
                        #     choices=["Option 1", "Option 2", "Option 3"],
                        #     value="Option 1",
                        #     label="Sample Dropdown"
                        # )
                        
                        # sample_checkbox = gr.Checkbox(value=False, label="Sample Checkbox")
                        
                        
                        
                    # -------------------------------------------------
                    # COL 2                    
                    # -------------------------------------------------
                    with gr.Column(elem_classes=["fixed-width-column"]):
                                            
                        with gr.Group():
                            # gr.Markdown("  Noise Settings")
                            with gr.Row():
                                edm_steps = gr.Slider(minimum=10, maximum=100, value=50, step=1, label="Steps") # sampler steps
                                s_churn = gr.Slider(minimum=0, maximum=20, value=5, step=1, label="S-Churn") # stochastic churn
                                s_noise = gr.Slider(minimum=1.0, maximum=2.0, value=1.003, step=0.001, label="S-Noise") # stochastic noise                        
                                
                        with gr.Group():
                            gr.Markdown("CFG Guidance")
                            with gr.Row():
                                cfg_scale_start = gr.Slider(minimum=0.0, maximum=10.0, value=2.0, step=0.1, label="CFG Scale Start")        
                                cfg_scale_end = gr.Slider(minimum=1.0, maximum=15.0, value=4.0, step=0.1, label="CFG Scale End")
                            
                        with gr.Group():
                            gr.Markdown("Control Guidance")                            
                            with gr.Row():
                                control_scale_start = gr.Slider(minimum=0.0, maximum=2.0, value=0.9, step=0.1, label="Control Scale Start")
                                control_scale_end = gr.Slider(minimum=0.0, maximum=2.0, value=0.9, step=0.1, label="Control Scale End")

                            restoration_scale = gr.Slider(minimum=-1, maximum=10, value=-1, step=1, label="Restoration Scale(-1=Off)")
                                
                # additional prompt and standard neg. prompt
                with gr.Accordion("Prompts", open=False):
                    with gr.Row():
                        a_prompt = gr.Textbox(value=default_positive_prompt, lines=4, label="Additional Positive Prompt (appended to main caption)")
                        n_prompt = gr.Textbox(value=default_negative_prompt, lines=4, label="Negative Prompt")

                with gr.Row():
                    process_supir_btn = gr.Button("Process", variant="primary")

                with gr.Row():
                    output_image = gr.Image(
                        label="Enhanced Image", 
                        height=300
                    )
                    # with gr.Row():
                        # compare_btn = gr.Button("Compare with Original", size="sm")                    
                        # result_textbox = gr.Textbox(label="Results", lines=5, placeholder="Results will appear here...", interactive=True)
                    
                        # Add another button
                        # export_btn = gr.Button("Export Results", variant="secondary")

        # Choose the appropriate generate function based on the argument 'use_stream'
        # and assign to function reference 'generate_function'  
        if use_stream:
            generate_function = generate_caption_streaming 
        else:
            generate_function = generate_caption_non_streaming

        # ==============================================================================================
        # Tab 1 Event Handler(s)
        # ==============================================================================================        
        submit_btn.click(
            fn=generate_function,
            inputs=[
                input_image,
                caption_style,
                max_tokens,
                rep_penalty,
                do_sample_checkbox,
                temperature_slider,
                top_p_slider
            ],
            outputs=[output_text]
        )

        # process_btn.click(
        #     fn=process_edited_caption,
        #     inputs=[output_text]
        # )
        
        # ==============================================================================================
        # Tab 2 event handlers
        # ==============================================================================================
        
        def process_supir(input_image, super_model, sampler_config_path,
                        seed, upscale, skip_denoise_stage,
                        loading_half_params, ae_dtype, diff_dtype,
                        use_tile_vae, encoder_tile_size, decoder_tile_size,
                        edm_steps, s_churn, s_noise,
                        cfg_scale_start, cfg_scale_end,
                        control_scale_start, control_scale_end, restoration_scale,
                        a_prompt, n_prompt
                        ):
            
            
            print("\n")
            print(f"input_image: {input_image}", color.YELLOW)
            print(f"super_model: {super_model}", color.YELLOW)
            print(f"sampler_config_path: {sampler_config_path}", color.YELLOW)
            print(f"seed: {seed}", color.YELLOW)
            print(f"upscale: {upscale}", color.YELLOW)
            print(f"skip_denoise_stage: {skip_denoise_stage}", color.YELLOW)
            print(f"loading_half_params: {loading_half_params}", color.YELLOW)
            print(f"ae_dtype: {ae_dtype}", color.YELLOW)
            print(f"diff_dtype: {diff_dtype}", color.YELLOW)
            print(f"use_tile_vae: {use_tile_vae}", color.YELLOW)
            print(f"encoder_tile_size: {encoder_tile_size}", color.YELLOW)
            print(f"decoder_tile_size: {decoder_tile_size}", color.YELLOW)
            print(f"edm_steps: {edm_steps}", color.YELLOW)
            print(f"s_churn: {s_churn}", color.YELLOW)
            print(f"s_noise: {s_noise}", color.YELLOW)
            print(f"cfg_scale_start: {cfg_scale_start}", color.YELLOW)
            print(f"cfg_scale_end: {cfg_scale_end}", color.YELLOW)
            print(f"control_scale_start: {control_scale_start}", color.YELLOW)
            print(f"control_scale_end: {control_scale_end}", color.YELLOW)
            print(f"restoration_scale: {restoration_scale}", color.YELLOW)
            print(f"a_prompt: {a_prompt}", color.YELLOW)
            print(f"n_prompt: {n_prompt}", color.YELLOW)


        process_supir_btn.click(
            fn=process_supir,
            inputs=[
                input_image,
                supir_model,
                sampler_config_path,
                seed,
                upscale,
                skip_denoise_stage,
                loading_half_params,  
                ae_dtype,            
                diff_dtype,          
                use_tile_vae, 
                encoder_tile_size, 
                decoder_tile_size,
                edm_steps,
                s_churn,
                s_noise,
                cfg_scale_start,
                cfg_scale_end,
                control_scale_start, 
                control_scale_end, 
                restoration_scale,
                a_prompt, 
                n_prompt
            ],
            outputs=output_image
        )
        
        def export_function(text):
            return "Export functionality would save: " + text
            


        server_name = "0.0.0.0" if listen_on_network else None
        demo.launch(server_name=server_name, server_port=port)
        

def main():
    global processor, model, DEVICE, MODEL_PATH

    # Parse CLI arguments (can be passed manually as `argv` for testing)
    parser = argparse.ArgumentParser(description="Run SmolVLM with Gradio")
    parser.add_argument("--use_stream", action="store_true", help="Use streaming mode for text generation")
    parser.add_argument("--listen", action="store_true", help="Launch Gradio with server_name='0.0.0.0' to listen on all interfaces")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio on (default: 7860)")
    parser.add_argument("--model", 
                        choices=["SmolVLM-Instruct", "SmolVLM-500M-Instruct", "SmolVLM-256M-Instruct"],
                        default="SmolVLM-Instruct", 
                        help="Select model (default: SmolVLM-Instruct)")
    args = parser.parse_args()

    # Set model path
    MODEL_PATH = f"models/{args.model}"
    
    # Set mode for UI display
    global UI_MODE
    UI_MODE = "Streaming" if args.use_stream else "Non-streaming"

    # Load/check model
    start_time = time.time()

    filesokay = check_model_files(MODEL_PATH)
    if not filesokay:
        download_model_from_HF(MODEL_PATH)

    processor, model, DEVICE = load_model(MODEL_PATH)

    end_time = time.time()
    model_load_time = end_time - start_time
    print(f"Model {os.path.basename(MODEL_PATH)} loaded on {DEVICE} in {model_load_time:.2f} seconds.", color.GREEN)

    # Attach to Gradio (if needed)
    launch_gradio(args.use_stream, args.listen, args.port)

# Launch the Gradio app
if __name__ == "__main__":
    main()
