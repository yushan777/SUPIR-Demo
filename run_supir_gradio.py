import argparse
import torch
from PIL import Image, PngImagePlugin
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForImageTextToText
import gradio as gr
from Y7.colored_print import color, style
import os
import sys
import time
import glob
from threading import Thread
from transformers.generation.streamers import TextIteratorStreamer
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
import gc
import ctypes
import platform
import json

# from huggingface_hub import snapshot_download
from Y7.verify_model import check_smolvlm_model_files, check_supir_model_files, check_clip_model_file, check_for_any_sdxl_model

# macOS shit, just in case some pytorch ops are not supported on mps yes, fallback to cpu
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ================================================
# DEFAULT PARAM VALUESS
MAX_NEW_TOKENS = 256
REP_PENALTY = 1.2

# TOP P SAMPLING VALUES
DO_SAMPLING = True
TOP_P = 0.8
TEMP = 0.4

# Define caption style prompts
STYLE_PROMPTS = {
    "Brief and concise": "Generate a short and concise caption of this image, suitable for use as an image-generation prompt. Describe the subject, environment, lighting, mood, and style",
    "Moderately detailed": "Generate a moderately detailed and descriptive caption of this image, suitable for use as an image-generation prompt. Describe the subject, environment, lighting, mood, and style",
    "Highly detailed": "Generate a highly detailed and descriptive caption of this image, suitable for use as an image-generation prompt. Describe the subject, environment, lighting, mood, and style."
}

# path to smolvlm model (global)
SMOLVLM_MODEL_PATH = None

#  ===================================================================

# GLOBAL VARIABLES SO SUPIR DOESN'T NEED TO BE RE-LOADED EACH TIME
SUPIR_MODEL = None
SUPIR_SETTINGS = {}

# SUPIR SAMPLER CONFIG PATHS
RestoreEDMSampler_config = 'options/SUPIR_v0.yaml'
TiledRestoreEDMSampler_config = 'options/SUPIR_v0_tiled.yaml'

# Constants for SUPIR_SETTINGS keys
# critical to model initialization and used for caching the model to avoid unecessary reloads
SUPIR_SAMPLER_TYPE = 'sampler_config_path'
SUPIR_MODEL_TYPE = 'supir_model_type'
SUPIR_HALF_PARAMS = 'loading_half_params'
SUPIR_TILE_VAE = 'use_tile_vae'
SUPIR_ENCODER_TILE_SIZE = 'encoder_tile_size'
SUPIR_DECODER_TILE_SIZE = 'decoder_tile_size'
SUPIR_TILE_VAE_WORKERS = 'tile_vae_num_of_workers'
SUPIR_AE_DTYPE = 'ae_dtype'
SUPIR_DIFF_DTYPE = 'diff_dtype'
SUPIR_DIFF_SAMPLER_TILE_SIZE = 'sampler_tile_size'
SUPIR_DIFF_SAMPLER_TILE_STRIDE = 'sampler_tile_stride'

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
# LOAD SMOVLM MODEL
def load_smolvlm_model(model_path):
    device = get_device()
    print(f"Using {device} device")
    
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
                print(f"  flash_attention_2 not available (package not installed)")
            else:
                print(f"  Failed with {impl}: {e}", color.RED)
            continue
        except Exception as e:
            print(f"  Failed with {impl}: {e}", color.RED)
            continue
    
    # If we get here, all attempts failed
    raise Exception("Failed to load model with any attention implementation")


# Create SUPIR model with specified settings
def load_supir_model(sampler_type, 
               supir_model_type='Q', 
               loading_half_params=False, 
               ae_dtype="bf16", 
               diff_dtype="fp16",
               use_tile_vae=False, 
               encoder_tile_size=512, 
               decoder_tile_size=64,
               num_workers=1,
               sampler_tile_size=128,
               sampler_tile_stride=64):
    
    device = get_device()

    

    if sampler_type == "RestoreEDMSampler":
        sampler_config_path = RestoreEDMSampler_config
    elif sampler_type == "TiledRestoreEDMSampler":
        sampler_config_path = TiledRestoreEDMSampler_config

    print(f"Loading SUPIR model from config: {sampler_config_path}")

    model = create_SUPIR_model(sampler_config_path, SUPIR_sign=supir_model_type)
    if loading_half_params:
        model = model.half()
    if use_tile_vae:
        model.init_tile_vae(encoder_tile_size=encoder_tile_size, decoder_tile_size=decoder_tile_size, num_parallel_workers=num_workers)

    # Set the precision for the VAE component
    model.ae_dtype = convert_dtype(ae_dtype)
    # Set the precision for the diffusion component (unet)
    model.model.dtype = convert_dtype(diff_dtype)
    model = model.to(device)

    # if using TiledRestoreEDMSampler - set sampler tile size and stride   
    if sampler_type == "TiledRestoreEDMSampler":
        # set/override tile size and tile stride
        model.sampler.tile_size = sampler_tile_size
        model.sampler.tile_stride = sampler_tile_stride

    return model


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
    
    # load the smolvlm model. 
    processor, model, DEVICE = load_smolvlm_model(SMOLVLM_MODEL_PATH)
    print(f"Model {os.path.basename(SMOLVLM_MODEL_PATH)} loaded on {DEVICE}", color.GREEN)

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

# process SUPIR on the image
def process_supir(
            input_image,
            image_caption, 
            supir_model_type,  
            sampler_type,
            seed, 
            upscale_by,
            use_upscale_to,
            upscale_to_width,
            upscale_to_height,
            skip_denoise_stage,
            loading_half_params, 
            ae_dtype, 
            diff_dtype,
            use_tile_vae, 
            encoder_tile_size, 
            decoder_tile_size,
            num_of_workers,
            edm_steps, 
            s_churn, 
            s_noise,
            cfg_scale_start, 
            cfg_scale_end,
            control_scale_start, 
            control_scale_end, 
            restoration_scale,
            sampler_tile_size,
            sampler_tile_stride,            
            a_prompt, 
            n_prompt
        ):
    
    # Check if input_image is provided, if not, quit and show msg
    if input_image is None:
        # Return a tuple with None for the image and an error message
        return None, "Please upload an image first in Tab 1."
        
    # ============================================================
    # Clear GPU memory (VRAM) if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("✅ CUDA cache cleared")

    # Trigger Python's garbage collector
    collected = gc.collect()
    print(f"✅ Python garbage collection complete ({collected} objects collected)")

    # Try to release malloc'd memory to the OS (Linux only)
    if platform.system() == "Linux":
        try:
            libc = ctypes.CDLL("libc.so.6")
            result = libc.malloc_trim(0)
            if result == 1:
                print("✅ Unused system memory returned to OS (via malloc_trim)")
            else:
                print("ℹ️ malloc_trim executed, but nothing was returned to the OS")
        except Exception as e:
            print(f"❌ malloc_trim failed: {e}")
    else:
        print("ℹ️ malloc_trim skipped (not supported on this OS)")
    #  ============================================================
             
    print("SUPIR Settings\n")
    print(f"input_image: {input_image}", color.YELLOW)
    print(f"image_caption: {image_caption}", color.YELLOW)
    print(f"supir_model_type: {supir_model_type}", color.YELLOW)
    print(f"sampler_type: {sampler_type}", color.YELLOW)
    print(f"seed: {seed}", color.YELLOW)
    print(f"upscale: {upscale_by}", color.YELLOW)
    print(f"skip_denoise_stage: {skip_denoise_stage}", color.YELLOW)
    print(f"loading_half_params: {loading_half_params}", color.YELLOW)
    print(f"ae_dtype: {ae_dtype}", color.YELLOW)
    print(f"diff_dtype: {diff_dtype}", color.YELLOW)
    print(f"use_tile_vae: {use_tile_vae}", color.YELLOW)
    print(f"encoder_tile_size: {encoder_tile_size}", color.YELLOW)
    print(f"decoder_tile_size: {decoder_tile_size}", color.YELLOW)
    print(f"num_of_workers: {num_of_workers}", color.YELLOW)
    print(f"edm_steps: {edm_steps}", color.YELLOW)
    print(f"s_churn: {s_churn}", color.YELLOW)
    print(f"s_noise: {s_noise}", color.YELLOW)
    print(f"cfg_scale_start: {cfg_scale_start}", color.YELLOW)
    print(f"cfg_scale_end: {cfg_scale_end}", color.YELLOW)
    print(f"control_scale_start: {control_scale_start}", color.YELLOW)
    print(f"control_scale_end: {control_scale_end}", color.YELLOW)
    print(f"restoration_scale: {restoration_scale}", color.YELLOW)
    print(f"sampler_tile_size: {sampler_tile_size}", color.YELLOW)
    print(f"sampler_tile_stride: {sampler_tile_stride}", color.YELLOW)
    print(f"a_prompt: {a_prompt}", color.YELLOW)
    print(f"n_prompt: {n_prompt}", color.YELLOW)


    start_time = time.time()

    # Use the global SUPIR_MODEL and SUPIR_SETTINGS variables - declared at top level.
    global SUPIR_MODEL, SUPIR_SETTINGS

    # ONLY Reload SUPIR model if settings critical to its initialization are changed or not loaded yet
    # print(f"SUPIR_MODEL is already 'loaded' if {SUPIR_MODEL is not None else 'not initialized'}", color.MAGENTA)
    if SUPIR_MODEL:
        status = "already loaded."
    else:
        status = "loading."

    print(f"SUPIR_MODEL is {status}", color.MAGENTA)


    if (SUPIR_MODEL is None or
        not SUPIR_SETTINGS or  # Check if the dictionary is empty
        SUPIR_SETTINGS.get(SUPIR_SAMPLER_TYPE) != sampler_type or
        SUPIR_SETTINGS.get(SUPIR_MODEL_TYPE) != supir_model_type or
        SUPIR_SETTINGS.get(SUPIR_HALF_PARAMS) != loading_half_params or
        SUPIR_SETTINGS.get(SUPIR_TILE_VAE) != use_tile_vae or
        SUPIR_SETTINGS.get(SUPIR_ENCODER_TILE_SIZE) != encoder_tile_size or
        SUPIR_SETTINGS.get(SUPIR_DECODER_TILE_SIZE) != decoder_tile_size or
        SUPIR_SETTINGS.get(SUPIR_TILE_VAE_WORKERS) != num_of_workers or
        SUPIR_SETTINGS.get(SUPIR_AE_DTYPE) != ae_dtype or
        SUPIR_SETTINGS.get(SUPIR_DIFF_DTYPE) != diff_dtype or
        SUPIR_SETTINGS.get(SUPIR_DIFF_SAMPLER_TILE_SIZE) != sampler_tile_size or
        SUPIR_SETTINGS.get(SUPIR_DIFF_SAMPLER_TILE_STRIDE) != sampler_tile_stride):

        SUPIR_MODEL = load_supir_model(
            sampler_type=sampler_type,
            supir_model_type=supir_model_type,
            loading_half_params=loading_half_params,
            use_tile_vae=use_tile_vae,
            encoder_tile_size=encoder_tile_size,
            decoder_tile_size=decoder_tile_size,
            num_workers=num_of_workers,
            ae_dtype=ae_dtype,
            diff_dtype=diff_dtype,
            sampler_tile_size=sampler_tile_size,
            sampler_tile_stride=sampler_tile_stride 
        )

        # Store current settings for future comparison
        SUPIR_SETTINGS = {
            SUPIR_SAMPLER_TYPE: sampler_type,
            SUPIR_MODEL_TYPE: supir_model_type,
            SUPIR_HALF_PARAMS: loading_half_params,
            SUPIR_TILE_VAE: use_tile_vae,
            SUPIR_ENCODER_TILE_SIZE: encoder_tile_size,
            SUPIR_DECODER_TILE_SIZE: decoder_tile_size,
            SUPIR_TILE_VAE_WORKERS: num_of_workers,
            SUPIR_AE_DTYPE: ae_dtype,
            SUPIR_DIFF_DTYPE: diff_dtype,
            SUPIR_DIFF_SAMPLER_TILE_SIZE: sampler_tile_size,
            SUPIR_DIFF_SAMPLER_TILE_STRIDE: sampler_tile_stride
        }

    # Convert to PIL if needed
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)
    

    device = get_device()

    # Process input image by upscaling it to the min_size
    # min_size of 1024 is to make it work nicely with sdxl
    # after sampling if the input image was less than 1024 and upscale was 1, it will be resized back to its original size

    print(f"use_upscale_to = {use_upscale_to}", color.ORANGE)

    if use_upscale_to:
        print(f"using upscale to W/H", color.ORANGE)
        resized_image = input_image.resize((upscale_to_width, upscale_to_height), Image.BICUBIC)
        print(f"resized_image = {resized_image.size}", color.ORANGE)

        LQ_img, h0, w0 = PIL2Tensor(resized_image, upscale=1, min_size=1024)
    else:
        print(f"using upscale by", color.ORANGE)
        LQ_img, h0, w0 = PIL2Tensor(input_image, upscale=upscale_by, min_size=1024)

    print(f"LQ_img shape: {LQ_img.shape}", color.MAGENTA)

    LQ_img = LQ_img.unsqueeze(0).to(device)[:, :3, :, :]

    # print(f"h,w = {h0}, {w0}", color.ORANGE)
    
    # Run diffusion process
    samples = SUPIR_MODEL.batchify_sample(LQ_img, image_caption, 
                                    num_steps=edm_steps, 
                                    restoration_scale=restoration_scale, 
                                    s_churn=s_churn,
                                    s_noise=s_noise, 
                                    cfg_scale_start=cfg_scale_start,
                                    cfg_scale_end=cfg_scale_end, 
                                    control_scale_start=control_scale_start,
                                    control_scale_end=control_scale_end, 
                                    seed=seed,
                                    num_parallel_workers=num_of_workers,
                                    num_samples=1,  # Always 1 for UI 
                                    p_p=a_prompt, 
                                    n_p=n_prompt, 
                                    color_fix_type="Wavelet",
                                    skip_denoise_stage=skip_denoise_stage)
    
    # settings string for embedding into png
    supir_settings = "\n"
    supir_settings += f"input_image_resolution: {input_image.width}x{input_image.height}\n"
    supir_settings += f"image_caption: {image_caption}\n"
    supir_settings += f"supir_model_type: {supir_model_type}\n"
    supir_settings += f"sampler_type: {sampler_type}\n"
    supir_settings += f"seed: {seed}\n"
    supir_settings += f"upscale: {upscale_by}\n"
    supir_settings += f"skip_denoise_stage: {skip_denoise_stage}\n"
    supir_settings += f"loading_half_params: {loading_half_params}\n"
    supir_settings += f"ae_dtype: {ae_dtype}\n"
    supir_settings += f"diff_dtype: {diff_dtype}\n"
    supir_settings += f"use_tile_vae: {use_tile_vae}\n"
    supir_settings += f"encoder_tile_size: {encoder_tile_size}\n"
    supir_settings += f"decoder_tile_size: {decoder_tile_size}\n"
    supir_settings += f"num_of_workers: {num_of_workers}\n"    
    supir_settings += f"edm_steps: {edm_steps}\n"
    supir_settings += f"s_churn: {s_churn}\n"
    supir_settings += f"s_noise: {s_noise}\n"
    supir_settings += f"cfg_scale_start: {cfg_scale_start}\n"
    supir_settings += f"cfg_scale_end: {cfg_scale_end}\n"
    supir_settings += f"control_scale_start: {control_scale_start}\n"
    supir_settings += f"control_scale_end: {control_scale_end}\n"
    supir_settings += f"restoration_scale: {restoration_scale}\n"
    supir_settings += f"sampler_tile_size: {sampler_tile_size}\n"
    supir_settings += f"sampler_tile_stride: {sampler_tile_stride}\n"
    supir_settings += f"a_prompt: {a_prompt}\n"
    supir_settings += f"n_prompt: {n_prompt}"

    # Convert result to PIL image
    enhanced_image = Tensor2PIL(samples[0], h0, w0)

    # --- Save the image before returning ---
    try:
        # Ensure the output directory exists
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # Find existing gradio files that match the current model type
        base_filename = f"gradio_{supir_model_type}"
        existing_files = glob.glob(os.path.join(output_dir, f"{base_filename}_*.png"))
        
        # Find the highest index used
        max_index = -1
        for f in existing_files:
            try:
                # Extract index from filename like "gradio_modeltype_0123.png"
                filename = os.path.basename(f)
                index_str = filename.replace(f"{base_filename}_", "").replace(".png", "")
                index = int(index_str)
                if index > max_index:
                    max_index = index
            except ValueError:
                # Ignore files that don't match the expected pattern
                pass
        
        # Use the next available index, zero-padded to 4 digits
        next_index = max_index + 1
        padded_index = f"{next_index:04d}"  # This formats the number with leading zeros to 4 digits
        
        # Construct the full save path with the model type and padded index
        png_save_path = os.path.join(output_dir, f"{base_filename}_{padded_index}.png")

        # embed the supir settings into tEXt chunks 
        # Embed the contents of the files into 
        info = PngImagePlugin.PngInfo()
        info.add_text("SUPIR:", supir_settings)

        # Save the modified PNG with the tEXt chunks embedded
        enhanced_image.save(png_save_path, "PNG", pnginfo=info)

        print(f"Saved SUPIR'd image to: {png_save_path}")




    except Exception as e:
        print(f"Error saving image: {e}")
    # --- End saving logic ---

    end_time = time.time()
    SUPIR_Process_Time = end_time - start_time
    print(f"SUPIR Processing executed in {SUPIR_Process_Time:.2f} seconds.", color.GREEN)

    # prep for final comparison.  we need to make both images same size
    # Get the dimensions of both images
    img1_width, img1_height = input_image.size
    img2_width, img2_height = enhanced_image.size

    # Check if images have different dimensions
    if img1_width != img2_width or img1_height != img2_height:
        # Resize the first image to match the dimensions of the second image
        # Using BICUBIC resampling for better quality when upscaling
        input_image = input_image.resize((img2_width, img2_height), Image.BICUBIC)
        print(f"Resized first image from {img1_width}x{img1_height} to {img2_width}x{img2_height}")

    # returning the path to the png instead of 'enhanced_image' save image ensures that the download button will download that file instead of a webp
    # return [input_image, enhanced_image], "Processing complete! See results on Tab 3."
    
    return [input_image, png_save_path], "Processing complete! See results on Tab 3."


    
# ====================================================================
def process_edited_caption(additional_text):
    print(additional_text)


# for when image is added/removed in the gradio UI
def get_image_dimensions(img: Image.Image):
    if img is None:
        return ""
    width, height = img.size
    return f"Dimensions: {width} × {height}"

# ====================================================================
# ====================================================================
# GRADIO UI SHIT
# ====================================================================
# ====================================================================

# Helper function to load defaults from JSON
def load_app_defaults():
    defaults_path = "defaults.json"
    try:
        with open(defaults_path, 'r') as f:
            data = json.load(f)
            # Ensure top-level keys exist, even if empty, to simplify downstream .get() calls
            if 'smolvlm_settings' not in data:
                data['smolvlm_settings'] = {}
            if 'supir_settings' not in data:
                data['supir_settings'] = {}
            print(f"Successfully loaded defaults from {defaults_path}")
            return data
    except FileNotFoundError:
        print(f"Warning: {defaults_path} not found. Using hardcoded defaults.", color.ORANGE)
        return {"smolvlm_settings": {}, "supir_settings": {}}
    except json.JSONDecodeError:
        print(f"Error: Could not decode {defaults_path}. Using hardcoded defaults.", color.ORANGE)
        return {"smolvlm_settings": {}, "supir_settings": {}}
    except Exception as e:
        print(f"An unexpected error occurred while loading {defaults_path}: {e}. Using hardcoded defaults.", color.ORANGE)
        return {"smolvlm_settings": {}, "supir_settings": {}}

def create_launch_gradio(listen_on_network, port=None):
    # Load defaults from JSON file
    app_defaults = load_app_defaults()
    smolvlm_defaults = app_defaults.get('smolvlm_settings', {})
    supir_defaults = app_defaults.get('supir_settings', {})

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

    # model_name = os.path.basename(SMOLVLM_MODEL_PATH)
    # mode = "Streaming"

    # Create Gradio interface
    with gr.Blocks(title="SUPIR Detailer/Enhancer/Upscaler", theme=custom_theme,  
                   
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
                        .fixed-width-column-600 {
                            width: 600px !important;
                            flex: none !important;
                        }
                        .fixed-width-column-1216 {
                            width: 1216px !important;
                            flex: none !important;
                        }                        
                        .taller-row1 {
                            min-height: 120px !important; /* Adjust as needed */
                            align-items: center; /* Optional: vertically center */
                        }                        
                        /* Custom color for the editable text box */
                        #text_box textarea {
                            /*  color: #2563eb !important;  text color */
                            font-family: 'monospace', monospace !important; 
                            font-size: 11px !important; 
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
                        .gradio-imageslider img {
                            object-fit: contain !important;
                            max-width: 100% !important;
                            max-height: 100% !important;
                            width: auto !important;
                            height: auto !important;
}                        
                        /* Alternative approach for browsers that don't support scrollbar-gutter */
                        body {
                            padding-right: calc(100vw - 100%) !important; /* This adds padding equal to the scrollbar width */
                        }       
                                                                                                                       
                    """) as demo:   
        
        gr.Markdown("### SUPIR Restorer / Detailer / Upscaler")    
        # gr.Markdown(f"**Model**: {model_name} | **Mode**: {mode}")        
        

        # Create tabs
        with gr.Tabs() as tabs:
            # ==============================================================================================
            # TAB 1 - INPUT IMAGE + SMOLVLM
            # ==============================================================================================
            with gr.TabItem("1. Input Image / SmolVLM Captioner"):
                gr.Markdown("Upload image and generate a caption or write your own (optional).")
                
                with gr.Row():
                    # ================================================
                    # COL 1
                    with gr.Column(elem_classes=["fixed-width-column-600"]):
                        input_image = gr.Image(type="pil", label="Input Image", height=480)                        

                        submit_btn = gr.Button("Generate Caption", variant="primary")
                        
                    # ================================================
                    # COL 2                    
                    with gr.Column(elem_classes=["fixed-width-column-600"]):
                        # image dimensions info text box
                        image_dims = gr.Textbox(label="Input Image Dimensions", interactive=False)

                        with gr.Accordion("SmolVLM Settings", open=True):
                            with gr.Row():
                                caption_style = gr.Dropdown(
                                    choices=CAPTION_STYLE_OPTIONS,
                                    value=smolvlm_defaults.get('caption_style', CAPTION_STYLE_OPTIONS[1] if len(CAPTION_STYLE_OPTIONS) > 1 else CAPTION_STYLE_OPTIONS[0] if CAPTION_STYLE_OPTIONS else "Moderately detailed"),
                                    label="Caption Style"
                                )
                            gr.Markdown("Sampler Settings") 
                            with gr.Row():
                                max_tokens = gr.Slider(minimum=50, maximum=1024, value=smolvlm_defaults.get('max_new_tokens', MAX_NEW_TOKENS), step=1, label="Max New Tokens")
                                rep_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=smolvlm_defaults.get('repetition_penalty', REP_PENALTY), step=0.1, label="Repetition Penalty")
                            
                            # Group the sampling-related controls together
                            with gr.Group():
                                do_sample_checkbox = gr.Checkbox(value=smolvlm_defaults.get('do_sample', DO_SAMPLING), label="Do Sample")
                                with gr.Row():
                                    temperature_slider = gr.Slider(minimum=0.1, maximum=1.0, value=smolvlm_defaults.get('temperature', TEMP), step=0.1, label="Temperature")
                                    top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=smolvlm_defaults.get('top_p', TOP_P), step=0.1, label="Top P")


                
                with gr.Row():
                    with gr.Column():
                        image_caption = gr.Textbox(label="Generated Caption",
                                                   lines=5,
                                                   interactive=True,
                                                   elem_id="text_box",
                                                   placeholder="Can be left blank but captions nearly always produce better SUPIR results." \
                                                   "You can also edit the caption after it has been generated.")
                

                gr.Markdown("""    
                            `Max New Tokens`: Controls the maximum length of the generated caption   
                            `Repetition Penalty`: Higher values discourage repetition in the text                                 
                            `Do Sample`: Enabled: uses Top P sampling for more diverse outputs. Disabled: use greedy mode (deterministic)  
                            `Temperature`: Higher values (>1.0) = output more random, lower values = more deterministic  
                            `Top P`: Higher values (0.8-0.95): More variability, more diverse outputs, Lower values (0.1-0.5): Less variability, more consistent outputs  
                            """)                

                        # Add the Process button under the second column
                        # process_btn = gr.Button("Continue", variant="primary")
            
            # ==============================================================================================
            # TAB 2 - SUPIR
            # ==============================================================================================
            with gr.TabItem("2. SUPIR"):
                # gr.Markdown("Restore/Enhance/Upscale")
                
                # -------------------------------------------------
                # ROW - OUTER
                # -------------------------------------------------
                with gr.Row():
                                        
                    # -------------------------------------------------
                    # COL 1
                    # -------------------------------------------------
                    # Left column content
                    with gr.Column(elem_classes=["fixed-width-column-600"]):

                        with gr.Group():
                            loading_half_params = gr.Checkbox(value=supir_defaults.get('loading_half_params', True), label="Load Model fp16")
                            with gr.Row():
                                # supir_sign renamed to supir_model
                                supir_model = gr.Dropdown(choices=["Q", "F"], value=supir_defaults.get('supir_model_type', "Q"), label="Model Type")
                                # load half - (For <= 24GB VRAM)
                                

                                # sampler type[RestoreEDMSampler, TiledRestoreEDMSampler]
                                # internally returns the config_path to the correct config (yaml)
                                # RestoreEDMSampler (Higher VRAM)
                                # TiledRestoreEDMSampler (Lower VRAM)
                                sampler_type = gr.Dropdown(
                                    choices=["RestoreEDMSampler", "TiledRestoreEDMSampler"],
                                    value=supir_defaults.get('sampler_type', 'TiledRestoreEDMSampler'),
                                    label="Sampler Type"
                                )
                            with gr.Row():
                                

                                ae_dtype = gr.Dropdown(
                                    choices=["fp32", "bf16"],
                                    value=supir_defaults.get('ae_dtype', "bf16"),
                                    label="AE dType"
                                )
                                diff_dtype = gr.Dropdown(
                                    choices=["fp32", "fp16", "bf16"],
                                    value=supir_defaults.get('diff_dtype', "fp16"),
                                    label="Diffusion dType"
                                )

                        with gr.Group():
                            with gr.Row():                                
                                use_upscale_to = gr.Checkbox(value=supir_defaults.get('use_upscale_to', False), label="Use Upscale to...")
                            with gr.Row():
                                upscale_to_width = gr.Number(label="Upscale to width",value=supir_defaults.get('upscale_to_width', 1024),step=64,minimum=1024,maximum=8192, interactive=True)
                                upscale_to_height = gr.Number(label="Upscale to height",value=supir_defaults.get('upscale_to_height', 1024),step=64,minimum=1024,maximum=8192, interactive=True)
                                # UPSCALE FACTOR                                            
                                upscale_by = gr.Slider(minimum=1.0, maximum=10.0, value=supir_defaults.get('upscale_by', 2.0), step=0.25, label="Upscale by", interactive=True, scale=2)
                                
                            # with gr.Row():
                                # skip_denoise_stage = gr.Checkbox(value=False, label="Skip Denoise Stage", info="Use if input image is already clean and high quality.")

                        with gr.Group():
                            with gr.Row():
                                # SEED
                                seed = gr.Number(value=supir_defaults.get('seed', 1234567891), precision=0, label="Seed", interactive=True)
                                skip_denoise_stage = gr.Checkbox(value=supir_defaults.get('skip_denoise_stage', False), label="Skip Denoise Stage", info="Use if input image is already clean and high quality.")

                        # with gr.Group():
                        #     with gr.Row():
                                
                        #     with gr.Row():

                        

                        # Tile settings
                        with gr.Group() as tile_vae_settings:
                            with gr.Row():
                                use_tile_vae = gr.Checkbox(value=supir_defaults.get('use_tile_vae', True), label="Use Tile VAE")
                            with gr.Row():                                
                                # The AE processes the input image in tiles of specified size instead of the full image at once
                                encoder_tile_size = gr.Slider(minimum=256, maximum=3072, value=supir_defaults.get('encoder_tile_size', 512), step=64, label="Encoder Tile Size")
                                # The AE reconstructs the final image by stitching together outputs from smaller tile segments
                                decoder_tile_size = gr.Slider(minimum=64, maximum=256, value=supir_defaults.get('decoder_tile_size', 64), step=16, label="Decoder Tile Size")
                                num_of_workers = gr.Slider(minimum=1, maximum=8, value=supir_defaults.get('num_workers', 2), step=1, label="Number of workers")
                            
                                
                                                    
        
                    # -------------------------------------------------
                    # COL 2                    
                    # -------------------------------------------------
                    with gr.Column(elem_classes=["fixed-width-column-600"]):
                                            
                        with gr.Group():
                            gr.Markdown("Steps, S-Churn, S-Noise")
                            with gr.Row():
                                edm_steps = gr.Slider(minimum=10, maximum=100, value=supir_defaults.get('edm_steps', 50), step=1, label="Steps") # sampler steps
                                s_churn = gr.Slider(minimum=0, maximum=20, value=supir_defaults.get('s_churn', 5), step=1, label="S-Churn") # stochastic churn
                                s_noise = gr.Slider(minimum=1.0, maximum=2.0, value=supir_defaults.get('s_noise', 1.003), step=0.001, label="S-Noise") # stochastic noise                        

                        
                                                         
                        with gr.Group():
                            gr.Markdown("CFG Guidance")
                            with gr.Row():
                                cfg_scale_start = gr.Slider(minimum=0.0, maximum=10.0, value=supir_defaults.get('cfg_scale_start', 2.0), step=0.1, label="CFG Scale Start")
                                cfg_scale_end = gr.Slider(minimum=1.0, maximum=15.0, value=supir_defaults.get('cfg_scale_end', 4.0), step=0.1, label="CFG Scale End")
                        
                        

                        with gr.Group():
                            gr.Markdown("Control Guidance")                            
                            with gr.Row():
                                control_scale_start = gr.Slider(minimum=0.0, maximum=2.0, value=supir_defaults.get('control_scale_start', 0.9), step=0.05, label="Control Scale Start")
                                control_scale_end = gr.Slider(minimum=0.0, maximum=2.0, value=supir_defaults.get('control_scale_end', 0.9), step=0.05, label="Control Scale End")

                        with gr.Row():
                            restoration_scale = gr.Slider(minimum=0, maximum=4.0, value=supir_defaults.get('restoration_scale', 0), step=0.5, label="Restoration Scale(≤0 = Disabled)", info="Still a mystery, keep disabled unless image is very damaged")

                        with gr.Group():
                            gr.Markdown("Sampler Tiling (For TiledRestoreEDMSampler)")                            
                            with gr.Row():
                                sampler_tile_size = gr.Slider(minimum=128, maximum=512, value=supir_defaults.get('sampler_tile_size', 128), step=32, label="Sampler Tile Size")
                                sampler_tile_stride = gr.Slider(minimum=32, maximum=256, value=supir_defaults.get('sampler_tile_stride', 64), step=32, label="Sampler Tile Stride")

                        with gr.Group():
                            pass
                        with gr.Group():
                            pass
                        with gr.Group():
                            pass                        
                        with gr.Group():
                            pass                                                

                with gr.Row():
                    with gr.Column(elem_classes=["fixed-width-column-1216"]):
                        with gr.Accordion("Additional Prompt/Neg Prompt", open=False):
                            with gr.Row():
                                a_prompt = gr.Textbox(value=supir_defaults.get('a_prompt', default_positive_prompt), lines=4, label="Additional Positive Prompt (appended to main caption)")
                                n_prompt = gr.Textbox(value=supir_defaults.get('n_prompt', default_negative_prompt), lines=4, label="Negative Prompt")

                        process_supir_btn = gr.Button("Process", variant="primary")
                        # status message box 
                        status_message = gr.Textbox(label="", interactive=False)
                         
                # # additional prompt and standard neg. prompt
                # with gr.Accordion("Additional Prompt/Neg Prompt", open=False):
                #     with gr.Row():
                #         a_prompt = gr.Textbox(value=default_positive_prompt, lines=4, label="Additional Positive Prompt (appended to main caption)")
                #         n_prompt = gr.Textbox(value=default_negative_prompt, lines=4, label="Negative Prompt")

                # with gr.Row():
                #     with gr.Column():
                #         process_supir_btn = gr.Button("Process", variant="primary")

                #         # status message box 
                #         status_message = gr.Textbox(label="", interactive=False)         

                gr.Markdown("Default Settings can be set in the file `defaults.json`. If it doesn't exist, just copy and rename `defaults_example.json`")

                gr.Markdown(
                    """
                | **Parameter** | **Description** |
                |---------------|-----------------|
                | `Load Model fp16` | Loads the SUPIR model weights in half precision (FP16). Reduces VRAM usage and increases speed at the cost of slight precision loss. |
                | `Model Type` | - `Q model (Quality)`: <br>Optimized for moderate - heavy degradations. High generalization, high image quality in most cases, <br>but may overcorrect or hallucinate when used on lightly degraded images. <br>- `F model (Fidelity)`:<br>Optimized for mild degradations, preserving fine details and structure. Ideal for high-fidelity tasks with subtle restoration needs. |
                | `Sampler Type` | - `RestoreEDMSampler`: Uses more VRAM. <br>- `TiledRestoreEDMSampler`: Uses less VRAM. |
                | `AE dType` | Autoencoder precision. [`bf16`, `fp32`]|
                | `Diffusion dType` | Diffusion precision. Overrides the default precision of the loaded model, unless `Load Model fp16` is already set.<br>[`bf16`, `fp16`,`fp32`] |
                | `Seed` | Fixed or random seed. |
                | `Use Upscale to..`| If on, use `Update to width` and `Update to height` values for upscaling. If off, then `Upscale by` factor will be used. 
                | `Upscale to width`| Upscale input image width to specified dimension if `Use Upscale to..` is on. <br>Minimum: 1024 |
                | `Upscale to height`| Upscale input image height to specified dimension if `Use Upscale to..` is on. <br>Minimum: 1024 |
                | `Upscale by` | Upscale factor for the input image. <br>Default: `2` <br>Upscaling of the input image is performed before the denoising and sampling stage. <br>Both dimensions are multiplied by the upscale value. If the smaller of the dimensions is still < 1024px, the image is further enlarged to minimum of<br>1024px (aspect ratio maintained).  |
                | *** | **Notes about Upscaling**: <br>The reason for the minimum of 1024 is to give SDXL a comfortable working resolution.  **Note** that dimensions are snapped to the nearest multiple <br>of 64. The sweet spot seems to be between 2x and 4x (1024x1024) or 4x and 8x (512x512). Beyond that, the quality begins to collapse. <br>The higher the scale factor, the slower the process.| 
                | `Skip Denoise Stage` | Skips the VAE Denoiser Stage. Default: `'False'`<br> Bypass the artifact removal preprocessing step that uses the specialized VAE denoise encoder. This usually ends up with the image slightly softened <br>(if you inspected it at this stage). This is to avoid SUPIR treating low-res/compression artifacts as detail to be enhanced. <br>You may wish to skip this step if <br> - 1) You want do do your own pre-processing OR <br> - 2) Your input image is clean and free of low-res/compression artifacts or other degradations <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Can sometimes make closeups of skin textures a bit unnatural.|
                | `Use VAE Tile` | Enable tiled VAE encoding/decoding for large images. Saves VRAM. |
                | `Encoder Tile Size` | Tile size when encoding. <br> TileVAE code has recommended tile sizes based on available VRAM if a CUDA device is available. <br>Encoder: <br>- VRAM > 16GB: 3072<br>- VRAM > 12GB: 2048<br>- VRAM > 8GB: 1536<br>- VRAM <= 8GB: 960<br>- No GPU: 512 |
                | `Decoder Tile Size` | Tile size when encoding. <br> TileVAE code has recommended tile sizes based on available VRAM if a CUDA device is available. <br>Decoder: <br>- VRAM > 30GB: 256<br>- VRAM > 16GB: 192<br>- VRAM > 12GB: 128<br>- VRAM > 8GB: 96<br>- VRAM <= 8GB: 64<br>- No GPU: 64 |
                | `Number of Workers` | Number of parallel CPU processes for VAE encoding/decoding. <br>Improves speed on the CPU by efficiently preparing data for the GPU. Increase if you have a ton of Memory<br>Default: `1` |
                | `Steps` | Number of diffusion steps. Default: `50` |
                | `S-Churn` | Controls how much extra randomness is added during the process. This helps the model explore a more varied result. Default: `5` <br>`0`: No noise (deterministic) <br>`1-5`: Mild/moderate <br>`6-10+`: Strong |
                | `S-Noise` | Scales S-Churn noise strength. Default: `1.003` <br>Slightly < 1: More stable <br>Slightly > 1: More variation |
                | `CFG Guidance Scale` | Guides how much to adhere to the prompt and conditioning<br>- `CFG Scale Start`: Prompt guidance strength start. Default: `2.0` <br>- `CFG Scale End`: Prompt guidance strength end. Default: `4.0` <br>If `Start` and `End` have the same value, no scaling occurs. When they differ, linear scheduling is applied from `Start` to `End`. <br>Start can be greater than End (or vice versa), depending on whether you want creative freedom early or later. |
                | `Control Guidance Scale` | Guides how strongly the overall structure of the input image is preserved<br>- `Control Scale Start`: Structural guidance from input image, start strength. Default: `0.9` <br>- `Control Scale End`: Structural guidance from input image, end strength. Default: `0.9` |
                | `Restoration Scale` | Early-stage restoration strength. <br>Controls how strongly the model pulls the structure of the output image back toward the original image. <br>Only applies during the early stages of sampling when the noise level is high.<br>Default: `≤0` (disabled). |
                | `Sampler Tile Size` | Tile size for when using `TiledRestoreEDMSampler` sampler. |
                | `Sampler Tile Stride` | Tile stride for when using `TiledRestoreEDMSampler` sampler. Controls how much tiles overlap during sampling. <br>A **smaller** tile_stride means **more** overlap between tiles, better blending, reduces seams, but increases computation. <br>A **larger** tile_stride means **less** overlap (or none), which is faster but may cause visible seams near tile boundaries. <br>`Overlap = tile_size - tile_stride` <br>`Greater overlap ⇨ smaller stride` <br>`Less overlap ⇨ larger stride` <br>Example: `tile_size` = 128 and `tile_stride` = 64 → 64px overlap. |
                | `Additional Positive Prompt` | Additional positive prompt (appended to input caption). The default is taken from SUPIR's own demo code. |
                | `Negative Prompt` | Negative prompt used for all images. The default is taken from SUPIR's own demo code. |
                    """
                )
     

            
            # ==============================================================================================
            # TAB 3 - RESULTS
            # ==============================================================================================
            with gr.TabItem("3. Results"):
                with gr.Row():
                    with gr.Column(elem_classes=["fixed-width-column-1216"]):
        
                        # Output component - Native ImageSlider
                        output_slider = gr.ImageSlider(
                            type="pil", 
                            label="Before / After (Slide to compare), Mouse Wheel to Zoom",
                            height=900, # height of container
                            max_height=900, # max height of image
                            container=True,
                            interactive=False,
                            slider_position=50  # Default position at 50%
                        )


        # Choose the appropriate generate function based on the argument 'use_stream'
        # and assign to function reference 'generate_function'  
        # if use_stream:
        # generate_function = generate_caption_streaming 
        # else:
        #     generate_function = generate_caption_non_streaming


        # ==============================================================================================
        # Tab 1 Event Handler(s)
        # ==============================================================================================        
        submit_btn.click(fn=generate_caption_streaming,
                        inputs=[
                            input_image,
                            caption_style,
                            max_tokens,
                            rep_penalty,
                            do_sample_checkbox,
                            temperature_slider,
                            top_p_slider
                        ],
            outputs=[image_caption]
        )

        # input_image.change(fn=get_image_dimensions, inputs=input_image, outputs=image_dims)
        input_image.upload(fn=get_image_dimensions, inputs=input_image, outputs=image_dims)

        # ==============================================================================================
        # Tab 2 Event Handlers
        # ==============================================================================================
        
        process_supir_btn.click(
            fn=process_supir,
            # The gradio Input components(s) whose values are passed to the function
            inputs=[
                input_image,
                image_caption,
                supir_model,
                sampler_type,
                seed,
                upscale_by,
                use_upscale_to,
                upscale_to_width,
                upscale_to_height,
                skip_denoise_stage,
                loading_half_params,  
                ae_dtype,            
                diff_dtype,          
                use_tile_vae, 
                encoder_tile_size, 
                decoder_tile_size,
                num_of_workers,
                edm_steps,
                s_churn,
                s_noise,
                cfg_scale_start,
                cfg_scale_end,
                control_scale_start, 
                control_scale_end, 
                restoration_scale,
                sampler_tile_size,
                sampler_tile_stride,
                a_prompt, 
                n_prompt
            ],        
            # The gradio component(s) where the function's return value(s) is displayed
            outputs=[output_slider, status_message] 
        )
        
        # ==============================================================================================
        # Tab 3 Event Handlers
        # ==============================================================================================
        




        # def export_function(text):
        #     return "Export functionality would save: " + text
            


        server_name = "0.0.0.0" if listen_on_network else None
        demo.launch(server_name=server_name, server_port=port)
        

def main():
    

    # Clear CUDA cache and garbage collect at startup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("CUDA cache cleared.")
        import gc
        gc.collect()

    # Parse CLI arguments (can be passed manually as `argv` for testing)
    parser = argparse.ArgumentParser(description="Run SmolVLM with Gradio")
    parser.add_argument("--listen", action="store_true", help="Launch Gradio with server_name='0.0.0.0' to listen on all interfaces")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio on (default: 7860)")
    args = parser.parse_args()

    # Set model path to global SMOLVLM_MODEL_PATH
    # required by generate_caption_streaming() and generate_caption_non_streaming()
    global SMOLVLM_MODEL_PATH
    SMOLVLM_MODEL_PATH = f"models/SmolVLM-500M-Instruct"
    
    
    # Check SMOLVLM MODEL FILES ARE OKAY
    filesokay = check_smolvlm_model_files(SMOLVLM_MODEL_PATH)
    if not filesokay:
        print(f"ERROR: Required model files not found at {SMOLVLM_MODEL_PATH}", color.MAGENTA)
        print("Please download the model files manually and try again.", color.MAGENTA)
        sys.exit(1)  # Exit with error code 1        


    SUPIR_PATH = "models/SUPIR"
    filesokay = check_supir_model_files(SUPIR_PATH)
    if not filesokay:
        print(f"ERROR: Required SUPIR files not found for at {SUPIR_PATH}", color.MAGENTA)
        print("Please download the model files manually and try again.", color.MAGENTA)

    CLIP1_PATH = "models/CLIP1"
    filesokay = check_clip_model_file(CLIP1_PATH)
    if not filesokay:
        print(f"ERROR: Required CLIP1 file not found for at {CLIP1_PATH}", color.MAGENTA)
        print("Please download the model files manually and try again.", color.MAGENTA)

    CLIP2_PATH = "models/CLIP2"
    filesokay = check_clip_model_file(CLIP2_PATH)
    if not filesokay:
        print(f"ERROR: Required CLIP2 file not found for at {CLIP2_PATH}", color.MAGENTA)
        print("Please download the model files manually and try again.", color.MAGENTA)

    # for sdxl we will just check for any safetensors file (since any can be used)
    SDXL_PATH = "models/SDXL"
    filesokay = check_for_any_sdxl_model(SDXL_PATH)
    if not filesokay:
        print(f"ERROR: No sdxl safetensors file not found for at {SDXL_PATH}", color.MAGENTA)
        print("Please download your preferred sdxl model and try again.", color.MAGENTA)
    
    # Attach to Gradio (if needed)
    create_launch_gradio(args.listen, args.port)

# Launch the Gradio app
if __name__ == "__main__":
    main()
