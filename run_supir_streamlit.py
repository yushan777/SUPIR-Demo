import argparse
import torch
from PIL import Image, PngImagePlugin
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForImageTextToText
import streamlit as st
from Y7.colored_print import color, style
import os
import sys
import time
import glob
from threading import Thread
from transformers.generation.streamers import TextIteratorStreamer
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype

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
SUPIR_AE_DTYPE = 'ae_dtype'
SUPIR_DIFF_DTYPE = 'diff_dtype'
SUPIR_DIFF_SAMPLER_TILE_SIZE = 'sampler_tile_size'
SUPIR_DIFF_SAMPLER_TILE_STRIDE = 'sampler_tile_stride'

# Default prompts (the positive is appended to the main caption (whether it exists or not))
default_positive_prompt = 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.'
default_negative_prompt = 'painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth'

# list of just the keys for dropdown
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
                print(f"  flash_attention_2 not available (package not installed)", color.YELLOW)
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
        model.init_tile_vae(encoder_tile_size=encoder_tile_size, decoder_tile_size=decoder_tile_size)

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
def generate_caption(
    image,
    caption_style,
    max_new_tokens=MAX_NEW_TOKENS,
    repetition_penalty=REP_PENALTY,
    do_sample=DO_SAMPLING,
    temperature=TEMP,
    top_p=TOP_P
):
    """Generate caption for the image"""
    # Check if image is provided, if not, quit and show msg
    if image is None:
        return "Please upload an image first to generate a caption."
    
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

    # Create a placeholder in Streamlit
    caption_placeholder = st.empty()
    caption_placeholder.text("Generating caption...")

    # Run generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_args)
    thread.start()

    # Display generated text as it comes in
    generated_text_so_far = ""
    is_first_chunk = True # used for stripping away leading space from first chunk (below)
    
    for new_text_chunk in streamer:
        # Strip leading space from the first chunk only
        if is_first_chunk:
            new_text_chunk = new_text_chunk.lstrip()
            is_first_chunk = False
        
        generated_text_so_far += new_text_chunk
        caption_placeholder.text(generated_text_so_far)

    thread.join()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Optional: print final caption to console for logging
    if generated_text_so_far:
        print(f"Generated caption: '{generated_text_so_far.strip()}'", color.GREEN)
    print(f"Execution_time = {execution_time:.2f} seconds.", color.BRIGHT_BLUE)
    
    return generated_text_so_far

# process SUPIR on the image
def process_supir(
            input_image,
            image_caption, 
            supir_model_type,  
            sampler_type,
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
            sampler_tile_size,
            sampler_tile_stride,            
            a_prompt, 
            n_prompt
        ):
    
    # Check if input_image is provided, if not, quit and show msg
    if input_image is None:
        # Return a tuple with None for the image and an error message
        return None, "Please upload an image first in Tab 1."
        
    
    # Clear CUDA cache and garbage collect at startup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("CUDA cache cleared at startup")
        import gc
        gc.collect()
            
    print("SUPIR Settings\n")
    print(f"input_image: {input_image}", color.YELLOW)
    print(f"image_caption: {image_caption}", color.YELLOW)
    print(f"supir_model_type: {supir_model_type}", color.YELLOW)
    print(f"sampler_type: {sampler_type}", color.YELLOW)
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
    print(f"sampler_tile_size: {sampler_tile_size}", color.YELLOW)
    print(f"sampler_tile_stride: {sampler_tile_stride}", color.YELLOW)
    print(f"a_prompt: {a_prompt}", color.YELLOW)
    print(f"n_prompt: {n_prompt}", color.YELLOW)

    # Create a progress message in Streamlit
    progress_message = st.empty()
    progress_message.info("Processing image with SUPIR...")

    start_time = time.time()

    # Use the global SUPIR_MODEL and SUPIR_SETTINGS variables - declared at top level.
    global SUPIR_MODEL, SUPIR_SETTINGS

    # ONLY Reload SUPIR model if settings critical to its initialization are changed or not loaded yet
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
        SUPIR_SETTINGS.get(SUPIR_AE_DTYPE) != ae_dtype or
        SUPIR_SETTINGS.get(SUPIR_DIFF_DTYPE) != diff_dtype or
        SUPIR_SETTINGS.get(SUPIR_DIFF_SAMPLER_TILE_SIZE) != sampler_tile_size or
        SUPIR_SETTINGS.get(SUPIR_DIFF_SAMPLER_TILE_STRIDE) != sampler_tile_stride):

        progress_message.info("Loading SUPIR model with current settings...")
        SUPIR_MODEL = load_supir_model(
            sampler_type=sampler_type,
            supir_model_type=supir_model_type,
            loading_half_params=loading_half_params,
            use_tile_vae=use_tile_vae,
            encoder_tile_size=encoder_tile_size,
            decoder_tile_size=decoder_tile_size,
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
            SUPIR_AE_DTYPE: ae_dtype,
            SUPIR_DIFF_DTYPE: diff_dtype,
            SUPIR_DIFF_SAMPLER_TILE_SIZE: sampler_tile_size,
            SUPIR_DIFF_SAMPLER_TILE_STRIDE: sampler_tile_stride
        }

    # Convert to PIL if needed
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)
    
    # Process input image by upscaling it to the min_size
    # min_size of 1024 is to make it work nicely with sdxl
    # after sampling if the input image was less than 1024 and upscale was 1, it will be resized back to its original size

    progress_message.info("Preparing image for processing...")
    device = get_device()
    LQ_img, h0, w0 = PIL2Tensor(input_image, upscale=upscale, min_size=1024)
    LQ_img = LQ_img.unsqueeze(0).to(device)[:, :3, :, :]

    print(f"h,w = {h0}, {w0}", color.ORANGE)
    
    # Run diffusion process
    progress_message.info("Running SUPIR diffusion process...")
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
                                    num_samples=1,  # Always 1 for UI 
                                    p_p=a_prompt, 
                                    n_p=n_prompt, 
                                    color_fix_type="Wavelet",
                                    skip_denoise_stage=skip_denoise_stage)
    
    # settings string for embedding into png
    supir_settings = ""
    # supir_settings += f"input_image: {input_image}\n"
    supir_settings += f"image_caption: {image_caption}\n"
    supir_settings += f"supir_model_type: {supir_model_type}\n"
    supir_settings += f"sampler_type: {sampler_type}\n"
    supir_settings += f"seed: {seed}\n"
    supir_settings += f"upscale: {upscale}\n"
    supir_settings += f"skip_denoise_stage: {skip_denoise_stage}\n"
    supir_settings += f"loading_half_params: {loading_half_params}\n"
    supir_settings += f"ae_dtype: {ae_dtype}\n"
    supir_settings += f"diff_dtype: {diff_dtype}\n"
    supir_settings += f"use_tile_vae: {use_tile_vae}\n"
    supir_settings += f"encoder_tile_size: {encoder_tile_size}\n"
    supir_settings += f"decoder_tile_size: {decoder_tile_size}\n"
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
    progress_message.info("Saving processed image...")
    try:
        # Ensure the output directory exists
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # Find existing files that match the current model type
        base_filename = f"streamlit_{supir_model_type}"
        existing_files = glob.glob(os.path.join(output_dir, f"{base_filename}_*.png"))
        
        # Find the highest index used
        max_index = -1
        for f in existing_files:
            try:
                # Extract index from filename like "streamlit_modeltype_0123.png"
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
        info.add_text("SUPIR", supir_settings)

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

    # Clear progress message
    progress_message.success("Processing complete!")
    
    return input_image, enhanced_image, png_save_path

def main():
    # Clear CUDA cache and garbage collect at startup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("CUDA cache cleared.")
        import gc
        gc.collect()

    # Set model path to global SMOLVLM_MODEL_PATH
    global SMOLVLM_MODEL_PATH
    SMOLVLM_MODEL_PATH = f"models/SmolVLM-500M-Instruct"
    
    # Check if models exist
    SUPIR_PATH = "models/SUPIR"
    CLIP1_PATH = "models/CLIP1"
    CLIP2_PATH = "models/CLIP2"
    SDXL_PATH = "models/SDXL"
    
    missing_models = []
    
    if not check_smolvlm_model_files(SMOLVLM_MODEL_PATH):
        missing_models.append(f"SmolVLM at {SMOLVLM_MODEL_PATH}")
    
    if not check_supir_model_files(SUPIR_PATH):
        missing_models.append(f"SUPIR at {SUPIR_PATH}")
    
    if not check_clip_model_file(CLIP1_PATH):
        missing_models.append(f"CLIP1 at {CLIP1_PATH}")
    
    if not check_clip_model_file(CLIP2_PATH):
        missing_models.append(f"CLIP2 at {CLIP2_PATH}")
    
    if not check_for_any_sdxl_model(SDXL_PATH):
        missing_models.append(f"SDXL at {SDXL_PATH}")
    
    # Streamlit app setup
    st.set_page_config(
        page_title="SUPIR Enhancer",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("SUPIR Enhancer / Detailer / Upscaler")
    
    # Show missing models warning if any
    if missing_models:
        st.error("⚠️ Missing required model files:")
        for model in missing_models:
            st.warning(f"- {model}")
        st.warning("Please download the required models before using the application.")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["1. Input Image / SmolVLM Captioner", "2. SUPIR", "3. Results"])
    
    # Initialize session state for storing data between tabs
    if 'input_image' not in st.session_state:
        st.session_state.input_image = None
    if 'image_caption' not in st.session_state:
        st.session_state.image_caption = ""
    if 'enhanced_image' not in st.session_state:
        st.session_state.enhanced_image = None
    if 'output_path' not in st.session_state:
        st.session_state.output_path = None
    
    # Tab 1: Input Image / SmolVLM Captioner
    with tab1:
        st.header("Upload image and generate a caption")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Image upload
            uploaded_file = st.file_uploader("Input Image", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                st.session_state.input_image = Image.open(uploaded_file)
                st.image(st.session_state.input_image, caption="Input Image", use_column_width=True)
            
            # Generate caption button
            if st.button("Generate Caption", key="gen_caption_btn"):
                if st.session_state.input_image is not None:
                    with st.spinner("Generating caption..."):
                        st.session_state.image_caption = generate_caption(
                            st.session_state.input_image,
                            st.session_state.caption_style,
                            st.session_state.max_tokens,
                            st.session_state.rep_penalty,
                            st.session_state.do_sample,
                            st.session_state.temperature,
                            st.session_state.top_p
                        )
                else:
                    st.error("Please upload an image first to generate a caption.")
        
        with col2:
            # SmolVLM Settings
            with st.expander("SmolVLM Settings", expanded=True):
                caption_style_options = CAPTION_STYLE_OPTIONS
                st.session_state.caption_style = st.selectbox(
                    "Caption Style",
                    options=caption_style_options,
                    index=1 if len(caption_style_options) > 1 else 0
                )
                
                st.subheader("Sampler Settings")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.session_state.max_tokens = st.slider("Max New Tokens", 50, 1024, MAX_NEW_TOKENS)
                with col_b:
                    st.session_state.rep_penalty = st.slider("Repetition Penalty", 1.0, 2.0, REP_PENALTY, 0.1)
                
                st.session_state.do_sample = st.checkbox("Do Sample", value=DO_SAMPLING)
                
                col_c, col_d = st.columns(2)
                with col_c:
                    st.session_state.temperature = st.slider("Temperature", 0.1, 1.0, TEMP, 0.1)
                with col_d:
                    st.session_state.top_p = st.slider("Top P", 0.1, 1.0, TOP_P, 0.1)
        
        # Caption text area
        st.session_state.image_caption = st.text_area(
            "Generated Caption", 
            value=st.session_state.image_caption,
            height=150,
            help="Can be left blank but captions nearly always produce better SUPIR results. You can also edit the caption after it has been generated."
        )
        
        # Help text
        st.markdown("""
        **SmolVLM Settings Explained:**
        - `Max New Tokens`: Controls the maximum length of the generated caption
        - `Repetition Penalty`: Higher values discourage repetition in the text
        - `Do Sample`: Enabled: uses Top P sampling for more diverse outputs. Disabled: use greedy mode (deterministic)
        - `Temperature`: Higher values (>1.0) = output more random, lower values = more deterministic
        - `Top P`: Higher values (0.8-0.95): More variability, more diverse outputs, Lower values (0.1-0.5): Less variability, more consistent outputs
        """)
    
    # Tab 2: SUPIR
    with tab2:
        st.header("SUPIR Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Model settings
            with st.expander("Model Settings", expanded=True):
                st.session_state.loading_half_params = st.checkbox("Load Model fp16", value=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.session_state.supir_model_type = st.selectbox(
                        "Model Type", 
                        options=["Q", "F"], 
                        index=0,
                        help="Q: Quality model, F: Fidelity model"
                    )
                with col_b:
                    st.session_state.sampler_type = st.selectbox(
                        "Sampler Type", 
                        options=["RestoreEDMSampler", "TiledRestoreEDMSampler"],
                        index=1,
                        help="RestoreEDMSampler: Higher VRAM, TiledRestoreEDMSampler: Lower VRAM"
                    )
                
                col_c, col_d = st.columns(2)
                with col_c:
                    st.session_state.ae_dtype = st.selectbox(
                        "AE dType", 
                        options=["fp32", "bf16"],
                        index=1
                    )
                with col_d:
                    st.session_state.diff_dtype = st.selectbox(
                        "Diffusion dType", 
                        options=["fp32", "fp16", "bf16"],
                        index=1
                    )
            
            # Basic settings
            with st.expander("Basic Settings", expanded=True):
                col_e, col_f = st.columns(2)
                with col_e:
                    st.session_state.seed = st.number_input("Seed", value=1234567891, step=1)
                with col_f:
                    st.session_state.upscale = st.slider("Upscale", 1.0, 10.0, 2.0, 0.5)
                
                st.session_state.skip_denoise_stage = st.checkbox(
                    "Skip Denoise Stage", 
                    value=False,
                    help="Use if input image is already clean and high quality."
                )
            
            # Tile settings
            with st.expander("Tile Settings", expanded=True):
                st.session_state.use_tile_vae = st.checkbox("Use Tile VAE", value=True)
                
                col_g, col_h = st.columns(2)
                with col_g:
                    st.session_state.encoder_tile_size = st.slider("Encoder Tile Size", 256, 1024, 512, 64)
                with col_h:
                    st.session_state.decoder_tile_size = st.slider("Decoder Tile Size", 32, 128, 64, 8)
        
        with col2:
            # Steps settings
            with st.expander("Steps Settings", expanded=True):
                col_i, col_j, col_k = st.columns(3)
                with col_i:
                    st.session_state.edm_steps = st.slider("Steps", 10, 100, 50, 1)
                with col_j:
                    st.session_state.s_churn = st.slider("S-Churn", 0, 20, 5, 1)
                with col_k:
                    st.session_state.s_noise = st.slider("S-Noise", 1.0, 2.0, 1.003, 0.001)
            
            # CFG Guidance
            with st.expander("CFG Guidance", expanded=True):
                col_l, col_m = st.columns(2)
                with col_l:
                    st.session_state.cfg_scale_start = st.slider("CFG Scale Start", 0.0, 10.0, 2.0, 0.1)
                with col_m:
                    st.session_state.cfg_scale_end = st.slider("CFG Scale End", 1.0, 15.0, 4.0, 0.1)
            
            # Control Guidance
            with st.expander("Control Guidance", expanded=True):
                col_n, col_o = st.columns(2)
                with col_n:
                    st.session_state.control_scale_start = st.slider("Control Scale Start", 0.0, 2.0, 0.9, 0.05)
                with col_o:
                    st.session_state.control_scale_end = st.slider("Control Scale End", 0.0, 2.0, 0.9, 0.05)
            
            # Restoration Scale
            st.session_state.restoration_scale = st.slider(
                "Restoration Scale(≤0 = Disabled)", 
                0.0, 4.0, 0.0, 0.5,
                help="Still a mystery, keep disabled unless image is very damaged"
            )
            
            # Sampler Tiling
            with st.expander("Sampler Tiling (For TiledRestoreEDMSampler)", expanded=True):
                col_p, col_q = st.columns(2)
                with col_p:
                    st.session_state.sampler_tile_size = st.slider("Sampler Tile Size", 128, 512, 128, 32)
                with col_q:
                    st.session_state.sampler_tile_stride = st.slider("Sampler Tile Stride", 32, 256, 64, 32)
        
        # Prompts
        with st.expander("Additional Prompt/Negative Prompt", expanded=False):
            col_r, col_s = st.columns(2)
            with col_r:
                st.session_state.a_prompt = st.text_area(
                    "Additional Positive Prompt (appended to main caption)",
                    value=default_positive_prompt,
                    height=150
                )
            with col_s:
                st.session_state.n_prompt = st.text_area(
                    "Negative Prompt",
                    value=default_negative_prompt,
                    height=150
                )
        
        # Process button
        if st.button("Process SUPIR", key="process_supir_btn"):
            if st.session_state.input_image is not None:
                with st.spinner("Processing image with SUPIR..."):
                    input_image, enhanced_image, output_path = process_supir(
                        st.session_state.input_image,
                        st.session_state.image_caption,
                        st.session_state.supir_model_type,
                        st.session_state.sampler_type,
                        st.session_state.seed,
                        st.session_state.upscale,
                        st.session_state.skip_denoise_stage,
                        st.session_state.loading_half_params,
                        st.session_state.ae_dtype,
                        st.session_state.diff_dtype,
                        st.session_state.use_tile_vae,
                        st.session_state.encoder_tile_size,
                        st.session_state.decoder_tile_size,
                        st.session_state.edm_steps,
                        st.session_state.s_churn,
                        st.session_state.s_noise,
                        st.session_state.cfg_scale_start,
                        st.session_state.cfg_scale_end,
                        st.session_state.control_scale_start,
                        st.session_state.control_scale_end,
                        st.session_state.restoration_scale,
                        st.session_state.sampler_tile_size,
                        st.session_state.sampler_tile_stride,
                        st.session_state.a_prompt,
                        st.session_state.n_prompt
                    )
                    
                    # Store results in session state
                    st.session_state.enhanced_image = enhanced_image
                    st.session_state.output_path = output_path
                    
                    # Automatically switch to results tab
                    st.experimental_set_query_params(tab="results")
            else:
                st.error("Please upload an image first in Tab 1.")
        
        # Settings explanation
        with st.expander("Settings Explanation", expanded=False):
            st.markdown("""
            | **Parameter** | **Description** |
            |---------------|-----------------|
            | `Load Model fp16` | Loads the SUPIR model weights in half precision (FP16). Reduces VRAM usage and increases speed at the cost of slight precision loss. |
            | `Model Type` | - `Q model (Quality)`: <br>Optimized for moderate - heavy degradations. High generalization, high image quality in most cases, <br>but may overcorrect or hallucinate when used on lightly degraded images. <br>- `F model (Fidelity)`:<br>Optimized for mild degradations, preserving fine details and structure. Ideal for high-fidelity tasks with subtle restoration needs. |
            | `Sampler Type` | - `RestoreEDMSampler`: Uses more VRAM. <br>- `TiledRestoreEDMSampler`: Uses less VRAM. |
            | `AE dType` | Autoencoder precision. [`bf16`, `fp32`]|
            | `Diffusion dType` | Diffusion precision. Overrides the default precision of the loaded model, unless `Load Model fp16` is already set.<br>[`bf16`, `fp16`,`fp32`] |
            | `Seed` | Fixed or random seed. |
            | `Upscale` | Upscale factor for the original input image. The higher the scale factor, the slower the process.<br>Default: `2` |
            | `Skip Denoise Stage` | Disables the VAE denoising step that softens low-quality images. Enable only if your input is already clean or high-resolution. |
            | `Use VAE Tile` | Enable tiled VAE encoding/decoding for large images. Saves VRAM. |
            | `Encoder Tile Size` | Tile size when encoding. Default: 512 |
            | `Decoder Tile Size` | Tile size when decoding. Default: 64 |
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
            """)
    
    # Tab 3: Results
    with tab3:
        st.header("Results")
        
        if st.session_state.enhanced_image is not None and st.session_state.input_image is not None:
            # Create a custom image comparison slider
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(st.session_state.input_image, caption="Before", use_column_width=True)
            with col2:
                st.image(st.session_state.enhanced_image, caption="After", use_column_width=True)
            
            # Download button for output image
            if st.session_state.output_path:
                with open(st.session_state.output_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Enhanced Image",
                        data=file,
                        file_name=os.path.basename(st.session_state.output_path),
                        mime="image/png"
                    )
        else:
            st.info("Process an image in tabs 1 and 2 to see results here.")

if __name__ == "__main__":
    # Parse CLI arguments 
    parser = argparse.ArgumentParser(description="Run SUPIR with Streamlit")
    parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on (default: 8501)")
    args = parser.parse_args()
    
    main()