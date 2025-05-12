import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForImageTextToText
import gradio as gr
from colored_print import color, style
import os
import time
from threading import Thread
from transformers.generation.streamers import TextIteratorStreamer
from huggingface_hub import snapshot_download
import xxhash
import json

# macOS shit, just in case some pytorch ops are not supported on mps yes, fallback to cpu
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# # Parse command line arguments
# parser = argparse.ArgumentParser(description="Run SmolVLM with Gradio")
# parser.add_argument("--use_stream", action="store_true", help="Use streaming mode for text generation")
# parser.add_argument("--model", 
#                     choices=["SmolVLM-Instruct", "SmolVLM-500M-Instruct", "SmolVLM-256M-Instruct"],
#                     default="SmolVLM-Instruct", 
#                     help="Model to use (default: SmolVLM-Instruct)")
# args = parser.parse_args()

# # MODEL SELECTION AND PATH
# MODEL_PATH = f"model/{args.model}"

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
    "Brief and concise": "Caption this image with a brief and concise description.",
    "Moderately detailed": "Caption this image with a moderately detailed description.",
    "Highly detailed": "Caption this image with a highly detailed and comprehensive description."
}
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

# ==============================================================
def download_model_from_HF(model_path):
    
    # Download model from HF.
    REPO_NAME = f"yushan777/{os.path.basename(model_path)}"

    try:
        print(f"⬇️ Downloading model from HuggingFace repo: {REPO_NAME}", color.ORANGE)
        
        # Download the repository to the specified path
        snapshot_download(
            repo_id=REPO_NAME,
            local_dir=model_path,
            local_dir_use_symlinks=False,  
        )
        
        print(f"✅ Model downloaded successfully", color.GREEN)
        
        # Verify the downloaded files
        if validate_model_files(model_path):
            print(f"✅ Downloaded files validated", color.GREEN)
            return True
        else:
            print(f"❌ Downloaded files validation failed", color.RED)
            return False
            
    except Exception as e:
        print(f"❌ Failed to download model: {str(e)}", color.RED)
        return False

# ==============================================================
def check_model_files(model_path):
    print(f"Checking Model Path: {model_path}", color.ORANGE)

    # validate model files first
    if validate_model_files(model_path):
        print(f"✅ All model files are valid", color.GREEN)
        return True
    
    # If we get here, either directory doesn't exist or files are invalid
    print(f"⚠️ Model files are missing or corrupted - attempting to download", color.YELLOW)
    return False 

            
# ==============================================================
def hash_file(filepath, chunk_size=1024 * 1024):  # default 1MB
    h = xxhash.xxh3_64()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

# ==========================================================================
def validate_model_files(model_path, chunk_size=1024 * 1024, config_path="model_checksums.json"):
    # files hashed with xxhash.xxh3_64()
    
    # Load file hash data from JSON
    try:
        with open(config_path, 'r') as f:
            model_configs = json.load(f)
    except Exception as e:
        print(f"❌ Error loading config file: {str(e)}")
        return False
        
    # Determine which model we're validating
    model_type = None
    for possible_type in model_configs.keys():
        if possible_type in model_path:
            model_type = possible_type
            break
            
    if not model_type:
        print(f"❌ Unknown model type: {model_path}")
        return False
        
    required_files = model_configs[model_type]  

    valid = True  # assume everything is fine until proven otherwise

    for file_info in required_files:
        file_path = os.path.join(model_path, file_info["name"])
        
        if not os.path.isfile(file_path):
            print(f"❌ Missing file: {file_info['name']}", color.RED)
            valid = False
            continue

        try:
            file_hash = hash_file(file_path, chunk_size=chunk_size)

            if file_hash == file_info["hash"]:    
                # print(f' - {file_info["name"]}: {file_hash}: OKAY', color.BRIGHT_GREEN)
                pass
            else:
                print(f' - {file_info["name"]}: {file_hash}: MISMATCH. Expected {file_info["hash"]}', color.BRIGHT_RED)
                valid = False

        except Exception as e:
            print(f'❌ Error checking hash for {file_info["name"]}: {str(e)}', color.RED)
            valid = False

    return valid

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
    custom_prompt, # Added custom_prompt
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
    
    if custom_prompt and custom_prompt.strip():
        prompt_text = custom_prompt.strip()
    else:
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

    # # Yield generated text as it comes in
    # generated_text_so_far = ""
    # for new_text_chunk in streamer:
    #     generated_text_so_far += new_text_chunk
    #     yield generated_text_so_far

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
    custom_prompt, # Added custom_prompt
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
        
    if custom_prompt and custom_prompt.strip():
        prompt_text = custom_prompt.strip()
    else:
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
# GRADIO SHIT
# ====================================================================
def launch_gradio(use_stream):

    # Create custom theme
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
                    """) as demo:   
        
        gr.Markdown("# Image Captioner : SmolVLM-Instruct")    
        gr.Markdown(f"**Model**: {model_name} | **Mode**: {mode}")        
        gr.Markdown("Upload an image and adjust the settings to generate a caption")
        
        with gr.Row():
            # ================================================
            # COL 1
            with gr.Column(elem_classes=["fixed-width-column"]):
                input_image = gr.Image(type="pil", label="Input Image", height=512)
                                        
                submit_btn = gr.Button("Generate Caption", variant="primary")
                
            # ================================================
            # COL 2                    
            with gr.Column(elem_classes=["fixed-width-column"]):
                
                custom_prompt_textbox = gr.Textbox(
                                            label="Custom Query/Prompt (Optional)", 
                                            placeholder="Enter your custom query here, or select a caption preset below.", 
                                            lines=2
                                            )

                caption_style = gr.Dropdown(
                    choices=CAPTION_STYLE_OPTIONS,
                    value=CAPTION_STYLE_OPTIONS[1] if len(CAPTION_STYLE_OPTIONS) > 1 else CAPTION_STYLE_OPTIONS[0] if CAPTION_STYLE_OPTIONS else "Moderately detailed",
                    label="Caption Style"
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        max_tokens = gr.Slider(minimum=50, maximum=1024, value=MAX_NEW_TOKENS, step=1, label="Max New Tokens")
                        rep_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=REP_PENALTY, step=0.1, label="Repetition Penalty")

                    # Group the sampling-related controls together
                    with gr.Group():
                        do_sample_checkbox = gr.Checkbox(value=DO_SAMPLING, label="Do Sample")
                        with gr.Row():
                            temperature_slider = gr.Slider(minimum=0.1, maximum=1.0, value=TEMP, step=0.1, label="Temperature")
                            top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=TOP_P, step=0.1, label="Top P")

                    gr.Markdown("""    
                                ### Parameters:
                                - **Max New Tokens**: Controls the maximum length of the generated caption
                                - **Repetition Penalty**: Higher values discourage repetition in the text
                                - **Do Sample**: Enabled: uses Top P sampling for more diverse outputs. Disabled: use greedy mode (deterministic)
                                - **Temperature**: Higher values (>1.0) = output more random, lower values = more deterministic
                                - **Top P**: Higher values (0.8-0.95): More variability, more diverse outputs, Lower values (0.1-0.5): Less variability, more consistent outputs
                                
                                """)
                    
        with gr.Row():
            with gr.Column():
                output_text = gr.Textbox(label="Generated Caption", lines=5, interactive=True, elem_id="text_box", info="you can edit the caption here before proceeding")
        
                # Add the Process button under the second column
                process_btn = gr.Button("Continue", variant="primary")

        # Choose the appropriate generate function based on the argument
        generate_function = generate_caption_streaming if use_stream else generate_caption_non_streaming

        submit_btn.click(
            fn=generate_function,
            inputs=[
                input_image,
                custom_prompt_textbox, # Added custom_prompt_textbox
                caption_style,
                max_tokens,
                rep_penalty,
                do_sample_checkbox,
                temperature_slider,
                top_p_slider
                
            ],
            outputs=[output_text]
        )

        # Add the click handler for the Process button
        process_btn.click(
            fn=process_edited_caption,
            inputs=[output_text]
        )

        demo.launch()
        

def main():
    global processor, model, DEVICE, MODEL_PATH

    # Parse CLI arguments (can be passed manually as `argv` for testing)
    parser = argparse.ArgumentParser(description="Run SmolVLM with Gradio")
    parser.add_argument("--use_stream", action="store_true", help="Use streaming mode for text generation")
    parser.add_argument("--model", 
                        choices=["SmolVLM-Instruct", "SmolVLM-500M-Instruct", "SmolVLM-256M-Instruct"],
                        default="SmolVLM-Instruct", 
                        help="Model to use (default: SmolVLM-Instruct)")
    args = parser.parse_args()

    # Set model path
    MODEL_PATH = f"model/{args.model}"
    
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
    launch_gradio(args.use_stream)

# Launch the Gradio app
if __name__ == "__main__":
    main()
