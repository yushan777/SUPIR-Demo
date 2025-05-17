import gradio as gr
from PIL import Image
import argparse

def combine_images(image1):
    """Combine two images for comparison in the image slider.
    If the images are different sizes, the first image will be resized to match the second."""
    if image1 is None or image2 is None:
        return None
    
    # Get the dimensions of both images
    img1_width, img1_height = image1.size
    img2_width, img2_height = image2.size
    
    # Check if images have different dimensions
    if img1_width != img2_width or img1_height != img2_height:
        # Resize the first image to match the dimensions of the second image
        # Using BICUBIC resampling for better quality when upscaling
        image1 = image1.resize((img2_width, img2_height), Image.BICUBIC)
        print(f"Resized first image from {img1_width}x{img1_height} to {img2_width}x{img2_height}")
    
    # Return a tuple of the resized first image and the second image
    return (image1, image2)

def launch_gradio(listen_on_network, port=None):
    # Create the Gradio interface
    with gr.Blocks(title="Image Comparison with Slider") as demo:
        gr.Markdown("# Two-Image Comparison")
        gr.Markdown("Upload two images to compare them side by side using the slider.")
        
        # Store the current zoom level
        zoom_level = gr.State(value=1.0)

        with gr.Row():
            with gr.Column(scale=1):
                # Input components for two separate images
                image1 = gr.Image(type="pil", label="Upload First Image", height=320)
                # image2 = gr.Image(type="pil", label="Upload Second Image", height=320)
                compare_button = gr.Button("Compare Images")
            
            
            with gr.Column(scale=2):
                # Output component - Native ImageSlider
                output_slider = gr.ImageSlider(
                    type="pil", 
                    label="Image Comparer (Slide to compare), Mouse Wheel to Zoom", 
                    height=200, # height of container
                    max_height=200, # max height of image
                    container=False,
                    slider_position=50  # Default position at 50%
                )
        
        # Set up event handler for compare_button
        compare_button.click(
            # Function to call when button is clicked
            fn=combine_images,      
            # The gradio Input components(s) whose values are passed to the function
            inputs=[image1],
            # The gradio Output component(s) where the function's return value(s) is displayed 
            outputs=output_slider   
)
        
        gr.Markdown(
            """
        | **Parameter** | **Description** |
        |---------------|-----------------|
        | `Load Model fp16` | Loads the SUPIR model weights in half precision (FP16). Reduces VRAM usage and increases speed at the cost of slight precision loss. |
        | `Model Type` | - `Q model (Quality)`: <br>Moderate - heavy degradations. Robust for real-world damage, but may overcorrect or hallucinate when used on lightly degraded images. <br>- `F model (Fidelity)`:<br>Optimized for mild degradations, preserving fine details and structure. Ideal for high-fidelity tasks with subtle restoration needs. |
        | `Sampler Type` | - `RestoreEDMSampler`: Uses more VRAM. <br>- `TiledRestoreEDMSampler`: Uses less VRAM. |
        | `AE dType` | Autoencoder precision. |
        | `Diffusion dType` | Diffusion precision. Overrides the default precision of the loaded model, unless `--loading_half_params` is already set. |
        | `Seed` | Fixed or random seed. |
        | `Upscale` | Upsampling ratio for the input. The higher the scale factor, the slower the process. |
        | `Skip Denoise Stage` | Disables the VAE denoising step that softens low-quality images. Enable only if your input is already clean or high-resolution. |
        | `Use VAE Tile` | Enable tiled VAE encoding/decoding for large images. Saves VRAM. |
        | `Encoder Tile Size` | Tile size when encoding. Default: 512 |
        | `Decoder Tile Size` | Tile size when decoding. Default: 64 |
        | `Steps` | Number of diffusion steps. Default: `50` |
        | `S-Churn` | Adds random noise to encourage variation. Default: `5` <br>`0`: No noise (deterministic) <br>`1–5`: Mild/moderate <br>`6–10+`: Strong |
        | `S-Noise` | Scales churn noise strength. Default: `1.003` <br>Slightly < 1: More stable <br>Slightly > 1: More variation |
        | `CFG Guidance Scale` | - `CFG Scale Start`: Prompt guidance strength start. Default: `2.0` <br>- `CFG Scale End`: Prompt guidance strength end. Default: `4.0` <br>If `Start` and `End` have the same value, no scaling occurs. When they differ, linear scheduling is applied from `Start` to `End`. <br>Start can be greater than End (or vice versa), depending on whether you want creative freedom early or later. |
        | `Control Guidance Scale` | - `Control Scale Start`: Structural guidance from input image, start strength. Default: `0.9` <br>- `Control Scale End`: Structural guidance from input image, end strength. Default: `0.9` |
        | `Restoration Scale` | Early-stage restoration strength. <br>Works as an additional guidance mechanism to the control scale. <br>Specifically targets fine details and textures rather than overall structure. <br>When high, it will try to maintain the exact pixel-level details rather than just the general structure. <br>Default: `-1` (disabled). Typical values: `1–6` |
        | `Sampler Tile Size` | Tile size for when using `TiledRestoreEDMSampler` sampler. |
        | `Sampler Tile Stride` | Tile stride for when using `TiledRestoreEDMSampler` sampler. Controls how much tiles overlap during sampling. <br>A **smaller** tile_stride means **more** overlap between tiles, which helps blend the edges and reduce visible seams, but increases computation. <br>A **larger** tile_stride means **less** overlap (or none), which is faster but may cause visible artifacts at the tile boundaries. <br>`Overlap = tile_size - tile_stride` <br>`Greater overlap ⇨ smaller stride` <br>`Less overlap ⇨ larger stride` <br>Example: `tile_size` = 128 and `tile_stride` = 64 → 64px overlap. |
        | `Additional Positive Prompt` | Additional positive prompt (appended to input caption). The default is taken from SUPIR's own demo code. |
        | `Negative Prompt` | Negative prompt used for all images. The default is taken from SUPIR's own demo code. |
            """
        )
            


    # Launch the app with the provided arguments
    server_name = "0.0.0.0" if listen_on_network else None
    demo.launch(server_name=server_name, server_port=port)
    

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Image Comparison App")
    parser.add_argument("--listen", action="store_true", help="Make the server listen on all network interfaces")
    parser.add_argument("--port", type=int, default=3000, help="Port to run the server on (default: 3000)")
    args = parser.parse_args()

    launch_gradio(args.listen, args.port)

if __name__ == "__main__":
    main()

