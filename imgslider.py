import gradio as gr
from PIL import Image
import argparse

def combine_images(image1, image2):
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
                image2 = gr.Image(type="pil", label="Upload Second Image", height=320)
                compare_button = gr.Button("Compare Images")
            
            
            with gr.Column(scale=2):
                # Output component - Native ImageSlider
                output_slider = gr.ImageSlider(
                    type="pil", 
                    label="Image Comparer (Slide to compare), Mouse Wheel to Zoom", 
                    height=800, # height of container
                    max_height=800, # max height of image
                    container=False,
                    slider_position=50  # Default position at 50%
                )
        
        # Set up event handler for compare_button
        compare_button.click(
            # Function to call when button is clicked
            fn=combine_images,      
            # The gradio Input components(s) whose values are passed to the function
            inputs=[image1, image2],
            # The gradio Output component(s) where the function's return value(s) is displayed 
            outputs=output_slider   
)
        
        gr.Markdown("## How to use")
        gr.Markdown("""
        1. Upload two images you want to compare
        2. Click 'Compare Images' to see them side by side
        3. Use the image slider to compare the two images
        4. Adjust the slider position using the slider control below
        """)

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

