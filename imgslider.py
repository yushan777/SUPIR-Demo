import gradio as gr
from gradio_imageslider import ImageSlider
from PIL import Image

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
        # Using LANCZOS resampling for better quality when upscaling
        image1 = image1.resize((img2_width, img2_height), Image.LANCZOS)
        print(f"Resized first image from {img1_width}x{img1_height} to {img2_width}x{img2_height}")
    
    # Return a tuple of the resized first image and the second image
    return (image1, image2)

def update_slider_color(color):
    """Update the slider color and return the component with the new color."""
    return gr.ImageSlider(slider_color=color, type="pil")

# Create the Gradio interface
with gr.Blocks(title="Two-Image Comparison with Slider") as demo:
    gr.Markdown("# Two-Image Comparison")
    gr.Markdown("Upload two images to compare them side by side using the slider.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input components for two separate images
            image1 = gr.Image(type="pil", label="Upload First Image")
            image2 = gr.Image(type="pil", label="Upload Second Image")
            slider_color_picker = gr.ColorPicker(value="#1E88E5", label="Slider Color")
            compare_button = gr.Button("Compare Images")
        
        with gr.Column(scale=2):
            # Output component - ImageSlider
            output_slider = ImageSlider(type="pil", label="Image Comparison (Slide to compare)", 
                                      slider_color="#1E88E5")
    
    # Set up event handlers
    compare_button.click(
        fn=combine_images,
        inputs=[image1, image2],
        outputs=output_slider
    )
    
    slider_color_picker.change(
        fn=update_slider_color,
        inputs=[slider_color_picker],
        outputs=output_slider
    )
    
    # No examples section
    
    gr.Markdown("## How to use")
    gr.Markdown("""
    1. Upload two images you want to compare
    2. Click 'Compare Images' to see them side by side
    3. Use the image slider to compare the two images
    4. Optionally, change the slider color using the color picker
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()