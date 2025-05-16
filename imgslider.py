import gradio as gr
from gradio_imageslider import ImageSlider
from PIL import Image, ImageFilter, ImageEnhance

def apply_filters(input_img, filter_type, intensity):
    """Apply different filters to the input image based on the selected filter type."""
    if input_img is None:
        return None
    
    # Create a copy of the input image to avoid modifying the original
    img = input_img.copy()
    
    # Apply the selected filter with the specified intensity
    if filter_type == "Blur":
        filtered_img = img.filter(ImageFilter.GaussianBlur(radius=intensity))
    elif filter_type == "Sharpen":
        enhancer = ImageEnhance.Sharpness(img)
        filtered_img = enhancer.enhance(intensity)
    elif filter_type == "Brightness":
        enhancer = ImageEnhance.Brightness(img)
        filtered_img = enhancer.enhance(intensity)
    elif filter_type == "Contrast":
        enhancer = ImageEnhance.Contrast(img)
        filtered_img = enhancer.enhance(intensity)
    elif filter_type == "Saturation":
        enhancer = ImageEnhance.Color(img)
        filtered_img = enhancer.enhance(intensity)
    else:
        # Default to grayscale if none of the above
        filtered_img = img.convert("L").convert("RGB")
    
    # Return a tuple of the original and filtered images
    return (img, filtered_img)

def update_slider_color(color):
    """Update the slider color and return the component with the new color."""
    return gr.ImageSlider(slider_color=color, type="pil")

# Define the available filter types
filter_types = ["Blur", "Sharpen", "Brightness", "Contrast", "Saturation", "Grayscale"]

# Create the Gradio interface
with gr.Blocks(title="Image Filter Comparison with Slider") as demo:
    gr.Markdown("# Image Filter Comparison")
    gr.Markdown("Upload an image and apply filters to see the comparison using the slider.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input components
            input_image = gr.Image(type="pil", label="Upload Image")
            filter_dropdown = gr.Dropdown(choices=filter_types, value="Blur", label="Filter Type")
            intensity_slider = gr.Slider(minimum=0.5, maximum=5.0, value=2.0, step=0.1, label="Filter Intensity")
            slider_color_picker = gr.ColorPicker(value="#FF0000", label="Slider Color")
            apply_button = gr.Button("Apply Filter")
        
        with gr.Column(scale=2):
            # Output component - ImageSlider
            output_slider = ImageSlider(type="pil", label="Comparison (Slide to compare)", slider_color="#FF0000")
    
    # Set up event handlers
    apply_button.click(
        fn=apply_filters,
        inputs=[input_image, filter_dropdown, intensity_slider],
        outputs=output_slider
    )
    
    slider_color_picker.change(
        fn=update_slider_color,
        inputs=[slider_color_picker],
        outputs=output_slider
    )
    
    # Examples
    example_images = [
        ["https://source.unsplash.com/random/800x600/?nature"],
        ["https://source.unsplash.com/random/800x600/?city"],
        ["https://source.unsplash.com/random/800x600/?portrait"]
    ]
    gr.Examples(
        examples=example_images,
        inputs=input_image
    )
    
    gr.Markdown("## How to use")
    gr.Markdown("""
    1. Upload an image or choose one from the examples
    2. Select a filter type from the dropdown
    3. Adjust the filter intensity using the slider
    4. Click 'Apply Filter' to see the comparison
    5. Use the image slider to compare the original and filtered images
    6. Optionally, change the slider color using the color picker
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()