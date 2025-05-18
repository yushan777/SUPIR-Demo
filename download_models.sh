#!/bin/bash

# Install Hugging Face CLI and related packages
pip install -U "huggingface_hub[cli]"
pip install huggingface_hub[hf_transfer]
pip install hf_transfer

# Ask user about enabling high-speed transfers
read -p "Do you want to enable high-speed transfers? Better for fast connections (y/n): " enable_high_speed
if [[ $enable_high_speed == "y" || $enable_high_speed == "Y" ]]; then
    echo "Enabling high-speed transfers..."
    export HF_HUB_ENABLE_HF_TRANSFER=1
    echo "High-speed transfers enabled!"
else
    echo "Using standard transfer speeds."
fi

# # Log in to Hugging Face CLI with your token
# echo "Please note: You'll need to be logged in to download these models."
# echo "If not logged in already, uncomment and use the following line with your token:"
# echo "# huggingface-cli login --token \$HF_TOKEN"

REPO_NAME="yushan777/-Instruct"

# Download models
# ====================================
# smolvlm-instruct - full repo directly to models/SmolVLM-Instruct
huggingface-cli download $REPO_NAME --local-dir 'models/SmolVLM-Instruct'

# ====================================
REPO_NAME="yushan777/SUPIR"
# SUPIR - individual files
echo "Downloading models from $REPO_NAME..."
huggingface-cli download $REPO_NAME SUPIR/SUPIR-v0Q_fp16.safetensors --local-dir downloads
huggingface-cli download $REPO_NAME SUPIR/SUPIR-v0F_fp16.safetensors --local-dir downloads
huggingface-cli download $REPO_NAME SDXL/juggernautXL_v9Rundiffusionphoto2.safetensors --local-dir downloads
huggingface-cli download $REPO_NAME CLIP1/clip-vit-large-patch14/safetensors/clip-vit-large-patch14.safetensors --local-dir downloads
huggingface-cli download $REPO_NAME CLIP2/CLIP-ViT-bigG-14-laion2B-39B-b160k/safetensors/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors --local-dir downloads

# Move the models
echo "Moving downloaded models to their respective directories..."
mv downloads/SUPIR/SUPIR-v0Q_fp16.safetensors models/SUPIR/
mv downloads/SUPIR/SUPIR-v0F_fp16.safetensors models/SUPIR/
mv downloads/SDXL/juggernautXL_v9Rundiffusionphoto2.safetensors models/SDXL/
mv downloads/CLIP1/clip-vit-large-patch14.safetensors models/CLIP1/
mv downloads/CLIP2/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors models/CLIP2/

echo "All models downloaded and moved successfully!"