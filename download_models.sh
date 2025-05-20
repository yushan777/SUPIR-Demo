#!/bin/bash

# Set up error handling
set -e  # Exit immediately if a command exits with non-zero status

# Install Hugging Face CLI and related packages
echo "Installing/updating required packages..."
pip install -q -U "huggingface_hub[cli]"
pip install -q huggingface_hub[hf_transfer]
pip install -q hf_transfer

# Base directory for all models
BASE_DIR="models"
DOWNLOADS_DIR="downloads"
mkdir -p "$BASE_DIR" "$DOWNLOADS_DIR"


# ===========================================
# download if needed
download_model() {
    local repo="$1"
    local file_path="$2"
    local target_dir="$3"
    
    local filename=$(basename "$file_path")
    local target_file="$target_dir/$filename"
    
    # Create target directory if it doesn't exist
    mkdir -p "$target_dir"
    
    if [ -f "$target_file" ]; then
        echo "File already exists: $target_file"
    else
        echo "Downloading: $file_path to $target_dir"
        huggingface-cli download "$repo" "$file_path" --local-dir "$DOWNLOADS_DIR"
        # Move from temp download location to final location
        mkdir -p "$(dirname "$target_dir/$filename")"
        mv "$DOWNLOADS_DIR/$file_path" "$target_dir/$filename"
    fi
}

# ===========================================
# Ask user about enabling high-speed transfers
read -p "Enable high-speed transfers? Better for fast connections (y/n): " enable
if [[ $enable == "y" || $enable == "Y" ]]; then
    echo "Enabling high-speed transfers..."
    export HF_HUB_ENABLE_HF_TRANSFER=1
    echo "High-speed transfers enabled!"
else
    echo "Using standard transfer speeds."
fi

# SmolVLM model
REPO_NAME="yushan777/SmolVLM-500M-Instruct"
TARGET_DIR="$BASE_DIR/SmolVLM-500M-Instruct"

# Check if entire SmolVLM directory exists and has files
if [ -d "$TARGET_DIR" ] && [ "$(ls -A "$TARGET_DIR" 2>/dev/null)" ]; then
    echo "✓ SmolVLM-500M-Instruct model already exists in $TARGET_DIR"
else
    echo "↓ Downloading complete SmolVLM-500M-Instruct repository..."
    mkdir -p "$TARGET_DIR"
    huggingface-cli download $REPO_NAME --local-dir "$TARGET_DIR"
fi

# SUPIR models - individual files
REPO_NAME="yushan777/SUPIR"
echo "Checking SUPIR models..."
download_model "$REPO_NAME" "SUPIR/SUPIR-v0Q_fp16.safetensors" "$BASE_DIR/SUPIR"
download_model "$REPO_NAME" "SUPIR/SUPIR-v0F_fp16.safetensors" "$BASE_DIR/SUPIR"
download_model "$REPO_NAME" "SDXL/juggernautXL_v9Rundiffusionphoto2.safetensors" "$BASE_DIR/SDXL"
download_model "$REPO_NAME" "CLIP1/clip-vit-large-patch14.safetensors" "$BASE_DIR/CLIP1"
download_model "$REPO_NAME" "CLIP2/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors" "$BASE_DIR/CLIP2"

# Clean up temp directory if it's empty
if [ -z "$(ls -A "$DOWNLOADS_DIR" 2>/dev/null)" ]; then
    rmdir "$DOWNLOADS_DIR"
fi

echo "All models checked/downloaded successfully!"