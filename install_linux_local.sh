#!/bin/bash

# Prompt for HF token
# read -sp "Enter your Hugging Face access token: " HF_TOKEN
# echo

# install deps

python3 -m venv venv
source venv/bin/activate

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu126
# pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# # Install Hugging Face CLI and related packages
# pip install -U "huggingface_hub[cli]"
# pip install huggingface_hub[hf_transfer]
# pip install hf_transfer

# # Log in to Hugging Face CLI with your token
# # huggingface-cli login --token $HF_TOKEN
# export HF_HUB_ENABLE_HF_TRANSFER=1
# REPO_NAME="yushan777/SUPIR"

# # download models
# huggingface-cli download $REPO_NAME SUPIR/SUPIR-v0Q_fp32.safetensors --local-dir downloads
# huggingface-cli download $REPO_NAME SUPIR/SUPIR-v0F_fp32.safetensors --local-dir downloads
# huggingface-cli download $REPO_NAME SDXL/juggernautXL_v9Rundiffusionphoto2.safetensors --local-dir downloads
# huggingface-cli download $REPO_NAME CLIP1/clip-vit-large-patch14/safetensors/clip-vit-large-patch14.safetensors --local-dir downloads
# huggingface-cli download $REPO_NAME CLIP2/CLIP-ViT-bigG-14-laion2B-39B-b160k/safetensors/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors --local-dir downloads

# # move the models
# mv downloads/SUPIR/SUPIR-v0Q_fp16.safetensors models/SUPIR/
# mv downloads/SUPIR/SUPIR-v0F_fp16.safetensors models/SUPIR/
# mv downloads/SDXL/juggernautXL_v9Rundiffusionphoto2.safetensors models/SDXL
# mv downloads/CLIP1/clip-vit-large-patch14.safetensors models/CLIP1
# mv downloads/CLIP2/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors models/CLIP2

