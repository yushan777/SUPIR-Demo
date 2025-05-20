@echo off
REM Install Hugging Face CLI and related packages
pip install -U "huggingface_hub[cli]"
pip install "huggingface_hub[hf_transfer]"
pip install hf_transfer

REM Ask user about enabling high-speed transfers
set /p enable_high_speed=Do you want to enable high-speed transfers? For connections higher than 1Gbps (y/n): 
if /I "%enable_high_speed%"=="y" (
    echo Enabling high-speed transfers...
    setx HF_HUB_ENABLE_HF_TRANSFER 1
    echo High-speed transfers enabled!
) else (
    echo Using standard transfer speeds.
)

REM Optional login
REM echo Please note: You'll need to be logged in to download these models.
REM echo If not logged in already, uncomment and use the following line with your token:
REM echo huggingface-cli login --token YOUR_HF_TOKEN

set REPO_NAME=yushan777/SmolVLM-500M-Instruct

REM Download models
huggingface-cli download %REPO_NAME% --local-dir "models\SmolVLM-500M-Instruct"


set REPO_NAME=yushan777/SUPIR
echo Downloading models from %REPO_NAME%...
huggingface-cli download %REPO_NAME% SUPIR/SUPIR-v0Q_fp16.safetensors --local-dir downloads
huggingface-cli download %REPO_NAME% SUPIR/SUPIR-v0F_fp16.safetensors --local-dir downloads
huggingface-cli download %REPO_NAME% SDXL/juggernautXL_v9Rundiffusionphoto2.safetensors --local-dir downloads
huggingface-cli download %REPO_NAME% CLIP1/clip-vit-large-patch14.safetensors --local-dir downloads
huggingface-cli download %REPO_NAME% CLIP2/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors --local-dir downloads


REM Move files to appropriate directories
echo Moving downloaded models to their respective directories...
move downloads\SUPIR\SUPIR-v0Q_fp16.safetensors models\SUPIR\
move downloads\SUPIR\SUPIR-v0F_fp16.safetensors models\SUPIR\
move downloads\SDXL\juggernautXL_v9Rundiffusionphoto2.safetensors models\SDXL\
move downloads\CLIP1\clip-vit-large-patch14.safetensors models\CLIP1\
move downloads\CLIP2\CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors models\CLIP2\

echo All models downloaded and moved successfully!