@echo off
setlocal enabledelayedexpansion
title Model Setup Script

:: Set up error handling (batch equivalent)
echo Installing/updating required packages...
pip install -q -U "huggingface_hub[cli]"
pip install -q huggingface_hub[hf_transfer]
pip install -q hf_transfer

:: Check that huggingface-cli is available
where huggingface-cli >nul 2>&1
if errorlevel 1 (
    echo ❌ huggingface-cli not found. Please ensure it's installed and in your PATH.
    exit /b 1
)

:: Base directory for all models
set BASE_DIR=models
set DOWNLOADS_DIR=downloads
if not exist "%BASE_DIR%" mkdir "%BASE_DIR%"
if not exist "%DOWNLOADS_DIR%" mkdir "%DOWNLOADS_DIR%"

:: Ask user about enabling high-speed transfers
set /p enable=Enable high-speed transfers? Better for fast connections (y/n): 
if /i "%enable%"=="y" (
    echo Enabling high-speed transfers...
    set HF_HUB_ENABLE_HF_TRANSFER=1
    echo High-speed transfers enabled!
) else (
    echo Using standard transfer speeds.
)

:: SmolVLM model
set REPO_NAME=yushan777/SmolVLM-500M-Instruct
set TARGET_DIR=%BASE_DIR%\SmolVLM-500M-Instruct

:: Required files for SmolVLM
set REQUIRED_FILES=added_tokens.json chat_template.json config.json generation_config.json merges.txt model.safetensors preprocessor_config.json processor_config.json special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json

:: Check if all required SmolVLM files exist
set all_files_exist=true
for %%f in (%REQUIRED_FILES%) do (
    if not exist "%TARGET_DIR%\%%f" (
        set all_files_exist=false
        echo Missing required file: %%f
    )
)

if "!all_files_exist!"=="true" (
    echo ✓ SmolVLM-500M-Instruct model already exists with all required files in %TARGET_DIR%
) else (
    echo ↓ Downloading complete SmolVLM-500M-Instruct repository...
    mkdir "%TARGET_DIR%" 2>nul
    huggingface-cli download %REPO_NAME% --local-dir "%TARGET_DIR%"
)

:: Download SUPIR models - individual files
set REPO_NAME=yushan777/SUPIR
echo Checking SUPIR models...

call :download_model "%REPO_NAME%" "SUPIR/SUPIR-v0Q_fp16.safetensors" "%BASE_DIR%\SUPIR"
call :download_model "%REPO_NAME%" "SUPIR/SUPIR-v0F_fp16.safetensors" "%BASE_DIR%\SUPIR"
call :download_model "%REPO_NAME%" "SDXL/juggernautXL_v9Rundiffusionphoto2.safetensors" "%BASE_DIR%\SDXL"
call :download_model "%REPO_NAME%" "CLIP1/clip-vit-large-patch14.safetensors" "%BASE_DIR%\CLIP1"
call :download_model "%REPO_NAME%" "CLIP2/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors" "%BASE_DIR%\CLIP2"

:: Clean up temp directory if it's empty
dir /b /a "%DOWNLOADS_DIR%" | findstr . >nul
if errorlevel 1 rmdir "%DOWNLOADS_DIR%"

echo.
echo ✅ All models checked/downloaded successfully!
pause
exit /b

:: ------------------------------
:: Download model subroutine
:: ------------------------------
:download_model
set repo=%~1
set file_path=%~2
set target_dir=%~3

set filename=%~nx2
set target_file=%target_dir%\%filename%

:: Create target directory if it doesn't exist
if not exist "%target_dir%" mkdir "%target_dir%"

if exist "%target_file%" (
    echo File already exists: %target_file%
) else (
    echo Downloading: %file_path% to %target_dir%
    huggingface-cli download "%repo%" "%file_path%" --local-dir "%DOWNLOADS_DIR%"
    :: Ensure intermediate folder exists (for nested files)
    for %%i in ("%target_file%") do if not exist "%%~dpi" mkdir "%%~dpi"
    :: Move file from temp download location to target
    move "%DOWNLOADS_DIR%\%file_path%" "%target_file%" >nul
)
goto :eof
