@echo off
setlocal enabledelayedexpansion

:: Set up error handling
echo Installing/updating required packages...
pip install -q -U "huggingface_hub[cli]"
pip install -q huggingface_hub[hf_transfer]
pip install -q hf_transfer

:: Base directory for all models
set BASE_DIR=models
set DOWNLOADS_DIR=downloads_temp
if not exist "%BASE_DIR%" mkdir "%BASE_DIR%"
if not exist "%DOWNLOADS_DIR%" mkdir "%DOWNLOADS_DIR%"

goto :main

@REM ============================================================
:download_model
    set repo=%~1
    set file_path=%~2
    set target_dir=%~3
    
    for %%F in ("%file_path%") do set filename=%%~nxF
    set target_file=%target_dir%\%filename%
    
    :: Create target directory if it doesn't exist
    if not exist "%target_dir%" mkdir "%target_dir%"
    
    if exist "%target_file%" (
        echo File already exists: %target_file%
    ) else (
        echo Downloading: %file_path% to %target_dir%
        huggingface-cli download "%repo%" "%file_path%" --local-dir "%DOWNLOADS_DIR%"
        
        :: Create all parent directories for the target file
        for %%F in ("%target_dir%\%filename%") do if not exist "%%~dpF" mkdir "%%~dpF"
        
        :: Debug output to see what's happening
        echo Source: %DOWNLOADS_DIR%\%file_path%
        echo Target: %target_dir%\%filename%
        
        :: Check if source file exists
        if exist "%DOWNLOADS_DIR%\%file_path%" (
            move "%DOWNLOADS_DIR%\%file_path%" "%target_dir%\%filename%"
        ) else (
            echo WARNING: Source file does not exist: %DOWNLOADS_DIR%\%file_path%
            
            :: Try to find the file in a different location
            for /r "%DOWNLOADS_DIR%" %%G in (*%filename%) do (
                echo Found file: %%G
                move "%%G" "%target_dir%\%filename%"
                goto :found_file
            )
            echo ERROR: Could not find downloaded file %filename%
            :found_file
        )
    )
    goto :eof

@REM ============================================================
:main
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

:: Check if entire SmolVLM directory exists and has files
dir /a-d "%TARGET_DIR%\*.*" >nul 2>&1
if not errorlevel 1 (
    echo ✓ SmolVLM-500M-Instruct model already exists in %TARGET_DIR%
) else (
    echo ↓ Downloading complete SmolVLM-500M-Instruct repository...
    if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"
    huggingface-cli download %REPO_NAME% --local-dir "%TARGET_DIR%"
)

:: SUPIR models - individual files
set REPO_NAME=yushan777/SUPIR
echo Checking SUPIR models...
call :download_model "%REPO_NAME%" "SUPIR/SUPIR-v0Q_fp16.safetensors" "%BASE_DIR%\SUPIR"
call :download_model "%REPO_NAME%" "SUPIR/SUPIR-v0F_fp16.safetensors" "%BASE_DIR%\SUPIR"
call :download_model "%REPO_NAME%" "SDXL/juggernautXL_v9Rundiffusionphoto2.safetensors" "%BASE_DIR%\SDXL"
call :download_model "%REPO_NAME%" "CLIP1/clip-vit-large-patch14.safetensors" "%BASE_DIR%\CLIP1"
call :download_model "%REPO_NAME%" "CLIP2/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors" "%BASE_DIR%\CLIP2"

:: Clean up temp directory if it's empty
dir /a-d "%DOWNLOADS_DIR%\*.*" >nul 2>&1
if errorlevel 1 (
    rmdir "%DOWNLOADS_DIR%" 2>nul
)

echo All models checked/downloaded successfully!