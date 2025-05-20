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

:: Download SUPIR-v0Q_fp16.safetensors
set FILE_PATH=SUPIR/SUPIR-v0Q_fp16.safetensors
set TARGET_DIR=%BASE_DIR%\SUPIR
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"
for %%F in ("%FILE_PATH%") do set FILENAME=%%~nxF
if exist "%TARGET_DIR%\%FILENAME%" (
    echo File already exists: %TARGET_DIR%\%FILENAME%
) else (
    echo Downloading: %FILE_PATH% to %TARGET_DIR%
    huggingface-cli download "%REPO_NAME%" "%FILE_PATH%" --local-dir "%DOWNLOADS_DIR%"
    echo Attempting to move file...
    move "%DOWNLOADS_DIR%\%FILE_PATH%" "%TARGET_DIR%\%FILENAME%" 2>nul
    if not exist "%TARGET_DIR%\%FILENAME%" (
        echo Searching for file...
        for /r "%DOWNLOADS_DIR%" %%G in (*%FILENAME%) do (
            echo Found: %%G
            move "%%G" "%TARGET_DIR%\%FILENAME%"
        )
    )
)

:: Download SUPIR-v0F_fp16.safetensors
set FILE_PATH=SUPIR/SUPIR-v0F_fp16.safetensors
set TARGET_DIR=%BASE_DIR%\SUPIR
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"
for %%F in ("%FILE_PATH%") do set FILENAME=%%~nxF
if exist "%TARGET_DIR%\%FILENAME%" (
    echo File already exists: %TARGET_DIR%\%FILENAME%
) else (
    echo Downloading: %FILE_PATH% to %TARGET_DIR%
    huggingface-cli download "%REPO_NAME%" "%FILE_PATH%" --local-dir "%DOWNLOADS_DIR%"
    echo Attempting to move file...
    move "%DOWNLOADS_DIR%\%FILE_PATH%" "%TARGET_DIR%\%FILENAME%" 2>nul
    if not exist "%TARGET_DIR%\%FILENAME%" (
        echo Searching for file...
        for /r "%DOWNLOADS_DIR%" %%G in (*%FILENAME%) do (
            echo Found: %%G
            move "%%G" "%TARGET_DIR%\%FILENAME%"
        )
    )
)

:: Download juggernautXL_v9Rundiffusionphoto2.safetensors
set FILE_PATH=SDXL/juggernautXL_v9Rundiffusionphoto2.safetensors
set TARGET_DIR=%BASE_DIR%\SDXL
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"
for %%F in ("%FILE_PATH%") do set FILENAME=%%~nxF
if exist "%TARGET_DIR%\%FILENAME%" (
    echo File already exists: %TARGET_DIR%\%FILENAME%
) else (
    echo Downloading: %FILE_PATH% to %TARGET_DIR%
    huggingface-cli download "%REPO_NAME%" "%FILE_PATH%" --local-dir "%DOWNLOADS_DIR%"
    echo Attempting to move file...
    move "%DOWNLOADS_DIR%\%FILE_PATH%" "%TARGET_DIR%\%FILENAME%" 2>nul
    if not exist "%TARGET_DIR%\%FILENAME%" (
        echo Searching for file...
        for /r "%DOWNLOADS_DIR%" %%G in (*%FILENAME%) do (
            echo Found: %%G
            move "%%G" "%TARGET_DIR%\%FILENAME%"
        )
    )
)

:: Download clip-vit-large-patch14.safetensors
set FILE_PATH=CLIP1/clip-vit-large-patch14.safetensors
set TARGET_DIR=%BASE_DIR%\CLIP1
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"
for %%F in ("%FILE_PATH%") do set FILENAME=%%~nxF
if exist "%TARGET_DIR%\%FILENAME%" (
    echo File already exists: %TARGET_DIR%\%FILENAME%
) else (
    echo Downloading: %FILE_PATH% to %TARGET_DIR%
    huggingface-cli download "%REPO_NAME%" "%FILE_PATH%" --local-dir "%DOWNLOADS_DIR%"
    echo Attempting to move file...
    move "%DOWNLOADS_DIR%\%FILE_PATH%" "%TARGET_DIR%\%FILENAME%" 2>nul
    if not exist "%TARGET_DIR%\%FILENAME%" (
        echo Searching for file...
        for /r "%DOWNLOADS_DIR%" %%G in (*%FILENAME%) do (
            echo Found: %%G
            move "%%G" "%TARGET_DIR%\%FILENAME%"
        )
    )
)

:: Download CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors
set FILE_PATH=CLIP2/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors
set TARGET_DIR=%BASE_DIR%\CLIP2
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"
for %%F in ("%FILE_PATH%") do set FILENAME=%%~nxF
if exist "%TARGET_DIR%\%FILENAME%" (
    echo File already exists: %TARGET_DIR%\%FILENAME%
) else (
    echo Downloading: %FILE_PATH% to %TARGET_DIR%
    huggingface-cli download "%REPO_NAME%" "%FILE_PATH%" --local-dir "%DOWNLOADS_DIR%"
    echo Attempting to move file...
    move "%DOWNLOADS_DIR%\%FILE_PATH%" "%TARGET_DIR%\%FILENAME%" 2>nul
    if not exist "%TARGET_DIR%\%FILENAME%" (
        echo Searching for file...
        for /r "%DOWNLOADS_DIR%" %%G in (*%FILENAME%) do (
            echo Found: %%G
            move "%%G" "%TARGET_DIR%\%FILENAME%"
        )
    )
)

:: Clean up temp directory if it's empty
dir /a-d "%DOWNLOADS_DIR%\*.*" >nul 2>&1
if errorlevel 1 (
    rmdir "%DOWNLOADS_DIR%" 2>nul
)

echo All models checked/downloaded successfully!