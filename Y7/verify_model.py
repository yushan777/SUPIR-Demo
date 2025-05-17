# helper functions for checking, validating, downloading (if necessary) smolvlm model files

import os
import xxhash
import json
from Y7.colored_print import color



# ==============================================================
# def hash_file_full(filepath, chunk_size=1024 * 1024):  # default 1MB chunk
#     """
#     Hash a file using xxhash for checksumming
#     """
#     h = xxhash.xxh3_64()
#     with open(filepath, 'rb') as f:
#         for chunk in iter(lambda: f.read(chunk_size), b''):
#             h.update(chunk)
#     return h.hexdigest()

# ==============================================================
def hash_file_partial(filepath, chunk_size=1024 * 1024, max_chunks=25):
    
    # compute a partial xxHash (xxh3_64) of a file for faster hashing of large files.
    # reads and hashes only the first few chunks of a file (by default, the first 25 MB).
    # useful when you want a fast checksum that doesn't require hashing the entire file.
    # if the file is smaller than (chunk_size * max_chunks), then the result is effectively a full hash.
    # Parameters:
    #     filepath (str): Path to the file to be hashed.
    #     chunk_size (int): Size (in bytes) of each chunk to read from the file. Default is 1MB.
    #     max_chunks (int): Maximum number of chunks to read and hash. Default is 25.
    # Returns:
    #     str: The resulting hexadecimal hash string from the partial file content.
    
    h = xxhash.xxh3_64()
    chunks_processed = 0
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
            chunks_processed += 1
            if chunks_processed >= max_chunks:
                break
    
    return h.hexdigest()

# ==========================================================================
def validate_all_model_files(model_path):
    # Validate all model files against expected checksums
    # all must be valid or exist
    # called by check_model_files()

    # files hashed with xxhash.xxh3_64()

    chunk_size=1024 * 1024
    config_path=os.path.join(os.path.dirname(__file__), "model_checksums.json")

    print(f"Checking files in {model_path}", color.ORANGE)

    # Load file hash data from JSON
    try:
        with open(config_path, 'r') as f:
            model_configs = json.load(f)
    except Exception as e:
        print(f"ⅹ Error loading config file: {str(e)}")
        return ["Error loading config file"]
        
    # Determine which model we're validating
    model_type = None
    for possible_type in model_configs.keys():
        if possible_type in model_path:
            model_type = possible_type
            break
            
    if not model_type:
        print(f"ⅹ Unknown model type: {model_path}")
        return ["Unknown model type"]
        
    required_files = model_configs[model_type]      

    missing_or_invalid_files = []

    for file_info in required_files:
        file_path = os.path.join(model_path, file_info["name"])
        
        if not os.path.isfile(file_path):
            print(f"ⅹ Missing file: {file_info['name']}", color.RED)
            missing_or_invalid_files.append(file_info['name'])
            continue

        try:
            file_hash = hash_file_partial(file_path, chunk_size=chunk_size)

            if file_hash == file_info["hash"]:    
                # print(f' - {file_info["name"]}: {file_hash}: OKAY', color.BRIGHT_GREEN)
                pass
            else:
                print(f' - {file_info["name"]}: {file_hash}: MISMATCH. Expected {file_info["hash"]}', color.BRIGHT_RED)
                missing_or_invalid_files.append(file_info['name'])

        except Exception as e:
            print(f'ⅹ Error checking hash for {file_info["name"]}: {str(e)}', color.RED)
            missing_or_invalid_files.append(file_info['name'])

    return missing_or_invalid_files

# ======================================================================================
def validate_any_model_files(model_path):
    # Validate any model files found against expected checksums
    # Function succeeds if ANY valid file exists
    # Only reports a problem if NO files exist
    # called by check_model_files()

    # files hashed with xxhash.xxh3_64()

    chunk_size=1024 * 1024
    config_path=os.path.join(os.path.dirname(__file__), "model_checksums.json")

    print(f"Checking files in {model_path}",color.ORANGE)

    # Load file hash data from JSON
    try:
        with open(config_path, 'r') as f:
            model_configs = json.load(f)
    except Exception as e:
        print(f"ⅹ Error loading config file: {str(e)}")
        return ["Error loading config file"]
        
    # Determine which model we're validating
    model_type = None
    for possible_type in model_configs.keys():
        if possible_type in model_path:
            model_type = possible_type
            break
            
    if not model_type:
        print(f"ⅹ Unknown model type: {model_path}")
        return False
        
    required_files = model_configs[model_type]      

    files_exist = False
    valid_files = []
    invalid_files = []

    for file_info in required_files:
        file_path = os.path.join(model_path, file_info["name"])
        
        if os.path.isfile(file_path):
            files_exist = True
            
            try:
                file_hash = hash_file_partial(file_path, chunk_size=chunk_size)

                if file_hash == file_info["hash"]:    
                    # print(f' - {file_info["name"]}: {file_hash}: OKAY', color.BRIGHT_GREEN)
                    valid_files.append(file_info['name'])
                else:
                    # print(f' - {file_info["name"]}: {file_hash}: MISMATCH. Expected {file_info["hash"]}', color.BRIGHT_RED)
                    invalid_files.append(file_info['name'])

            except Exception as e:
                # print(f'ⅹ Error checking hash for {file_info["name"]}: {str(e)}', color.RED)
                invalid_files.append(file_info['name'])
        else:
            # File doesn't exist - just skip it
            pass

    # Only report a problem if NO files exist
    if not files_exist:
        print(f"ⅹ No model files found in {model_path}", color.RED)
        return False
    else:
        # Success if any files exist, regardless of validity
        if valid_files:
            print(f" - Found {len(valid_files)} valid model files", color.GREEN)
            # Print all valid files
            for file_name in valid_files:
                print(f"   └── {file_name}", color.BRIGHT_GREEN)

        if invalid_files:
            print(f" - Found {len(invalid_files)} invalid model files", color.RED)
            # Print all valid files
            for file_name in invalid_files:
                print(f"   └── {file_name}", color.BRIGHT_RED)   

        # we have some files that exist so return true 
        return True
    
# ==============================================================
def check_smolvlm_model_files(model_path):
    """Check if ALL model files exist and are valid"""
    

    # validate model files first
    missing_or_invalid_files = validate_all_model_files(model_path)
    if not missing_or_invalid_files:
        print(f"✓ All model files are valid", color.GREEN)
        return True
    
    # If we get here, either directory doesn't exist or files are invalid
    print(f"⚠️ Model files are missing or corrupted - Try downloading them again either manually or with the download script.", color.YELLOW)
    print(f"Missing files: {missing_or_invalid_files}", color.RED)
    return False

# ==============================================================
def check_supir_model_files(model_path):
    # validate model files first
    anyfound = validate_any_model_files(model_path)
    if anyfound:
        print(f"✓ All available SUPIR model files are valid", color.GREEN)
        return True
    
    # If we get here, either directory doesn't exist or files are invalid
    print(f"⚠️ SUPIR Model files are missing or corrupted - Try downloading them again either manually or with the download script.", color.YELLOW)
    return False

# ==============================================================
def check_clip_model_file(model_path):
    # validate model files first
    anyfound = validate_any_model_files(model_path)
    if anyfound:
        print(f"✓ CLIP1 model valid", color.GREEN)
        return True
    
    # If we get here, either directory doesn't exist or files are invalid
    print(f"⚠️ CLIP1 model missing or corrupt - Try downloading again either manually or with the download script.", color.YELLOW)
    return False

def check_for_any_sdxl_model(model_path):
    # Success if ANY .safetensors files exist

    print(f"Checking files in {model_path}", color.ORANGE)
    # Get all files in the directory
    try:
        all_files = os.listdir(model_path)
    except Exception as e:
        print(f"ⅹ Error accessing model directory: {str(e)}", color.RED)
        return False

    # Filter for .safetensors files
    safetensor_files = [f for f in all_files if f.endswith('.safetensors')]
    
    if not safetensor_files:
        print(f"ⅹ No safetensors files found in {model_path}", color.RED)
        return False
    else:
        print(f" - Found {len(safetensor_files)} safetensors files", color.BRIGHT_GREEN)
        # Print all .safetensors files
        for file_name in safetensor_files:
            print(f"   └── {file_name}", color.GREEN)
        
        return True
# ==============================================================
# def download_smolvlm_model_from_HF(model_path):
#     """Download model from HuggingFace"""
#     # Download model from HF.
#     REPO_NAME = f"yushan777/{os.path.basename(model_path)}"

#     try:
#         print(f"⬇️ Downloading model from HF Repo: {REPO_NAME}", color.ORANGE)
        
#         # Download the repository to the specified path
#         snapshot_download(
#             repo_id=REPO_NAME,
#             local_dir=model_path,
#             local_dir_use_symlinks=False,  
#         )
        
#         print(f"✓ Model downloaded successfully", color.GREEN)
        
#         # Verify the downloaded files
#         if validate_model_files(model_path):
#             print(f"✓ Downloaded files validated", color.GREEN)
#             return True
#         else:
#             print(f"ⅹ Downloaded files validation failed", color.RED)
#             return False
            
#     except Exception as e:
#         print(f"ⅹ Failed to download model: {str(e)}", color.RED)
#         return False
