# helper functions for checking, validating, downloading (if necessary) smolvlm model files

import os
import xxhash
import json
from Y7.colored_print import color
from huggingface_hub import snapshot_download


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
def validate_model_files(model_path, chunk_size=1024 * 1024, config_path=os.path.join(os.path.dirname(__file__), "model_checksums.json")):
    # Validate model files against expected checksums
    # called by check_model_files()

    # files hashed with xxhash.xxh3_64()
    
    # Load file hash data from JSON
    try:
        with open(config_path, 'r') as f:
            model_configs = json.load(f)
    except Exception as e:
        print(f"ⅹ Error loading config file: {str(e)}")
        return False
        
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

    valid = True  # assume everything is fine until proven otherwise

    for file_info in required_files:
        file_path = os.path.join(model_path, file_info["name"])
        
        if not os.path.isfile(file_path):
            print(f"ⅹ Missing file: {file_info['name']}", color.RED)
            valid = False
            continue

        try:
            file_hash = hash_file_partial(file_path, chunk_size=chunk_size)

            if file_hash == file_info["hash"]:    
                # print(f' - {file_info["name"]}: {file_hash}: OKAY', color.BRIGHT_GREEN)
                pass
            else:
                print(f' - {file_info["name"]}: {file_hash}: MISMATCH. Expected {file_info["hash"]}', color.BRIGHT_RED)
                valid = False

        except Exception as e:
            print(f'ⅹ Error checking hash for {file_info["name"]}: {str(e)}', color.RED)
            valid = False

    return valid

# ==============================================================
def check_model_files(model_path):
    """Check if model files exist and are valid"""
    print(f"Checking Model Path: {model_path}", color.ORANGE)

    # validate model files first
    if validate_model_files(model_path):
        print(f"✓ All model files are valid", color.GREEN)
        return True
    
    # If we get here, either directory doesn't exist or files are invalid
    print(f"⚠️ Model files are missing or corrupted - attempting to download", color.YELLOW)
    return False 

# ==============================================================
def download_model_from_HF(model_path):
    """Download model from HuggingFace"""
    # Download model from HF.
    REPO_NAME = f"yushan777/{os.path.basename(model_path)}"

    try:
        print(f"⬇️ Downloading model from HF Repo: {REPO_NAME}", color.ORANGE)
        
        # Download the repository to the specified path
        snapshot_download(
            repo_id=REPO_NAME,
            local_dir=model_path,
            local_dir_use_symlinks=False,  
        )
        
        print(f"✓ Model downloaded successfully", color.GREEN)
        
        # Verify the downloaded files
        if validate_model_files(model_path):
            print(f"✓ Downloaded files validated", color.GREEN)
            return True
        else:
            print(f"ⅹ Downloaded files validation failed", color.RED)
            return False
            
    except Exception as e:
        print(f"ⅹ Failed to download model: {str(e)}", color.RED)
        return False
