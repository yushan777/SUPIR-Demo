import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.nn.functional import interpolate
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config
from Y7.colored_print import color, style

def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from {ckpt_path}', color.BRIGHT_BLUE)
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from {config_path}', color.BRIGHT_BLUE)
    return model


def create_SUPIR_model(config_path, SUPIR_sign=None):

    # Creates a SUPIR model instance by blending SDXL and SUPIR capabilities
    # 
    # This function instantiates a model using the provided config, then loads weights in layers:
    # 1. Base SDXL foundation weights (if specified in config)
    # 2. General SUPIR enhancement weights (if specified in config)
    # 3. Optional specialized SUPIR variant (F or Q) that selectively overrides weights
    #    for specific performance characteristics
    #
    # Args:
    #   config_path: Path to the YAML config file (options/SUPIR_v0_tiled.yaml or options/SUPIR_v0.yaml)
    #   SUPIR_sign: Optional variant specifier ('F' or 'Q') to load specialized weights
    #
    # Returns:
    #   The instantiated and weight-loaded SUPIR model

    config = OmegaConf.load(config_path)

    # instantiate model using the config loaded froom the yaml
    model = instantiate_from_config(config.model).cpu()

    print(f"Instantiated model using config from {config_path}", color.BRIGHT_BLUE)

    # Load checkpoints using paths from config, checking existence
    # first the SDXL model
    if hasattr(config, "SDXL_CKPT") and config.SDXL_CKPT is not None:
        print(f"Loading SDXL checkpoint: {config.SDXL_CKPT}", color.BRIGHT_BLUE)
        model.load_state_dict(load_state_dict(config.SDXL_CKPT), strict=False)

    # load supir model according to Q of F sign
    if SUPIR_sign is not None:
        assert SUPIR_sign in ["F", "Q"]
        ckpt_key = f"SUPIR_CKPT_{SUPIR_sign}"
        if hasattr(config, ckpt_key) and getattr(config, ckpt_key) is not None:
            ckpt_path = getattr(config, ckpt_key)
            print(f"Loading SUPIR {SUPIR_sign} checkpoint: {ckpt_path}", color.BRIGHT_BLUE)
            model.load_state_dict(load_state_dict(ckpt_path), strict=False)
        else:
            print(f"Warning: SUPIR sign '{SUPIR_sign}' provided, but checkpoint path '{ckpt_key}' not found or is None in config.", color.RED)

    return model

def PIL2Tensor(img, upscale=1, min_size=1024, fix_resize=None):
    '''
    PIL.Image -> Tensor[C, H, W], RGB, [-1, 1]

    # Converts a PIL image to a normalized PyTorch tensor (C, H, W) in [-1, 1] range.
    # Optionally upscales the image, ensures a minimum size (for sdxl processing), and rounds dimensions to multiples of 64.
    # Returns the tensor (x) and original post-upscale dimensions (before final resizing).
    # Input image size: 256x256, upscale: 2, min_size: 1024
    # x.shape  = (3, 1024, 1024)
    # h0 = 512
    # w0 = 512
    '''
    # Get the orig width and height 
    w, h = img.size
    
    # Apply upscale factor 
    w *= upscale
    h *= upscale
    
    # Store the dimensions after upscale as w0, h0 (rounded)
    w0, h0 = round(w), round(h)
    
    # If the smallest dimension is less than min_size, scale up both dimensions
    # to make the smallest dimension equal to min_size
    if min(w, h) < min_size:
        _upscale = min_size / min(w, h)  # Calculate the scaling factor needed
        w *= _upscale  # Scale the width
        h *= _upscale  # Scale the height
        # Note: w0 and h0 are NOT updated here, so they retain their original values
    
    # If fix_resize is provided, scale the image to make the smallest dimension
    # equal to fix_resize value
    if fix_resize is not None:
        _upscale = fix_resize / min(w, h)  # Calculate scaling factor for fixed resize
        w *= _upscale  # Scale the width
        h *= _upscale  # Scale the height
        w0, h0 = round(w), round(h)  # Update w0, h0 ONLY in this case
    
    # Round width and height to nearest multiple of 64 
    # (important for many neural network architectures)
    w = int(np.round(w / 64.0)) * 64
    h = int(np.round(h / 64.0)) * 64
    
    # Resize the image to the calculated dimensions using bicubic interp
    x = img.resize((w, h), Image.BICUBIC)
    
    # Convert PIL image to numpy array, ensure pixel values are integers 
    # between 0-255, and set data type to uint8
    x = np.array(x).round().clip(0, 255).astype(np.uint8)
    
    # Normalize pixel values from [0, 255] to [-1, 1] range
    # (common normalization for neural network inputs)
    x = x / 255 * 2 - 1
    
    # Convert numpy array to PyTorch tensor and rearrange dimensions
    # from (H, W, C) to (C, H, W) format as needed by PyTorch
    x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
    
    # Return the tensor and the original dimensions (which may or may not have been updated)
    return x, h0, w0

def Tensor2PIL(x, h0, w0):
    '''
    Tensor[C, H, W], RGB, [-1, 1] -> PIL.Image
    '''

    # Check for invalid values
    if torch.isnan(x).any() or torch.isinf(x).any():
        print(f">>>>> WARNING: Tensor contains NaN or Inf values. Min: {x.min().item()}, Max: {x.max().item()}")
        # Replace NaN/Inf with valid values
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

    x = x.unsqueeze(0)
    x = interpolate(x, size=(h0, w0), mode='bicubic')
    x = (x.squeeze(0).permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def upscale_image(input_image, upscale, min_size=None, unit_resolution=64):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    H *= upscale
    W *= upscale
    if min_size is not None:
        if min(H, W) < min_size:
            _upscale = min_size / min(W, H)
            W *= _upscale
            H *= _upscale
    H = int(np.round(H / unit_resolution)) * unit_resolution
    W = int(np.round(W / unit_resolution)) * unit_resolution
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if upscale > 1 else cv2.INTER_AREA)
    img = img.round().clip(0, 255).astype(np.uint8)
    return img


def fix_resize(input_image, size=512, unit_resolution=64):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    upscale = size / min(H, W)
    H *= upscale
    W *= upscale
    H = int(np.round(H / unit_resolution)) * unit_resolution
    W = int(np.round(W / unit_resolution)) * unit_resolution
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if upscale > 1 else cv2.INTER_AREA)
    img = img.round().clip(0, 255).astype(np.uint8)
    return img



def Numpy2Tensor(img):
    '''
    np.array[H, w, C] [0, 255] -> Tensor[C, H, W], RGB, [-1, 1]
    '''
    # size
    img = np.array(img) / 255 * 2 - 1
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    return img


def Tensor2Numpy(x, h0=None, w0=None):
    '''
    Tensor[C, H, W], RGB, [-1, 1] -> PIL.Image
    '''
    # Check for invalid values
    if torch.isnan(x).any() or torch.isinf(x).any():
        print(f">>>>> WARNING: Tensor contains NaN or Inf values. Min: {x.min().item()}, Max: {x.max().item()}")
        # Replace NaN/Inf with valid values (0 in this case)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

    if h0 is not None and w0 is not None:
        x = x.unsqueeze(0)
        x = interpolate(x, size=(h0, w0), mode='bicubic')
        x = x.squeeze(0)
    x = (x.permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    return x


def convert_dtype(dtype_str):
    if dtype_str == 'fp32':
        return torch.float32
    elif dtype_str == 'fp16':
        return torch.float16
    elif dtype_str == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError
