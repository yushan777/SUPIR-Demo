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
    config = OmegaConf.load(config_path)

    # --- Inject CLIP paths from root config into embedder params ---
    if hasattr(config, "CLIP1_PATH") or hasattr(config, "CLIP2_PATH"):
        conditioner_params = config.model.params.conditioner_config.params
        for embedder_config in conditioner_params.get("emb_models", []):
            # Ensure params exists as a mutable OmegaConf object if not present
            if "params" not in embedder_config:
                 embedder_config.params = OmegaConf.create()
            # Inject paths
            if embedder_config.target.endswith("FrozenCLIPEmbedder") and hasattr(config, "CLIP1_PATH"):
                embedder_config.params.clip1_path = config.CLIP1_PATH
                # print(f"  Injected CLIP1_PATH into {embedder_config.target}")
            elif embedder_config.target.endswith("FrozenOpenCLIPEmbedder2") and hasattr(config, "CLIP2_PATH"):
                embedder_config.params.clip2_path = config.CLIP2_PATH
                # print(f"  Injected CLIP2_PATH into {embedder_config.target}")

    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from {config_path}', color.BRIGHT_BLUE)

    # --- Load checkpoints using paths from config, checking existence ---
    if hasattr(config, "SDXL_CKPT") and config.SDXL_CKPT is not None:
        print(f"Loading SDXL checkpoint: {config.SDXL_CKPT}", color.BRIGHT_BLUE)
        model.load_state_dict(load_state_dict(config.SDXL_CKPT), strict=False)
    if hasattr(config, "SUPIR_CKPT") and config.SUPIR_CKPT is not None:
        print(f"Loading SUPIR base checkpoint: {config.SUPIR_CKPT}", color.BRIGHT_BLUE)
        model.load_state_dict(load_state_dict(config.SUPIR_CKPT), strict=False)

    if SUPIR_sign is not None:
        assert SUPIR_sign in ['F', 'Q']
        ckpt_key = f"SUPIR_CKPT_{SUPIR_sign}"
        if hasattr(config, ckpt_key) and getattr(config, ckpt_key) is not None:
            ckpt_path = getattr(config, ckpt_key)
            print(f"Loading SUPIR {SUPIR_sign} checkpoint: {ckpt_path}", color.BRIGHT_BLUE)
            model.load_state_dict(load_state_dict(ckpt_path), strict=False)
        else:
            print(f"Warning: SUPIR sign '{SUPIR_sign}' provided, but checkpoint path '{ckpt_key}' not found or is None in config.", color.RED)

    return model

def load_QF_ckpt(config_path):
    config = OmegaConf.load(config_path)
    ckpt_F = torch.load(config.SUPIR_CKPT_F, map_location='cpu')
    ckpt_Q = torch.load(config.SUPIR_CKPT_Q, map_location='cpu')
    return ckpt_Q, ckpt_F


def PIL2Tensor(img, upscale=1, min_size=1024, fix_resize=None):
    '''
    PIL.Image -> Tensor[C, H, W], RGB, [-1, 1]
    '''
    # size
    w, h = img.size
    w *= upscale
    h *= upscale
    w0, h0 = round(w), round(h)
    if min(w, h) < min_size:
        _upscale = min_size / min(w, h)
        w *= _upscale
        h *= _upscale
    if fix_resize is not None:
        _upscale = fix_resize / min(w, h)
        w *= _upscale
        h *= _upscale
        w0, h0 = round(w), round(h)
    w = int(np.round(w / 64.0)) * 64
    h = int(np.round(h / 64.0)) * 64
    x = img.resize((w, h), Image.BICUBIC)
    x = np.array(x).round().clip(0, 255).astype(np.uint8)
    x = x / 255 * 2 - 1
    x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
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
