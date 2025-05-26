# ------------------------------------------------------------------------
#
#   Ultimate VAE Tile Optimization
#
#   Introducing a revolutionary new optimization designed to make
#   the VAE work with giant images on limited VRAM!
#   Say goodbye to the frustration of OOM and hello to seamless output!
#
# ------------------------------------------------------------------------
#
#   This script is a wild hack that splits the image into tiles,
#   encodes each tile separately, and merges the result back together.
#
#   Advantages:
#   - The VAE can now work with giant images on limited VRAM
#       (~10 GB for 8K images!)
#   - The merged output is completely seamless without any post-processing.
#
#   Drawbacks:
#   - Giant RAM needed. To store the intermediate results for a 4096x4096
#       images, you need 32 GB RAM it consumes ~20GB); for 8192x8192
#       you need 128 GB RAM machine (it consumes ~100 GB)
#   - NaNs always appear in for 8k images when you use fp16 (half) VAE
#       You must use --no-half-vae to disable half VAE for that giant image.
#   - Slow speed. With default tile size, it takes around 50/200 seconds
#       to encode/decode a 4096x4096 image; and 200/900 seconds to encode/decode
#       a 8192x8192 image. (The speed is limited by both the GPU and the CPU.)
#   - The gradient calculation is not compatible with this hack. It
#       will break any backward() or torch.autograd.grad() that passes VAE.
#       (But you can still use the VAE to generate training data.)
#
#   How it works:
#   1) The image is split into tiles.
#       - To ensure perfect results, each tile is padded with 32 pixels
#           on each side.
#       - Then the conv2d/silu/upsample/downsample can produce identical
#           results to the original image without splitting.
#   2) The original forward is decomposed into a task queue and a task worker.
#       - The task queue is a list of functions that will be executed in order.
#       - The task worker is a loop that executes the tasks in the queue.
#   3) The task queue is executed for each tile.
#       - Current tile is sent to GPU.
#       - local operations are directly executed.
#       - Group norm calculation is temporarily suspended until the mean
#           and var of all tiles are calculated.
#       - The residual is pre-calculated and stored and addded back later.
#       - When need to go to the next tile, the current tile is send to cpu.
#   4) After all tiles are processed, tiles are merged on cpu and return.
#
#   Enjoy!
#
#   @author: LI YI @ Nanyang Technological University - Singapore
#   @date: 2023-03-02
#   @license: MIT License
#
#   Please give me a star if you like this project!
#
# -------------------------------------------------------------------------

import gc
from time import time
import math
import sys # Import sys module
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.version
import torch.nn.functional as F
from einops import rearrange
from diffusers.utils.import_utils import is_xformers_available

import SUPIR.utils.devices as devices

try:
    import xformers
    import xformers.ops
except ImportError:
    pass

sd_flag = True

def get_recommend_encoder_tile_size():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(
            devices.device).total_memory // 2**20
        if total_memory > 16*1000:
            ENCODER_TILE_SIZE = 3072
        elif total_memory > 12*1000:
            ENCODER_TILE_SIZE = 2048
        elif total_memory > 8*1000:
            ENCODER_TILE_SIZE = 1536
        else:
            ENCODER_TILE_SIZE = 960
    else:
        ENCODER_TILE_SIZE = 512
    return ENCODER_TILE_SIZE


def get_recommend_decoder_tile_size():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(
            devices.device).total_memory // 2**20
        if total_memory > 30*1000:
            DECODER_TILE_SIZE = 256
        elif total_memory > 16*1000:
            DECODER_TILE_SIZE = 192
        elif total_memory > 12*1000:
            DECODER_TILE_SIZE = 128
        elif total_memory > 8*1000:
            DECODER_TILE_SIZE = 96
        else:
            DECODER_TILE_SIZE = 64
    else:
        DECODER_TILE_SIZE = 64
    return DECODER_TILE_SIZE


if 'global const':
    DEFAULT_ENABLED = False
    DEFAULT_MOVE_TO_GPU = False
    DEFAULT_FAST_ENCODER = True
    DEFAULT_FAST_DECODER = True
    DEFAULT_COLOR_FIX = 0
    DEFAULT_ENCODER_TILE_SIZE = get_recommend_encoder_tile_size()
    DEFAULT_DECODER_TILE_SIZE = get_recommend_decoder_tile_size()
    DEFAULT_NUM_PARALLEL_WORKERS = 1 # Default to 1 to maintain original behavior


# inplace version of silu
def inplace_nonlinearity(x):
    # Test: fix for Nans
    return F.silu(x, inplace=True)

# extracted from ldm.modules.diffusionmodules.model

# ====================================================================================
# from diffusers lib
def attn_forward_new(self, h_):
    batch_size, channel, height, width = h_.shape
    hidden_states = h_.view(batch_size, channel, height * width).transpose(1, 2)

    attention_mask = None
    encoder_hidden_states = None
    batch_size, sequence_length, _ = hidden_states.shape
    attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    query = self.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif self.norm_cross:
        encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

    key = self.to_k(encoder_hidden_states)
    value = self.to_v(encoder_hidden_states)

    query = self.head_to_batch_dim(query)
    key = self.head_to_batch_dim(key)
    value = self.head_to_batch_dim(value)

    attention_probs = self.get_attention_scores(query, key, attention_mask)
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = self.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    return hidden_states

# ====================================================================================
def attn_forward_new_pt2_0(self, hidden_states,):
    scale = 1
    attention_mask = None
    encoder_hidden_states = None

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

    if self.group_norm is not None:
        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = self.to_q(hidden_states, scale=scale)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif self.norm_cross:
        encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

    key = self.to_k(encoder_hidden_states, scale=scale)
    value = self.to_v(encoder_hidden_states, scale=scale)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // self.heads

    query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = self.to_out[0](hidden_states, scale=scale)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    return hidden_states

# ====================================================================================
def attn_forward_new_xformers(self, hidden_states):
    scale = 1
    attention_op = None
    attention_mask = None
    encoder_hidden_states = None

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, key_tokens, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    attention_mask = self.prepare_attention_mask(attention_mask, key_tokens, batch_size)
    if attention_mask is not None:
        # expand our mask's singleton query_tokens dimension:
        #   [batch*heads,            1, key_tokens] ->
        #   [batch*heads, query_tokens, key_tokens]
        # so that it can be added as a bias onto the attention scores that xformers computes:
        #   [batch*heads, query_tokens, key_tokens]
        # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
        _, query_tokens, _ = hidden_states.shape
        attention_mask = attention_mask.expand(-1, query_tokens, -1)

    if self.group_norm is not None:
        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = self.to_q(hidden_states, scale=scale)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif self.norm_cross:
        encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

    key = self.to_k(encoder_hidden_states, scale=scale)
    value = self.to_v(encoder_hidden_states, scale=scale)

    query = self.head_to_batch_dim(query).contiguous()
    key = self.head_to_batch_dim(key).contiguous()
    value = self.head_to_batch_dim(value).contiguous()

    hidden_states = xformers.ops.memory_efficient_attention(
        query, key, value, attn_bias=attention_mask, op=attention_op#, scale=scale
    )
    hidden_states = hidden_states.to(query.dtype)
    hidden_states = self.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states, scale=scale)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    return hidden_states

# ====================================================================================
# new attention function for sdpa
def attn_forward_sdpa(self, h_):
    """
    Attention implementation using PyTorch's actual scaled_dot_product_attention
    function with compatible tensor reshaping for SUPIR's architecture.
    """
    # Get q, k, v using either naming convention
    if hasattr(self, 'q'):
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
    elif hasattr(self, 'to_q'):
        q = self.to_q(h_)
        k = self.to_k(h_)
        v = self.to_v(h_)
    else:
        # Fall back if the structure is unknown
        return attn_forward_new(self, h_)

    # Get shapes
    b, c, h, w = q.shape
    
    # For compatibility with different architectures, we'll use the actual standard attention
    # implementation first, then apply SDPA to the attention matrix directly
    
    # Reshape for standard attention calculation
    q_flat = q.reshape(b, c, h*w)
    k_flat = k.reshape(b, c, h*w)
    v_flat = v.reshape(b, c, h*w)
    
    q_flat = q_flat.permute(0, 2, 1)   # b,hw,c
    
    # Now we can use SDPA with the proper shape
    # We need to add the heads dimension for SDPA (batch, heads, seq_len, dim)
    # In our case, we're treating it as a single head with the full dimension
    q_sdpa = q_flat.unsqueeze(1)     # [b, 1, hw, c]
    k_sdpa = k_flat.permute(0, 2, 1).unsqueeze(1)  # [b, 1, hw, c]
    v_sdpa = v_flat.permute(0, 2, 1).unsqueeze(1)  # [b, 1, hw, c]
    
    # Apply SDPA
    attn_output = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)  # [b, 1, hw, c]
    
    # Remove the head dimension and reshape back
    attn_output = attn_output.squeeze(1)  # [b, hw, c]
    attn_output = attn_output.permute(0, 2, 1).reshape(b, c, h, w)  # [b, c, h, w]
    
    # Apply output projection
    if hasattr(self, 'proj_out'):
        out = self.proj_out(attn_output)
    elif hasattr(self, 'to_out'):
        if isinstance(self.to_out, torch.nn.Sequential):
            out = self.to_out[0](attn_output.reshape(b, c, h*w).permute(0, 2, 1))
            out = self.to_out[1](out)
            out = out.permute(0, 2, 1).reshape(b, c, h, w)
        else:
            out = self.to_out(attn_output.reshape(b, c, h*w).permute(0, 2, 1))
            out = out.permute(0, 2, 1).reshape(b, c, h, w)
    else:
        out = attn_output
        
    return out

# ====================================================================================
def attn_forward(self, h_):
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    # compute attention
    b, c, h, w = q.shape
    q = q.reshape(b, c, h*w)
    q = q.permute(0, 2, 1)   # b,hw,c
    k = k.reshape(b, c, h*w)  # b,c,hw
    w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
    w_ = w_ * (int(c)**(-0.5))
    w_ = torch.nn.functional.softmax(w_, dim=2)

    # attend to values
    v = v.reshape(b, c, h*w)
    w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
    # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
    h_ = torch.bmm(v, w_)
    h_ = h_.reshape(b, c, h, w)

    h_ = self.proj_out(h_)

    return h_

# ====================================================================================
def xformer_attn_forward(self, h_):
    """
    Safer version of xformer_attn_forward that handles import errors
    and different attention block structures.
    """
    try:
        import xformers
        import xformers.ops
    except Exception:
        return attn_forward_new(self, h_)
    
    # Get q, k, v using either naming convention
    if hasattr(self, 'q'):
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
    elif hasattr(self, 'to_q'):
        q = self.to_q(h_)
        k = self.to_k(h_)
        v = self.to_v(h_)
    else:
        return attn_forward_new(self, h_)

    # compute attention
    B, C, H, W = q.shape
    
    # Use einops if available, otherwise reshape manually
    try:
        from einops import rearrange
        q, k, v = map(lambda x: rearrange(x, 'b c h w -> b (h w) c'), (q, k, v))
    except ImportError:
        q = q.reshape(B, C, H*W).permute(0, 2, 1)  # B, HW, C
        k = k.reshape(B, C, H*W).permute(0, 2, 1)  # B, HW, C
        v = v.reshape(B, C, H*W).permute(0, 2, 1)  # B, HW, C

    # Reshape for xformers
    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(B, t.shape[1], 1, C)
        .permute(0, 2, 1, 3)
        .reshape(B * 1, t.shape[1], C)
        .contiguous(),
        (q, k, v),
    )
    
    # Get attention_op if available
    attention_op = getattr(self, 'attention_op', None)
    
    # Use xformers memory efficient attention
    out = xformers.ops.memory_efficient_attention(
        q, k, v, attn_bias=None, op=attention_op)

    # Reshape back
    out = (
        out.unsqueeze(0)
        .reshape(B, 1, out.shape[1], C)
        .permute(0, 2, 1, 3)
        .reshape(B, out.shape[1], C)
    )
    
    # Back to original shape
    try:
        from einops import rearrange
        out = rearrange(out, 'b (h w) c -> b c h w', b=B, h=H, w=W, c=C)
    except ImportError:
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
    
    # Apply output projection
    if hasattr(self, 'proj_out'):
        out = self.proj_out(out)
    elif hasattr(self, 'to_out'):
        if isinstance(self.to_out, torch.nn.Sequential):
            out = self.to_out[0](out.reshape(B, C, H*W).permute(0, 2, 1))
            out = self.to_out[1](out)
            out = out.permute(0, 2, 1).reshape(B, C, H, W)
        else:
            out = self.to_out(out)
    
    return out

# ====================================================================================
def attn2task(task_queue, net):
    if False: #isinstance(net, AttnBlock):
        task_queue.append(('store_res', lambda x: x))
        task_queue.append(('pre_norm', net.norm))
        task_queue.append(('attn', lambda x, net=net: attn_forward(net, x)))
        task_queue.append(['add_res', None])
    elif False: #isinstance(net, MemoryEfficientAttnBlock):
        task_queue.append(('store_res', lambda x: x))
        task_queue.append(('pre_norm', net.norm))
        task_queue.append(
            ('attn', lambda x, net=net: xformer_attn_forward(net, x)))
        task_queue.append(['add_res', None])
    else:
        task_queue.append(('store_res', lambda x: x))
        task_queue.append(('pre_norm', net.norm))
        
        # Prioritize SDPA if PyTorch supports it
        if hasattr(F, "scaled_dot_product_attention"):
            print("[Tiled VAE]: Using PyTorch SDPA-based attention")
            task_queue.append(('attn', lambda x, net=net: attn_forward_sdpa(net, x)))
        # Only use xformers if explicitly available and imported
        elif is_xformers_available and 'xformers' in sys.modules:
            try:
                import xformers
                import xformers.ops
                print("[Tiled VAE]: Using xformers for attention")
                task_queue.append(
                    ('attn', lambda x, net=net: xformer_attn_forward(net, x)))
            except Exception as e:
                print(f"[Tiled VAE]: xformers import failed ({str(e)}), falling back to standard attention")
                task_queue.append(('attn', lambda x, net=net: attn_forward_new(net, x)))
        else:
            print("[Tiled VAE]: Using standard attention")
            task_queue.append(('attn', lambda x, net=net: attn_forward_new(net, x)))
        



        task_queue.append(['add_res', None])

# ====================================================================================
def resblock2task(queue, block):
    """
    Turn a ResNetBlock into a sequence of tasks and append to the task queue

    @param queue: the target task queue
    @param block: ResNetBlock

    """
    if block.in_channels != block.out_channels:
        if sd_flag:
            if block.use_conv_shortcut:
                queue.append(('store_res', block.conv_shortcut))
            else:
                queue.append(('store_res', block.nin_shortcut))
        else:
            if block.use_in_shortcut:
                queue.append(('store_res', block.conv_shortcut))
            else:
                queue.append(('store_res', block.nin_shortcut))

    else:
        queue.append(('store_res', lambda x: x))
    queue.append(('pre_norm', block.norm1))
    queue.append(('silu', inplace_nonlinearity))
    queue.append(('conv1', block.conv1))
    queue.append(('pre_norm', block.norm2))
    queue.append(('silu', inplace_nonlinearity))
    queue.append(('conv2', block.conv2))
    queue.append(['add_res', None])


def build_sampling(task_queue, net, is_decoder):
    """
    Build the sampling part of a task queue
    @param task_queue: the target task queue
    @param net: the network
    @param is_decoder: currently building decoder or encoder
    """
    if is_decoder:
        if sd_flag:
            resblock2task(task_queue, net.mid.block_1)
            attn2task(task_queue, net.mid.attn_1)
            print(task_queue)
            resblock2task(task_queue, net.mid.block_2)
            resolution_iter = reversed(range(net.num_resolutions))
            block_ids = net.num_res_blocks + 1
            condition = 0
            module = net.up
            func_name = 'upsample'
        else:
            resblock2task(task_queue, net.mid_block.resnets[0])
            attn2task(task_queue, net.mid_block.attentions[0])
            resblock2task(task_queue, net.mid_block.resnets[1])
            resolution_iter = (range(len(net.up_blocks)))  # net.num_resolutions = 3
            block_ids = 2 + 1
            condition = len(net.up_blocks) - 1
            module = net.up_blocks
            func_name = 'upsamplers'
    else:
        if sd_flag:
            resolution_iter = range(net.num_resolutions)
            block_ids = net.num_res_blocks
            condition = net.num_resolutions - 1
            module = net.down
            func_name = 'downsample'
        else:
            resolution_iter = range(len(net.down_blocks))
            block_ids = 2
            condition = len(net.down_blocks) - 1
            module = net.down_blocks
            func_name = 'downsamplers'

    for i_level in resolution_iter:
        for i_block in range(block_ids):
            if sd_flag:
                resblock2task(task_queue, module[i_level].block[i_block])
            else:
                resblock2task(task_queue, module[i_level].resnets[i_block])
        if i_level != condition:
            if sd_flag:
                task_queue.append((func_name, getattr(module[i_level], func_name)))
            else:
                if is_decoder:
                    task_queue.append((func_name, module[i_level].upsamplers[0]))
                else:
                    task_queue.append((func_name, module[i_level].downsamplers[0]))

    if not is_decoder:
        if sd_flag:
            resblock2task(task_queue, net.mid.block_1)
            attn2task(task_queue, net.mid.attn_1)
            resblock2task(task_queue, net.mid.block_2)
        else:
            resblock2task(task_queue, net.mid_block.resnets[0])
            attn2task(task_queue, net.mid_block.attentions[0])
            resblock2task(task_queue, net.mid_block.resnets[1])


def build_task_queue(net, is_decoder):
    """
    Build a single task queue for the encoder or decoder
    @param net: the VAE decoder or encoder network
    @param is_decoder: currently building decoder or encoder
    @return: the task queue
    """
    task_queue = []
    task_queue.append(('conv_in', net.conv_in))

    # construct the sampling part of the task queue
    # because encoder and decoder share the same architecture, we extract the sampling part
    build_sampling(task_queue, net, is_decoder)
    if is_decoder and not sd_flag:
        net.give_pre_end = False
        net.tanh_out = False

    if not is_decoder or not net.give_pre_end:
        if sd_flag:
            task_queue.append(('pre_norm', net.norm_out))
        else:
            task_queue.append(('pre_norm', net.conv_norm_out))
        task_queue.append(('silu', inplace_nonlinearity))
        task_queue.append(('conv_out', net.conv_out))
        if is_decoder and net.tanh_out:
            task_queue.append(('tanh', torch.tanh))

    return task_queue


def clone_task_queue(task_queue):
    """
    Clone a task queue
    @param task_queue: the task queue to be cloned
    @return: the cloned task queue
    """
    return [[item for item in task] for task in task_queue]


def get_var_mean(input, num_groups, eps=1e-6):
    """
    Get mean and var for group norm
    """
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c/num_groups)
    input_reshaped = input.contiguous().view(
        1, int(b * num_groups), channel_in_group, *input.size()[2:])
    var, mean = torch.var_mean(
        input_reshaped, dim=[0, 2, 3, 4], unbiased=False)
    return var, mean


def custom_group_norm(input, num_groups, mean, var, weight=None, bias=None, eps=1e-6):
    """
    Custom group norm with fixed mean and var

    @param input: input tensor
    @param num_groups: number of groups. by default, num_groups = 32
    @param mean: mean, must be pre-calculated by get_var_mean
    @param var: var, must be pre-calculated by get_var_mean
    @param weight: weight, should be fetched from the original group norm
    @param bias: bias, should be fetched from the original group norm
    @param eps: epsilon, by default, eps = 1e-6 to match the original group norm

    @return: normalized tensor
    """
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c/num_groups)
    input_reshaped = input.contiguous().view(
        1, int(b * num_groups), channel_in_group, *input.size()[2:])

    out = F.batch_norm(input_reshaped, mean, var, weight=None, bias=None,
                       training=False, momentum=0, eps=eps)

    out = out.view(b, c, *input.size()[2:])

    # post affine transform
    if weight is not None:
        out *= weight.view(1, -1, 1, 1)
    if bias is not None:
        out += bias.view(1, -1, 1, 1)
    return out


def crop_valid_region(x, input_bbox, target_bbox, is_decoder):
    """
    Crop the valid region from the tile
    @param x: input tile
    @param input_bbox: original input bounding box
    @param target_bbox: output bounding box
    @param scale: scale factor
    @return: cropped tile
    """
    padded_bbox = [i * 8 if is_decoder else i//8 for i in input_bbox]
    margin = [target_bbox[i] - padded_bbox[i] for i in range(4)]
    return x[:, :, margin[2]:x.size(2)+margin[3], margin[0]:x.size(3)+margin[1]]

# ↓↓↓ https://github.com/Kahsolt/stable-diffusion-webui-vae-tile-infer ↓↓↓


def perfcount(fn):
    def wrapper(*args, **kwargs):
        ts = time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(devices.device)
        devices.torch_gc()
        gc.collect()

        ret = fn(*args, **kwargs)

        devices.torch_gc()
        gc.collect()
        if torch.cuda.is_available():
            vram = torch.cuda.max_memory_allocated(devices.device) / 2**20
            torch.cuda.reset_peak_memory_stats(devices.device)
            print(
                # f'[Tiled VAE]: Done in {time() - ts:.3f}s, max VRAM alloc {vram:.3f} MB')
                f'[Tiled VAE]: Done in {time() - ts:.3f}s')
        else:
            print(f'[Tiled VAE]: Done in {time() - ts:.3f}s')

        return ret
    return wrapper




class GroupNormParam:
    def __init__(self):
        self.var_list = []
        self.mean_list = []
        self.pixel_list = []
        self.weight = None
        self.bias = None

    def add_tile(self, tile, layer):
        var, mean = get_var_mean(tile, 32)
        # For giant images, the variance can be larger than max float16
        # In this case we create a copy to float32
        if var.dtype == torch.float16 and var.isinf().any():
            fp32_tile = tile.float()
            var, mean = get_var_mean(fp32_tile, 32)
        # ============= DEBUG: test for infinite =============
        # if torch.isinf(var).any():
        #    print('var: ', var)
        # ====================================================
        self.var_list.append(var)
        self.mean_list.append(mean)
        self.pixel_list.append(
            tile.shape[2]*tile.shape[3])
        if hasattr(layer, 'weight'):
            self.weight = layer.weight
            self.bias = layer.bias
        else:
            self.weight = None
            self.bias = None

    def summary(self):
        """
        summarize the mean and var and return a function
        that apply group norm on each tile
        """
        if len(self.var_list) == 0:
            return None
        var = torch.vstack(self.var_list)
        mean = torch.vstack(self.mean_list)
        max_value = max(self.pixel_list)
        pixels = torch.tensor(
            self.pixel_list, dtype=torch.float32, device=devices.device) / max_value
        sum_pixels = torch.sum(pixels)
        pixels = pixels.unsqueeze(
            1) / sum_pixels
        var = torch.sum(
            var * pixels, dim=0)
        mean = torch.sum(
            mean * pixels, dim=0)
        return lambda x:  custom_group_norm(x, 32, mean, var, self.weight, self.bias)

    @staticmethod
    def from_tile(tile, norm):
        """
        create a function from a single tile without summary
        """
        var, mean = get_var_mean(tile, 32)
        if var.dtype == torch.float16 and var.isinf().any():
            fp32_tile = tile.float()
            var, mean = get_var_mean(fp32_tile, 32)
            # if it is a macbook, we need to convert back to float16
            if var.device.type == 'mps':
                # clamp to avoid overflow
                var = torch.clamp(var, 0, 60000)
                var = var.half()
                mean = mean.half()
        if hasattr(norm, 'weight'):
            weight = norm.weight
            bias = norm.bias
        else:
            weight = None
            bias = None

        def group_norm_func(x, mean=mean, var=var, weight=weight, bias=bias):
            return custom_group_norm(x, 32, mean, var, weight, bias, 1e-6)
        return group_norm_func


class VAEHook:
    def __init__(self, net, tile_size, is_decoder, fast_decoder, fast_encoder, color_fix, to_gpu=False, num_parallel_workers=DEFAULT_NUM_PARALLEL_WORKERS, ae_dtype=torch.float32):
        self.net = net                  # encoder | decoder
        self.tile_size = tile_size
        self.is_decoder = is_decoder
        self.ae_dtype = ae_dtype # Store the autoencoder dtype
        self.fast_mode = (fast_encoder and not is_decoder) or (
            fast_decoder and is_decoder)
        self.color_fix = color_fix and not is_decoder
        self.to_gpu = to_gpu
        self.pad = 11 if is_decoder else 32
        self.num_parallel_workers = num_parallel_workers
        if self.num_parallel_workers > 1:
            print(f"[Tiled VAE]: Parallel processing enabled with {self.num_parallel_workers} workers.")

    def __call__(self, x):
        B, C, H, W = x.shape
        original_device = next(self.net.parameters()).device
        try:
            if self.to_gpu:
                self.net.to(devices.get_optimal_device())
            if max(H, W) <= self.pad * 2 + self.tile_size:
                print("[Tiled VAE]: the input size is tiny and unnecessary to tile.")
                return self.net.original_forward(x)
            else:
                return self.vae_tile_forward(x)
        finally:
            self.net.to(original_device)

    def get_best_tile_size(self, lowerbound, upperbound):
        """
        Get the best tile size for GPU memory
        """
        divider = 32
        while divider >= 2:
            remainer = lowerbound % divider
            if remainer == 0:
                return lowerbound
            candidate = lowerbound - remainer + divider
            if candidate <= upperbound:
                return candidate
            divider //= 2
        return lowerbound

    def split_tiles(self, h, w):
        """
        Tool function to split the image into tiles
        @param h: height of the image
        @param w: width of the image
        @return: tile_input_bboxes, tile_output_bboxes
        """
        tile_input_bboxes, tile_output_bboxes = [], []
        tile_size = self.tile_size
        pad = self.pad
        num_height_tiles = math.ceil((h - 2 * pad) / tile_size)
        num_width_tiles = math.ceil((w - 2 * pad) / tile_size)
        # If any of the numbers are 0, we let it be 1
        # This is to deal with long and thin images
        num_height_tiles = max(num_height_tiles, 1)
        num_width_tiles = max(num_width_tiles, 1)

        # Suggestions from https://github.com/Kahsolt: auto shrink the tile size
        real_tile_height = math.ceil((h - 2 * pad) / num_height_tiles)
        real_tile_width = math.ceil((w - 2 * pad) / num_width_tiles)
        real_tile_height = self.get_best_tile_size(real_tile_height, tile_size)
        real_tile_width = self.get_best_tile_size(real_tile_width, tile_size)

        print(f'[Tiled VAE]: split to {num_height_tiles}x{num_width_tiles} = {num_height_tiles*num_width_tiles} tiles. ' +
              f'Optimal tile size {real_tile_width}x{real_tile_height}, original tile size {tile_size}x{tile_size}')

        for i in range(num_height_tiles):
            for j in range(num_width_tiles):
                # bbox: [x1, x2, y1, y2]
                # the padding is is unnessary for image borders. So we directly start from (32, 32)
                input_bbox = [
                    pad + j * real_tile_width,
                    min(pad + (j + 1) * real_tile_width, w),
                    pad + i * real_tile_height,
                    min(pad + (i + 1) * real_tile_height, h),
                ]

                # if the output bbox is close to the image boundary, we extend it to the image boundary
                output_bbox = [
                    input_bbox[0] if input_bbox[0] > pad else 0,
                    input_bbox[1] if input_bbox[1] < w - pad else w,
                    input_bbox[2] if input_bbox[2] > pad else 0,
                    input_bbox[3] if input_bbox[3] < h - pad else h,
                ]

                # scale to get the final output bbox
                output_bbox = [x * 8 if self.is_decoder else x // 8 for x in output_bbox]
                tile_output_bboxes.append(output_bbox)

                # indistinguishable expand the input bbox by pad pixels
                tile_input_bboxes.append([
                    max(0, input_bbox[0] - pad),
                    min(w, input_bbox[1] + pad),
                    max(0, input_bbox[2] - pad),
                    min(h, input_bbox[3] + pad),
                ])

        return tile_input_bboxes, tile_output_bboxes

    @torch.no_grad()
    def _process_tile_segment(self, tile_data_cpu, task_queue_segment, device, worker_ae_dtype, pbar_update_func, is_fast_mode, input_bbox_for_crop, out_bbox_for_crop):
        """
        Worker function to process a segment of a tile's task queue.
        Returns a dictionary containing the processed tile (or None if fully processed),
        group norm stats (if applicable), and an error flag.
        """
        tile_idx, tile_cpu = tile_data_cpu
        current_tile_task_queue = clone_task_queue(task_queue_segment) # Work on a copy
        processed_tile_gpu = None
        group_norm_stats = None # Will be (var, mean, pixel_count, weight, bias)
        norm_layer_ref = None # To get weight/bias later
        tile_fully_processed = False
        error_occurred = False

        try:
            # Autocast context for the worker thread
            with torch.autocast(device_type=device.type, dtype=worker_ae_dtype, enabled=worker_ae_dtype != torch.float32):
                tile_gpu = tile_cpu.to(device, dtype=worker_ae_dtype)
                del tile_cpu # Free CPU memory for this tile

                while len(current_tile_task_queue) > 0:
                    task = current_tile_task_queue.pop(0)
                    task_type = task[0]

                    if task_type == 'pre_norm':
                        norm_layer_ref = task[1]
                        var, mean = get_var_mean(tile_gpu, 32)
                        if var.dtype == torch.float16 and var.isinf().any():
                            fp32_tile_gpu = tile_gpu.float()
                            var, mean = get_var_mean(fp32_tile_gpu, 32)
                            del fp32_tile_gpu
                        pixel_count = tile_gpu.shape[2] * tile_gpu.shape[3]
                        group_norm_stats = (var, mean, pixel_count)
                        processed_tile_gpu = tile_gpu # Return tile at pre_norm stage
                        break # Stop processing for this segment, needs global sync
                    elif task_type == 'store_res' or task_type == 'store_res_cpu':
                        task_id = 0
                        res_gpu = task[1](tile_gpu)
                        # In parallel mode, keep res on GPU if fast_mode, else move to CPU if it was store_res_cpu
                        # For simplicity and to avoid complex CPU/GPU transfers within worker,
                        # we'll primarily keep intermediate results on GPU if possible,
                        # or handle CPU transfer decision in the main loop if truly necessary.
                        # For now, 'store_res_cpu' implies the original logic wanted it on CPU,
                        # but in parallel, it might be better to keep on GPU until segment ends.
                        # Let's assume for now 'res' stays on GPU unless explicitly moved by main logic.
                        if not is_fast_mode or task_type == 'store_res_cpu':
                             # If strict CPU storage is needed, this would be res_gpu.cpu()
                             # but that adds complexity for re-merging.
                             # For now, we assume task[1] (the residual) is kept on GPU.
                             pass # Placeholder for potential CPU transfer logic

                        while current_tile_task_queue[task_id][0] != 'add_res':
                            task_id += 1
                        current_tile_task_queue[task_id][1] = res_gpu # Store the residual (on GPU)
                    elif task_type == 'add_res':
                        if task[1] is not None: # Residual might be None if already processed or not set
                            tile_gpu += task[1].to(device) # Ensure residual is on correct device
                            # The task item task[1] (the residual tensor) has been consumed.
                            # The task itself is popped from current_tile_task_queue. No need to modify via task_id.
                    elif task_type == 'apply_norm': # This is applied after global sync
                        group_norm_func = task[1]
                        tile_gpu = group_norm_func(tile_gpu)
                    else: # Other direct operations
                        tile_gpu = task[1](tile_gpu)
                    
                    if pbar_update_func:
                        pbar_update_func(1)

                if len(current_tile_task_queue) == 0: # Tile processing finished
                    tile_fully_processed = True
                # The final result for this tile will be assembled in the main thread
                # to avoid race conditions on the `result` tensor.
                # We return the processed tile on GPU.
                processed_tile_gpu = tile_gpu
            
            # If not fully processed and not stopped at pre_norm, it means it's an intermediate state.
            # The tile remains on GPU.

        except Exception as e:
            print(f"[Tiled VAE Worker Error for tile {tile_idx}]: {e}")
            error_occurred = True
            # Ensure tile_gpu is None or on CPU to avoid holding GPU memory on error
            if processed_tile_gpu is not None and processed_tile_gpu.device.type != 'cpu':
                processed_tile_gpu = processed_tile_gpu.cpu() if processed_tile_gpu is not None else None


        return {
            "tile_idx": tile_idx,
            "processed_tile_gpu": processed_tile_gpu, # On GPU, or CPU if error
            "remaining_task_queue": current_tile_task_queue,
            "group_norm_stats": group_norm_stats, # (var, mean, pixel_count)
            "norm_layer_ref": norm_layer_ref, # for weight/bias
            "tile_fully_processed": tile_fully_processed,
            "error_occurred": error_occurred,
            "input_bbox": input_bbox_for_crop, # Pass through for final cropping
            "out_bbox": out_bbox_for_crop,   # Pass through for final cropping
        }

    @torch.no_grad()
    def estimate_group_norm(self, z, task_queue, color_fix):
        device = z.device
        tile = z
        last_id = len(task_queue) - 1
        while last_id >= 0 and task_queue[last_id][0] != 'pre_norm':
            last_id -= 1
        if last_id <= 0 or task_queue[last_id][0] != 'pre_norm':
            raise ValueError('No group norm found in the task queue')
        # estimate until the last group norm
        for i in range(last_id + 1):
            task = task_queue[i]
            if task[0] == 'pre_norm':
                group_norm_func = GroupNormParam.from_tile(tile, task[1])
                task_queue[i] = ('apply_norm', group_norm_func)
                if i == last_id:
                    return True
                tile = group_norm_func(tile)
            elif task[0] == 'store_res':
                task_id = i + 1
                while task_id < last_id and task_queue[task_id][0] != 'add_res':
                    task_id += 1
                if task_id >= last_id:
                    continue
                task_queue[task_id][1] = task[1](tile)
            elif task[0] == 'add_res':
                tile += task[1].to(device)
                task[1] = None
            elif color_fix and task[0] == 'downsample':
                for j in range(i, last_id + 1):
                    if task_queue[j][0] == 'store_res':
                        task_queue[j] = ('store_res_cpu', task_queue[j][1])
                return True
            else:
                tile = task[1](tile)
            try:
                devices.test_for_nans(tile, "vae")
            except:
                print(f'Nan detected in fast mode estimation. Fast mode disabled.')
                return False

        raise IndexError('Should not reach here')

    # @perfcount
    @torch.no_grad()
    def vae_tile_forward(self, z):
        """
        Decode a latent vector z into an image in a tiled manner.
        @param z: latent vector
        @return: image
        """
        device = next(self.net.parameters()).device
        dtype = z.dtype
        net = self.net
        tile_size = self.tile_size
        is_decoder = self.is_decoder

        z = z.detach() # detach the input to avoid backprop

        N, height, width = z.shape[0], z.shape[2], z.shape[3]
        net.last_z_shape = z.shape

        # Split the input into tiles and build a task queue for each tile
        print(f'[Tiled VAE]: input_size: {z.shape}, tile_size: {tile_size}, padding: {self.pad}')

        in_bboxes, out_bboxes = self.split_tiles(height, width)

        # Prepare tiles by split the input latents
        tiles = []
        for input_bbox in in_bboxes:
            tile = z[:, :, input_bbox[2]:input_bbox[3], input_bbox[0]:input_bbox[1]].cpu()
            tiles.append(tile)

        num_tiles = len(tiles)
        num_completed = 0

        # Build task queues
        single_task_queue = build_task_queue(net, is_decoder)
        #print(single_task_queue)
        if self.fast_mode:
            # Fast mode: downsample the input image to the tile size,
            # then estimate the group norm parameters on the downsampled image
            scale_factor = tile_size / max(height, width)
            z = z.to(device)
            downsampled_z = F.interpolate(z, scale_factor=scale_factor, mode='nearest-exact')
            # use nearest-exact to keep statictics as close as possible
            print(f'[Tiled VAE]: Fast mode enabled, estimating group norm parameters on {downsampled_z.shape[3]} x {downsampled_z.shape[2]} image')

            # ======= Special thanks to @Kahsolt for distribution shift issue ======= #
            # The downsampling will heavily distort its mean and std, so we need to recover it.
            std_old, mean_old = torch.std_mean(z, dim=[0, 2, 3], keepdim=True)
            std_new, mean_new = torch.std_mean(downsampled_z, dim=[0, 2, 3], keepdim=True)
            downsampled_z = (downsampled_z - mean_new) / std_new * std_old + mean_old
            del std_old, mean_old, std_new, mean_new
            # occasionally the std_new is too small or too large, which exceeds the range of float16
            # so we need to clamp it to max z's range.
            downsampled_z = torch.clamp_(downsampled_z, min=z.min(), max=z.max())
            estimate_task_queue = clone_task_queue(single_task_queue)
            if self.estimate_group_norm(downsampled_z, estimate_task_queue, color_fix=self.color_fix):
                single_task_queue = estimate_task_queue
            del downsampled_z

        task_queues = [clone_task_queue(single_task_queue) for _ in range(num_tiles)]

        # Dummy result
        result = None
        result_approx = None
        #try:
        #    with devices.autocast():
        #        result_approx = torch.cat([F.interpolate(cheap_approximation(x).unsqueeze(0), scale_factor=opt_f, mode='nearest-exact') for x in z], dim=0).cpu()
        #except: pass
        # Free memory of input latent tensor
        del z

        # Task queue execution
        
        # If not using parallel workers, fall back to original sequential logic
        if self.num_parallel_workers <= 1:
            pbar = tqdm(total=num_tiles * len(task_queues[0]), desc=f"[Tiled VAE]: Executing {'Decoder' if is_decoder else 'Encoder'} Task Queue (Sequential): ")
            forward = True
            interrupted = False
            while True:
                group_norm_param = GroupNormParam()
                for i in range(num_tiles) if forward else reversed(range(num_tiles)):
                    if tiles[i] is None: # Already processed
                        continue
                    tile_gpu = tiles[i].to(device)
                    task_queue = task_queues[i]

                    while len(task_queue) > 0:
                        task = task_queue.pop(0)
                        if task[0] == 'pre_norm':
                            group_norm_param.add_tile(tile_gpu, task[1])
                            break 
                        elif task[0] == 'store_res' or task[0] == 'store_res_cpu':
                            task_id = 0
                            res = task[1](tile_gpu)
                            if not self.fast_mode or task[0] == 'store_res_cpu':
                                res = res.cpu()
                            while task_queue[task_id][0] != 'add_res':
                                task_id += 1
                            task_queue[task_id][1] = res
                        elif task[0] == 'add_res':
                            if task[1] is not None:
                                tile_gpu += task[1].to(device)
                                task[1] = None # Mark as used
                        elif task[0] == 'apply_norm': # Added after summary
                            tile_gpu = task[1](tile_gpu)
                        else:
                            tile_gpu = task[1](tile_gpu)
                        pbar.update(1)

                    if len(task_queue) == 0: # Tile finished
                        tiles[i] = None # Mark as processed
                        num_completed += 1
                        if result is None:
                            result = torch.zeros((N, tile_gpu.shape[1], height * 8 if is_decoder else height // 8, width * 8 if is_decoder else width // 8), dtype=dtype, device=device, requires_grad=False)
                        result[:, :, out_bboxes[i][2]:out_bboxes[i][3], out_bboxes[i][0]:out_bboxes[i][1]] = crop_valid_region(tile_gpu, in_bboxes[i], out_bboxes[i], is_decoder)
                        del tile_gpu
                    elif i == num_tiles - 1 and forward:
                        forward = False
                        tiles[i] = tile_gpu # Keep on GPU
                    elif i == 0 and not forward:
                        forward = True
                        tiles[i] = tile_gpu # Keep on GPU
                    else:
                        tiles[i] = tile_gpu.cpu() # Move to CPU
                        del tile_gpu
                
                if num_completed == num_tiles: break
                
                group_norm_func = group_norm_param.summary()
                if group_norm_func is not None:
                    for i in range(num_tiles):
                        if tiles[i] is not None: # Only for unprocessed tiles
                             task_queues[i].insert(0, ('apply_norm', group_norm_func))
            pbar.close()
            return result.to(dtype) if result is not None else result_approx.to(device)

        # Parallel execution logic starts here
        # Use self.ae_dtype for the parallel execution if available, otherwise fallback to z.dtype
        parallel_ae_dtype = self.ae_dtype if hasattr(self, 'ae_dtype') else dtype
        pbar = tqdm(total=num_tiles * len(single_task_queue), desc=f"[Tiled VAE]: Executing {'Decoder' if is_decoder else 'Encoder'} Task Queue (Parallel x{self.num_parallel_workers}, dtype: {parallel_ae_dtype}): ")
        
        # Store tiles that are currently on GPU, managed by workers
        # For simplicity, we'll re-fetch from `tiles` (CPU) for each segment submission,
        # and workers will return processed tiles (on GPU) which then update `tiles_gpu_state`.
        tiles_gpu_state = [None] * num_tiles # Stores tile tensor if on GPU, or None if on CPU / processed

        with ThreadPoolExecutor(max_workers=self.num_parallel_workers) as executor:
            while num_completed < num_tiles:
                futures = []
                group_norm_param_parallel = GroupNormParam()
                
                # Identify tiles ready for next segment processing
                # A tile is ready if its task_queue is not empty
                active_tile_indices = [i for i, tq in enumerate(task_queues) if tq and (tiles[i] is not None or tiles_gpu_state[i] is not None)]

                # Phase 1: Process until pre_norm or completion for a batch of tiles
                # We need to manage which tiles are submitted to avoid OOM.
                # Submit work for available worker slots.
                
                submitted_this_round = 0
                for tile_idx in active_tile_indices:
                    if submitted_this_round >= self.num_parallel_workers * 2 : # Heuristic to keep pipeline full but not oversubmit
                         break

                    current_tile_task_queue = task_queues[tile_idx]
                    if not current_tile_task_queue: # Should not happen if active_tile_indices is correct
                        continue

                    # Determine input tile: from CPU cache `tiles` or from previous GPU state `tiles_gpu_state`
                    tile_input_for_worker = tiles_gpu_state[tile_idx] if tiles_gpu_state[tile_idx] is not None else tiles[tile_idx]
                    
                    if tile_input_for_worker is None: # Already fully processed and cleared
                        continue

                    # Ensure input is on CPU for the worker to manage its GPU transfer
                    if tile_input_for_worker.device.type != 'cpu':
                        tile_input_for_worker = tile_input_for_worker.cpu()
                    
                    # Submit to executor
                    # The worker will process until a 'pre_norm' or end of its queue segment
                    futures.append(executor.submit(self._process_tile_segment,
                                                    (tile_idx, tile_input_for_worker),
                                                    current_tile_task_queue,
                                                    device,
                                                    parallel_ae_dtype, # Pass the authoritative ae_dtype
                                                    pbar.update,
                                                    self.fast_mode,
                                                    in_bboxes[tile_idx],
                                                    out_bboxes[tile_idx]))
                    tiles_gpu_state[tile_idx] = None # Mark as "in-flight" / worker owns GPU copy
                    tiles[tile_idx] = None # Original CPU copy is now with worker or cleared
                    submitted_this_round +=1
                
                if not futures: # No work submitted, might mean all queues are empty or waiting for group_norm
                    if all(not tq for tq in task_queues if tiles[i] is not None or tiles_gpu_state[i] is not None for i, _ in enumerate(tq)): # check if all active queues are empty
                        break # All tasks done
                    # This state should ideally be handled by group_norm logic below if that's the blocker

                needs_group_norm_sync = False
                for future in futures:
                    worker_result = future.result()
                    tile_idx = worker_result["tile_idx"]
                    
                    if worker_result["error_occurred"]:
                        print(f"[Tiled VAE] Error processing tile {tile_idx}. Aborting parallel run for safety.")
                        # Simplest error handling: attempt to revert to sequential or just raise
                        # For now, let's signal completion to break loops and return whatever is done.
                        num_completed = num_tiles 
                        break # Break from processing future results

                    task_queues[tile_idx] = worker_result["remaining_task_queue"]
                    tiles_gpu_state[tile_idx] = worker_result["processed_tile_gpu"] # Update with GPU tensor

                    if worker_result["group_norm_stats"]:
                        needs_group_norm_sync = True
                        var, mean, pixel_count = worker_result["group_norm_stats"]
                        # Temporarily store raw stats; GroupNormParam needs layer ref for weight/bias
                        group_norm_param_parallel.var_list.append(var)
                        group_norm_param_parallel.mean_list.append(mean)
                        group_norm_param_parallel.pixel_list.append(pixel_count)
                        if worker_result["norm_layer_ref"] and group_norm_param_parallel.weight is None:
                             if hasattr(worker_result["norm_layer_ref"], 'weight'):
                                group_norm_param_parallel.weight = worker_result["norm_layer_ref"].weight
                                group_norm_param_parallel.bias = worker_result["norm_layer_ref"].bias


                    if worker_result["tile_fully_processed"]:
                        num_completed += 1
                        if result is None:
                            result_c = tiles_gpu_state[tile_idx].shape[1]
                            result_h = height * 8 if is_decoder else height // 8
                            result_w = width * 8 if is_decoder else width // 8
                            result = torch.zeros((N, result_c, result_h, result_w), dtype=dtype, device=device, requires_grad=False)
                        
                        processed_tile_final_gpu = tiles_gpu_state[tile_idx]
                        result[:, :, worker_result["out_bbox"][2]:worker_result["out_bbox"][3], worker_result["out_bbox"][0]:worker_result["out_bbox"][1]] = \
                            crop_valid_region(processed_tile_final_gpu, worker_result["input_bbox"], worker_result["out_bbox"], is_decoder)
                        tiles_gpu_state[tile_idx] = None # Clear GPU state for this tile
                        del processed_tile_final_gpu
                
                if num_completed == num_tiles and not needs_group_norm_sync : break # All done

                # Phase 2: If group_norm stats were collected, summarize and update task queues
                if needs_group_norm_sync:
                    group_norm_func = group_norm_param_parallel.summary()
                    if group_norm_func is not None:
                        for i in range(num_tiles):
                            # Update only for tiles that are not yet fully processed and have a pending pre_norm
                            if tiles_gpu_state[i] is not None and task_queues[i] and task_queues[i][0][0] != 'apply_norm':
                                # Check if the next task was indeed the pre_norm that was just handled implicitly
                                # The worker returns the tile *before* pre_norm is applied if it's a sync point.
                                # So, we add 'apply_norm' to the *front* of its *remaining* queue.
                                task_queues[i].insert(0, ('apply_norm', group_norm_func))
                    # Now tiles are ready for the next segment of processing after norm application. Loop will continue.
            
            if num_completed < num_tiles:
                 print(f"[Tiled VAE] Warning: Parallel execution finished but not all tiles completed ({num_completed}/{num_tiles}). Result might be incomplete.")


        pbar.close()
        # Ensure all GPU tensors in tiles_gpu_state are cleared if not moved to result
        for i in range(num_tiles):
            if tiles_gpu_state[i] is not None:
                tensor_obj = tiles_gpu_state[i] # Store the tensor object
                tiles_gpu_state[i] = None       # Remove reference from the list
                del tensor_obj                  # Explicitly delete the tensor object
        
        devices.torch_gc() # Clean up GPU memory

        return result.to(dtype) if result is not None else result_approx.to(device) # result_approx is from original code, might be None
