import torch
from sgm.models.diffusion import DiffusionEngine
from sgm.util import instantiate_from_config
import copy
from sgm.modules.distributions.distributions import DiagonalGaussianDistribution
import random
from SUPIR.utils.colorfix import wavelet_reconstruction, adaptive_instance_normalization
from pytorch_lightning import seed_everything
from torch.nn.functional import interpolate
from SUPIR.utils.tilevae import VAEHook
from Y7.colored_print import color, style
from pathlib import Path
import inspect 

"""

self.model:
 - The main diffusion model component (U-Net)

self.first_stage_model
 - SUPIR VAE component. Even though it's called first stage, the vae is used to encode/decode at beginning and end of diffusion.      

control_model 
 - An additional neural network that works alongside the main diffusion model (U-Net) 
   to provide more guidance control over the image generation process.  It is loaded into the main model


"""

class SUPIRModel(DiffusionEngine):
    def __init__(self, control_stage_config, ae_dtype='fp32', diffusion_dtype='fp32', p_p='', n_p='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        control_model = instantiate_from_config(control_stage_config)
        self.model.load_control_model(control_model)
        self.first_stage_model.denoise_encoder = copy.deepcopy(self.first_stage_model.encoder)
        self.sampler_config = kwargs['sampler_config']

        assert (ae_dtype in ['fp32', 'fp16', 'bf16']) and (diffusion_dtype in ['fp32', 'fp16', 'bf16'])
        if ae_dtype == 'fp32':
            ae_dtype = torch.float32
        elif ae_dtype == 'fp16':
            raise RuntimeError('fp16 cause NaN in AE')
        elif ae_dtype == 'bf16':
            ae_dtype = torch.bfloat16

        if diffusion_dtype == 'fp32':
            diffusion_dtype = torch.float32
        elif diffusion_dtype == 'fp16':
            diffusion_dtype = torch.float16
        elif diffusion_dtype == 'bf16':
            diffusion_dtype = torch.bfloat16

        self.ae_dtype = ae_dtype
        self.model.dtype = diffusion_dtype

        self.p_p = p_p
        self.n_p = n_p




    # SUPIR VAE - ENCODE
    @torch.no_grad()    
    def encode_first_stage(self, x):
        print(f"Current function: {inspect.currentframe().f_code.co_name}() (SUPIR VAE - ENCODE)", color.ORANGE)
        

        # first_stage_model is actually the SUPIR VAE component
        with torch.autocast("cuda", dtype=self.ae_dtype):
            z = self.first_stage_model.encode(x)
        z = self.scale_factor * z
        return z

    # SUPIR VAE - ENCODE
    @torch.no_grad()
    def encode_first_stage_with_denoise(self, x, use_sample=True, is_stage1=False):
        print(f"Current function: {inspect.currentframe().f_code.co_name}() (SUPIR VAE - ENCODE WITH DENOISE: is_stage1={is_stage1})", color.ORANGE)

        # first_stage_model is actually the SUPIR VAE component
        with torch.autocast("cuda", dtype=self.ae_dtype):
            if is_stage1:
                h = self.first_stage_model.denoise_encoder_s1(x)
            else:
                h = self.first_stage_model.denoise_encoder(x)
            moments = self.first_stage_model.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            if use_sample:
                z = posterior.sample()
            else:
                z = posterior.mode()
        z = self.scale_factor * z
        return z

    # SUPIR VAE - DECODE
    @torch.no_grad()
    def decode_first_stage(self, z):
        print(f"Current function: {inspect.currentframe().f_code.co_name}() (SUPIR VAE - DECODE)", color.ORANGE)
        

        z = 1.0 / self.scale_factor * z
        with torch.autocast("cuda", dtype=self.ae_dtype):
            out = self.first_stage_model.decode(z)

        # =============================================================
        # # Save decoded image
        # _z = self.encode_first_stage_with_denoise(img, use_sample=False)
        # x_stage1 = self.decode_first_stage(_z)   

        # # Save x_stage1 as JPG
        # import numpy as np
        # from PIL import Image
        # import os

        # # Create output directory if it doesn't exist
        # output_dir = str(Path.home() / "ai" / "SUPIR" / "output")
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        # # Convert to numpy and save
        # x_stage1_float = x_stage1.float()  # Ensure we're working with float32
        # image_np = ((x_stage1_float[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2)
        # image_np = np.clip(image_np, 0, 1)
        # image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        # image_pil.save(os.path.join(output_dir, "x_stage1_denoised.jpg"))
        # print(f"x_stage1 saved to {os.path.join(output_dir, 'x_stage1_denoised.jpg')}")
        # =============================================================
   
        return out.float()

    # ================================================================
    # utility function that provides a simplified way to perform the 
    # denoise-encode-decode process on a batch of images.
    # this is not used in general inference
    @torch.no_grad()
    def batchify_denoise(self, x, is_stage1=False):
        print(f"Current function: {inspect.currentframe().f_code.co_name}()", color.ORANGE)
        '''
        [N, C, H, W], [-1, 1], RGB
        '''
        x = self.encode_first_stage_with_denoise(x, use_sample=False, is_stage1=is_stage1)
        return self.decode_first_stage(x)

    # ========================================================================================
    @torch.no_grad()
    def batchify_sample(self, img, prompt, 
                        p_p='default', 
                        n_p='default', 
                        num_steps=50, 
                        restoration_scale=-1, 
                        s_churn=5, 
                        s_noise=1.003, 
                        cfg_scale_end=4.0, 
                        seed=-1,
                        num_samples=1, 
                        control_scale_end=1, 
                        color_fix_type='Wavelet', 
                        cfg_scale_start=1.0, 
                        control_scale_start=0.0,
                        skip_denoise_stage=False, 
                        **kwargs):

        # img : image tensor
        # prompt_lst : prompt list

        print(f"Current function: {inspect.currentframe().f_code.co_name}()", color.ORANGE)        

        '''
        [N, C], [-1, 1], RGB
        '''

        # !>>> Assert that we only have one image
        assert len(img) == 1, "This version only supports processing one image at a time"
    
        # !>>> Assert that multiple samples isn't being requested (optional, remove if needed)
        # assert num_samples == 1, "This version doesn't support multiple samples"

        # check if the color_fix_type parameter has a valid value
        assert color_fix_type in ['Wavelet', 'AdaIn', 'None']

        # additional pos/neg prompts
        if p_p == 'default':
            # use built-in default value stored in self.p_p
            p_p = self.p_p
        if n_p == 'default':
            # use built-in default value stored in self.n_p
            n_p = self.n_p

        self.sampler_config.params.num_steps = num_steps

        # scale_min is set to cfg_scale (the ending scale)
        # scale is set to cfg_scale_start (the starting scale)    
        # if both the same, then it's like having linear scaling turned off        
        self.sampler_config.params.guider_config.params.scale_min = cfg_scale_end
        self.sampler_config.params.guider_config.params.scale = cfg_scale_start


        self.sampler_config.params.restore_cfg = restoration_scale
        self.sampler_config.params.s_churn = s_churn
        self.sampler_config.params.s_noise = s_noise
        self.sampler = instantiate_from_config(self.sampler_config)

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        # ==============================================================
        # first check if option to skip it has been set or not.
        # if not....
        if not skip_denoise_stage:          
            # denoise the image
            # Apply intelligent artifact removal using specialized VAE denoise encoder
            # This learned process identifies and removes compression artifacts and noise
            # while preserving important image details, preventing the main model from 
            # mistakenly enhancing these artifacts as legitimate features
            _z = self.encode_first_stage_with_denoise(img, use_sample=False)
            x_stage1 = self.decode_first_stage(_z)  # this is the decoded denoised image in pixel space.              
            z_stage1 = self.encode_first_stage(x_stage1)
        else:
            print(f"Skipping Stage 1 Denoise.", color.ORANGE)
            _z = self.encode_first_stage(img)
            x_stage1 = self.decode_first_stage(_z)
            z_stage1 = self.encode_first_stage(img)

        # !>>> Pass the string prompt directly to prepare_condition
        c, uc = self.prepare_condition(_z, prompt, p_p, n_p)


        denoiser = lambda input, sigma, c, control_scale: self.denoiser(
            self.model, input, sigma, c, control_scale, **kwargs
        )

        noised_z = torch.randn_like(_z).to(_z.device)

        # sgm/modules/diffusionmodules/sampling.py
        _samples = self.sampler(denoiser, noised_z, cond=c, uc=uc, x_center=z_stage1, control_scale=control_scale_end, control_scale_start=control_scale_start)
        
        samples = self.decode_first_stage(_samples)
        if color_fix_type == 'Wavelet':
            samples = wavelet_reconstruction(samples, x_stage1)
        elif color_fix_type == 'AdaIn':
            samples = adaptive_instance_normalization(samples, x_stage1)
        return samples

    def init_tile_vae(self, encoder_tile_size=512, decoder_tile_size=256):
        print(f"Current function: {inspect.currentframe().f_code.co_name}()", color.ORANGE)
        self.first_stage_model.denoise_encoder.original_forward = self.first_stage_model.denoise_encoder.forward
        self.first_stage_model.encoder.original_forward = self.first_stage_model.encoder.forward
        self.first_stage_model.decoder.original_forward = self.first_stage_model.decoder.forward
        self.first_stage_model.denoise_encoder.forward = VAEHook(
            self.first_stage_model.denoise_encoder, encoder_tile_size, is_decoder=False, fast_decoder=False,
            fast_encoder=False, color_fix=False, to_gpu=True)
        self.first_stage_model.encoder.forward = VAEHook(
            self.first_stage_model.encoder, encoder_tile_size, is_decoder=False, fast_decoder=False,
            fast_encoder=False, color_fix=False, to_gpu=True)
        self.first_stage_model.decoder.forward = VAEHook(
            self.first_stage_model.decoder, decoder_tile_size, is_decoder=True, fast_decoder=False,
            fast_encoder=False, color_fix=False, to_gpu=True)

    # ========================================================================================
    #  prepare the conditioning inputs that guide the diffusion model during inference
    def prepare_condition(self, _z, prompt, p_p, n_p):
        print(f"Current function: {inspect.currentframe().f_code.co_name}()", color.ORANGE)

        # !>>> Get the batch size from the image tensor
        N = len(_z)

        batch = {}
        batch['original_size_as_tuple'] = torch.tensor([1024, 1024]).repeat(N, 1).to(_z.device)
        batch['crop_coords_top_left'] = torch.tensor([0, 0]).repeat(N, 1).to(_z.device)
        batch['target_size_as_tuple'] = torch.tensor([1024, 1024]).repeat(N, 1).to(_z.device)
        batch['aesthetic_score'] = torch.tensor([9.0]).repeat(N, 1).to(_z.device)
        batch['control'] = _z

        batch_uc = copy.deepcopy(batch)

        # !>>> Use the negative prompt string directly
        batch_uc['txt'] = [n_p]
                
        # batch_uc['txt'] = [n_p for _ in prompt]

        # !>>> Create the positive prompt by concatenating prompt and p_p
        batch['txt'] = [''.join([prompt, p_p])]

        with torch.amp.autocast('cuda', dtype=self.ae_dtype):
            c, uc = self.conditioner.get_unconditional_conditioning(batch, batch_uc)

        # if not isinstance(prompt[0], list):
        #     batch['txt'] = [''.join([_p, p_p]) for _p in prompt]
        #     with torch.amp.autocast('cuda', dtype=self.ae_dtype):
        #         c, uc = self.conditioner.get_unconditional_conditioning(batch, batch_uc)
        # else:
        #     assert len(prompt) == 1, 'Support bs=1 only for local prompt conditioning.'
        #     p_tiles = prompt[0]
        #     c = []
        #     for i, p_tile in enumerate(p_tiles):
        #         batch['txt'] = [''.join([p_tile, p_p])]
        #         with torch.cuda.amp.autocast(dtype=self.ae_dtype):
        #             if i == 0:
        #                 _c, uc = self.conditioner.get_unconditional_conditioning(batch, batch_uc)
        #             else:
        #                 _c, _ = self.conditioner.get_unconditional_conditioning(batch, None)
        #         c.append(_c)
        
        return c, uc


if __name__ == '__main__':
    from SUPIR.util import create_model, load_state_dict

    model = create_model('../../options/dev/SUPIR_paper_version.yaml')

    SDXL_CKPT = '/opt/data/private/AIGC_pretrain/SDXL_cache/sd_xl_base_1.0_0.9vae.safetensors'
    SUPIR_CKPT = '/opt/data/private/AIGC_pretrain/SUPIR_cache/SUPIR-paper.ckpt'

    print(f"SDXL_CKPT = {SDXL_CKPT}", color.RED)

    model.load_state_dict(load_state_dict(SDXL_CKPT), strict=False)
    model.load_state_dict(load_state_dict(SUPIR_CKPT), strict=False)
    model = model.cuda()

    x = torch.randn(1, 3, 512, 512).cuda()
    p = ['a professional, detailed, high-quality photo']
    samples = model.batchify_sample(x, p, num_steps=50, restoration_scale=4.0, s_churn=0, cfg_scale=4.0, seed=-1, num_samples=1)
