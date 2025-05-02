Modification of [SUPIR](https://github.com/Fanghua-Yu/SUPIR) repository.

- Removed the LLaVA implementation. 
- Added safetensors support. 
- Updated dependencies. 





---
## 🔧 Dependencies and Installation

1. Clone repo
    ```bash
    git clone https://github.com/yushan777/SUPIR.git
    cd SUPIR
    ```

2. Install dependent packages
    ```bash
        python3 -m venv venv
        source venv/bin/activate
        pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
        pip install -r requirements.txt
    ```

3. Download Checkpoints


#### Dependent Models
* [CLIP Encoder-1](https://huggingface.co/yushan777/SUPIR/resolve/main/CLIP1/clip-vit-large-patch14/safetensors/clip-vit-large-patch14.safetensors)
* [CLIP Encoder-2](https://huggingface.co/yushan777/SUPIR/resolve/main/CLIP2/CLIP-ViT-bigG-14-laion2B-39B-b160k/safetensors/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors)
* [Juggernaut-XL_v9_RunDiffusionPhoto_v2](https://huggingface.co/yushan777/SUPIR/resolve/main/SDXL/juggernautXL_v9Rundiffusionphoto2.safetensors)


#### Models we provided:
* `SUPIR-v0Q`: [FP16](https://huggingface.co/yushan777/SUPIR/resolve/main/SUPIR/SUPIR-v0Q_fp16.safetensors)
* `SUPIR-v0F`: [FP16](https://huggingface.co/yushan777/SUPIR/resolve/main/SUPIR/SUPIR-v0F_fp16.safetensors)

* `SUPIR-v0Q`: [FP32](https://huggingface.co/yushan777/SUPIR/resolve/main/SUPIR/SUPIR-v0Q_fp32.safetensors)
* `SUPIR-v0F`: [FP32](https://huggingface.co/yushan777/SUPIR/resolve/main/SUPIR/SUPIR-v0F_fp32.safetensors)

SUPIR-v0Q : Default training settings with paper. High generalization and high image quality in most cases.
SUPIR-v0F : Training with light degradation settings. Stage1 encoder of `SUPIR-v0F` remains more details when facing light degradations.

4. Edit Custom Path for Checkpoints
    ```
    * [options/SUPIR_v0.yaml] --> SDXL_CKPT, SUPIR_CKPT_Q, SUPIR_CKPT_F. CLIP1, CLIP2
    ```
---

## ⚡ Quick Inference
### Val Dataset
RealPhoto60: [Baidu Netdisk](https://pan.baidu.com/s/1CJKsPGtyfs8QEVCQ97voBA?pwd=aocg), [Google Drive](https://drive.google.com/drive/folders/1yELzm5SvAi9e7kPcO_jPp2XkTs4vK6aR?usp=sharing)

### Usage of SUPIR
```Shell
Usage: 
-- python test.py [options] 
-- python gradio_demo.py [interactive options]

--img_dir                Input folder.
--save_dir               Output folder.
--upscale                Upsampling ratio of given inputs. Default: 1
--SUPIR_sign             Model selection. Default: 'Q'; Options: ['F', 'Q']
--seed                   Random seed. Default: 1234
--min_size               Minimum resolution of output images. Default: 1024
--edm_steps              Number of steps for EDM Sampling Scheduler. Default: 50
--s_stage1               Control Strength of Stage1. Default: -1 (negative means invalid)
--s_churn                Original hy-param of EDM. Default: 5
--s_noise                Original hy-param of EDM. Default: 1.003
--s_cfg                  Classifier-free guidance scale for prompts. Default: 7.5
--s_stage2               Control Strength of Stage2. Default: 1.0
--num_samples            Number of samples for each input. Default: 1
--a_prompt               Additive positive prompt for all inputs. 
    Default: 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, 
    hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme
     meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.'
--n_prompt               Fixed negative prompt for all inputs. 
    Default: 'painting, oil painting, illustration, drawing, art, sketch, oil painting, 
    cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, 
    low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth'
--color_fix_type         Color Fixing Type. Default: 'Wavelet'; Options: ['None', 'AdaIn', 'Wavelet']
--linear_CFG             Linearly (with sigma) increase CFG from 'spt_linear_CFG' to s_cfg. Default: False
--linear_s_stage2        Linearly (with sigma) increase s_stage2 from 'spt_linear_s_stage2' to s_stage2. Default: False
--spt_linear_CFG         Start point of linearly increasing CFG. Default: 1.0
--spt_linear_s_stage2    Start point of linearly increasing s_stage2. Default: 0.0
--ae_dtype               Inference data type of AutoEncoder. Default: 'bf16'; Options: ['fp32', 'bf16']
--diff_dtype             Inference data type of Diffusion. Default: 'fp16'; Options: ['fp32', 'fp16', 'bf16']
```

### Python Script
```Shell
python3 test.py --img_path 'input/bottle.png' --save_dir ./output --SUPIR_sign Q --upscale 2 --use_tile_vae --loading_half_params

# Seek for best quality for most cases
python3 test.py --img_path 'input/bottle.png' --save_dir ./output --SUPIR_sign Q --upscale 2
# for light degradation and high fidelity
python3 test.py --img_path 'input/bottle.png' --save_dir ./output --SUPIR_sign F --upscale 2 --s_cfg 4.0 --linear_CFG
```







---

## BibTeX
    @misc{yu2024scaling,
      title={Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild}, 
      author={Fanghua Yu and Jinjin Gu and Zheyuan Li and Jinfan Hu and Xiangtao Kong and Xintao Wang and Jingwen He and Yu Qiao and Chao Dong},
      year={2024},
      eprint={2401.13627},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }

---

## 📧 Contact
If you have any question, please email `fanghuayu96@gmail.com` or `jinjin.gu@suppixel.ai`.

---
## Non-Commercial Use Only Declaration
The SUPIR ("Software") is made available for use, reproduction, and distribution strictly for non-commercial purposes. For the purposes of this declaration, "non-commercial" is defined as not primarily intended for or directed towards commercial advantage or monetary compensation.

By using, reproducing, or distributing the Software, you agree to abide by this restriction and not to use the Software for any commercial purposes without obtaining prior written permission from Dr. Jinjin Gu.

This declaration does not in any way limit the rights under any open source license that may apply to the Software; it solely adds a condition that the Software shall not be used for commercial purposes.

IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

For inquiries or to obtain permission for commercial use, please contact Dr. Jinjin Gu (jinjin.gu@suppixel.ai).
