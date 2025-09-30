source venv/bin/activate

python3 run_supir_cli.py \
--img_path input/woman-low-res-sq.jpg \
--upscale 2 \
--SUPIR_sign 'Q' \
--sampler_mode 'TiledRestoreEDMSampler' \
--seed 1234567891 \
--tile_size 128 \
--tile_stride 64 




