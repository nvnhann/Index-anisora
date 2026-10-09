#!/bin/bash
# Usage: ./run_gen.sh [prompt_list] [size] [frame_num]
# prompt_list format: "<prompt>@@<image path>&&" per line (xem data/inference_any.txt)
cd "$(dirname "$0")"
PROMPTS=${1:-data/inference_any.txt}
SIZE=${2:-832*480}      # 1280*720 cần nhiều VRAM hơn
FRAMES=${3:-81}
python generate_txt_new.py \
  --task i2v-A14B --size "$SIZE" --frame_num "$FRAMES" \
  --ckpt_dir Index-anisora-V3.2 \
  --ckpt_dir_lowname low_noise_model --ckpt_dir_highname high_noise_model \
  --prompt_list "$PROMPTS" --save_dir outputs \
  --sample_steps 8 --sample_shift 5 --sample_guide_scale 1 --base_seed 4096 \
  --offload_model True --t5_cpu --convert_model_dtype
