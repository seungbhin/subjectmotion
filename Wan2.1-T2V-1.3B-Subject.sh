# Blackwell GPU (RTX PRO 6000) 兼容性环境变量
export CUDA_LAUNCH_BLOCKING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_BUFFSIZE=2097152
export NCCL_P2P_LEVEL=0
export NCCL_SHM_DISABLE=1

CUDA_VISIBLE_DEVICES="0,1" accelerate launch --gpu_ids "0,1" --num_processes 2 examples/wanvideo/model_training/train.py \
  --dataset_base_path datasets/subject_customized \
  --dataset_metadata_path datasets/subject_customized/dog/metadata.csv \
  --height 512 \
  --width 512 \
  --dataset_repeat 100 \
  --model_paths '[
    "./ckpts/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    "./ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    "./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
  ]' \
  --learning_rate 1e-4 \
  --num_epochs 4 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./outputs/train/subject/dog" \
  --lora_base_model "dit" \
  --spatial_lora_target_modules "q,k,ffn.0" \
  --train_type "Subject" \
  --spatial_lora_rank 32 \
  --data_file_keys "video,mask" \
  --test_prompts '["A sks dog is running through a dense forest, with sunlight streaming through the tall trees, leaves scattering in the air, and the ground covered in soft moss and fallen branches.", "A sks dog is walking calmly under the glowing aurora in the night sky, with shimmering green and purple lights dancing above, snow-covered ground reflecting the colors, and a serene silence surrounding the scene."]' \
  --sample_height 480 \
  --sample_width 832  \
  --encoder_type "dinov2-vit-g" \
  --proj_diff 0.05