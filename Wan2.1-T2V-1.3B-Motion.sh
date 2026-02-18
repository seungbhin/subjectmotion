# Blackwell GPU (RTX PRO 6000) 兼容性环境变量
export CUDA_LAUNCH_BLOCKING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_BUFFSIZE=2097152
export NCCL_P2P_LEVEL=0
export NCCL_SHM_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

CUDA_VISIBLE_DEVICES="0,1" accelerate launch --gpu_ids "0,1" --num_processes 2 examples/wanvideo/model_training/train.py \
  --dataset_base_path datasets/motion_customized/video \
  --dataset_metadata_path datasets/motion_customized/video/skateboarding/metadata.csv\
  --height 320\
  --width 576 \
  --dataset_repeat 100 \
  --model_paths '[
    "./ckpts/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    "./ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    "./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
  ]' \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --num_frames 49 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --use_gradient_checkpointing_offload \
  --output_path "./outputs/train/motion/skateboarding" \
  --lora_base_model "dit" \
  --spatial_lora_target_modules "q,k,ffn.0" \
  --spatial_lora_rank 32 \
  --temporal_lora_target_modules "v,o,ffn.0,ffn.2" \
  --temporal_lora_rank 64 \
  --train_type "Motion" \
  --data_file_keys "video" \
  --test_prompts '[
        "A firefighter is vvt skateboarding down the street, with the firetruck parked nearby and curious onlookers watching.",
        "An astronaut is vvt skateboarding across the moon dusty surface, performing tricks in low gravity. The barren lunar landscape stretches around, with Earth glowing brightly in the star-filled sky above."
        ]' \
  --sample_height 480 \
  --sample_width 832 \
  --proj_diff 1.0