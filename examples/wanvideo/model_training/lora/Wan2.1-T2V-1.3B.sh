accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /home/xuxc/data/xuxc/datasets/VideoCustom/VideoCustom/combined_subject \
  --dataset_metadata_path /home/xuxc/data/xuxc/datasets/VideoCustom/VideoCustom/combined_subject/dog2/metadata.csv \
  --height 512 \
  --width 512 \
  --dataset_repeat 100 \
  --model_paths '[
    "/home/xuxc/data/xuxc/model/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    "/home/xuxc/data/xuxc/model/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    "/home/xuxc/data/xuxc/model/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
  ]' \
  --learning_rate 1e-4 \
  --num_epochs 10 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-T2V-1.3B_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 33 \
  --data_file_keys "video,mask"