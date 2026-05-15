# Blackwell GPU (RTX PRO 6000) 
export CUDA_LAUNCH_BLOCKING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_BUFFSIZE=2097152
export NCCL_P2P_LEVEL=0
export NCCL_SHM_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

subject_image="bear_plushie"
motion_video="person_playing_flute"
# ── 경로 설정 ──────────────────────────────────────────────────────────────────
DATASET_BASE_PATH="datasets/subject_customized/${subject_image}"
DATASET_METADATA_PATH="${DATASET_BASE_PATH}/metadata.csv"
OUTPUT_PATH="./outputs/train/head_lora/subject/${subject_image}/${subject_image}_${motion_video}"
HEAD_TYPES_PATH="/home/sbjeon/workspace/SMRABooth/outputs/train/head_lora/motion/${motion_video}/head_types.pt"

# ── LoRA 적용 projection 선택 ──────────────────────────────────────────────
# 비워두면 전체(q,k,v,o) 적용
SPATIAL_TARGETS="q_spatial, k_spatial, v_spatial, o_spatial" 
num_temporal_heads=6
learning_rate=5e-5
max_epochs=50

# region Step 1: Head analysis
# ── Step 1: Head analysis (motion head_types 재사용 → 생략) ──────────────
# motion/skateboarding/head_types.pt를 HEAD_TYPES_PATH로 직접 지정하므로 별도 실행 불필요
# CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/model_training/train_head_lora.py \
#   --task head_analysis \
#   --model_paths \
#     "./ckpts/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
#     "./ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
#     "./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
#   --dataset_base_path "${DATASET_BASE_PATH}" \
#   --dataset_metadata_path "${DATASET_METADATA_PATH}" \
#   --height 320 \
#   --width 576 \
#   --num_temporal_heads 6 \
#   --n_analysis_samples 5 \
#   --output_path "${OUTPUT_PATH}"
# endregion

#── Step 2: Cache latents (subject 모드: 프레임별 개별 latent 저장) ────────
CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/model_training/train_head_lora.py \
  --task cache_latents \
  --mode subject \
  --model_paths \
    "./ckpts/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
    "./ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
    "./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --height 512 \
  --width 512 \
  --output_path "${OUTPUT_PATH}"

# ── Step 3: Train spatial head LoRA (외형 학습) ────────────────────────────
CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/model_training/train_head_lora.py \
  --task train \
  --mode subject \
  --model_paths \
    "./ckpts/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
    "./ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
    "./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --head_types_path "${HEAD_TYPES_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --height 512 \
  --width 512 \
  --learning_rate ${learning_rate} \
  --max_epochs ${max_epochs} \
  --steps_per_epoch 100 \
  --lora_rank 32 \
  --lora_alpha 32.0 \
  --spatial_targets "${SPATIAL_TARGETS}" \
  --use_gradient_checkpointing \
  --dataloader_num_workers 2 \
  --training_strategy auto