# Blackwell GPU (RTX PRO 6000) 兼容性환경변수
export CUDA_LAUNCH_BLOCKING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_BUFFSIZE=2097152
export NCCL_P2P_LEVEL=0
export NCCL_SHM_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL


motion_video="bear_walking"
DATASET_BASE_PATH="datasets/motion_customized/video"
DATASET_METADATA_PATH="datasets/motion_customized/video/${motion_video}/metadata.csv"
OUTPUT_PATH="./outputs/train/head_lora/motion/20260515/${motion_video}"
HEAD_TYPES_PATH="${OUTPUT_PATH}/head_types.pt"

# ── LoRA 적용 projection 선택 ──────────────────────────────────────────────
# 비워두면 전체(q,k,v,o) 적용
SPATIAL_TARGETS="q_spatial,k_spatial,v_spatial,o_spatial"  
TEMPORAL_TARGETS="q_temporal,k_temporal,v_temporal,o_temporal" 
num_temporal_heads=6
stage1_start_epoch=4  # epoch 0,1,2,3: spatial LoRA → epoch 4부터 temporal LoRA 추가
learning_rate=1e-4
max_epochs=20
lora_rank=64
lora_alpha=64.0
lora_rank_spatial=32
lora_alpha_spatial=32.0
lora_rank_temporal=64
lora_alpha_temporal=64.0

# ── 학습 설정 저장 ────────────────────────────────────────────────────────────
mkdir -p "${OUTPUT_PATH}"
cat > "${OUTPUT_PATH}/train_config.txt" <<EOF
motion_video       = ${motion_video}
SPATIAL_TARGETS    = ${SPATIAL_TARGETS}
TEMPORAL_TARGETS   = ${TEMPORAL_TARGETS}
num_temporal_heads = ${num_temporal_heads}
stage1_start_epoch = ${stage1_start_epoch}
learning_rate      = ${learning_rate}
max_epochs         = ${max_epochs}
lora_rank          = ${lora_rank}
lora_alpha         = ${lora_alpha}
lora_rank_spatial  = ${lora_rank_spatial}
lora_alpha_spatial = ${lora_alpha_spatial}
lora_rank_temporal = ${lora_rank_temporal}
lora_alpha_temporal = ${lora_alpha_temporal}
EOF
echo "[INFO] train_config saved: ${OUTPUT_PATH}/train_config.txt"

# ── Step 1: Head analysis (처음 한 번만 실행) ──────────────────────────────
CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/model_training/train_head_lora.py \
  --task head_analysis \
  --model_paths \
    "./ckpts/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
    "./ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
    "./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --height 320 \
  --width 576 \
  --num_frames 17 \
  --num_temporal_heads ${num_temporal_heads} \
  --n_analysis_samples 1 \
  --output_path "${OUTPUT_PATH}"

# ── Step 2: Cache latents
#   motion 모드: 전체 클립 latent(Stage 1용) + 프레임별 latent(Stage 0용) 동시 저장
CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/model_training/train_head_lora.py \
  --task cache_latents \
  --mode motion \
  --model_paths \
    "./ckpts/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
    "./ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
    "./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --height 320 \
  --width 576 \
  --num_frames 49 \
  --output_path "${OUTPUT_PATH}"

# ── Step 3: Train (2-stage per epoch)
#   Lightning epoch 0,2,4,... → Stage 0: spatial LoRA, 랜덤 1프레임
#   Lightning epoch 1,3,5,... → Stage 1: temporal LoRA, 전체 클립
#   --use_gradient_checkpointing \

CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/model_training/train_head_lora.py \
  --task train \
  --mode motion \
  --model_paths \
    "./ckpts/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
    "./ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
    "./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --head_types_path "${HEAD_TYPES_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --height 320 \
  --width 576 \
  --num_frames 49 \
  --learning_rate ${learning_rate} \
  --max_epochs ${max_epochs} \
  --stage1_start_epoch ${stage1_start_epoch} \
  --steps_per_epoch 100 \
  --lora_rank ${lora_rank} \
  --lora_alpha ${lora_alpha} \
  --lora_rank_spatial ${lora_rank_spatial} \
  --lora_alpha_spatial ${lora_alpha_spatial} \
  --lora_rank_temporal ${lora_rank_temporal} \
  --lora_alpha_temporal ${lora_alpha_temporal} \
  --spatial_targets  "${SPATIAL_TARGETS}" \
  --temporal_targets "${TEMPORAL_TARGETS}" \
  --dataloader_num_workers 2 \
  --training_strategy auto
