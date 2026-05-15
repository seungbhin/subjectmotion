#!/bin/bash
# Run HeadLoRA motion training for all classes sequentially.

export CUDA_LAUNCH_BLOCKING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_BUFFSIZE=2097152
export NCCL_P2P_LEVEL=0
export NCCL_SHM_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

DATASET_BASE_PATH="datasets/motion_customized/video"
CKPT_BASE="./ckpts/Wan2.1-T2V-1.3B"
OUTPUT_BASE="./outputs/train/head_lora/motion/20260515"

SPATIAL_TARGETS="q_spatial,k_spatial,v_spatial,o_spatial"
TEMPORAL_TARGETS="q_temporal,k_temporal,v_temporal,o_temporal"
num_temporal_heads=6
stage1_start_epoch=4
learning_rate=1e-4
max_epochs=20
lora_rank=64
lora_alpha=64.0
lora_rank_spatial=32
lora_alpha_spatial=32.0
lora_rank_temporal=64
lora_alpha_temporal=64.0

run_one() {
    local CLASS="$1"
    local DATASET_METADATA="${DATASET_BASE_PATH}/${CLASS}/metadata.csv"
    local OUTPUT_PATH="${OUTPUT_BASE}/${CLASS}"
    local HEAD_TYPES_PATH="${OUTPUT_PATH}/head_types.pt"

    echo "============================================================"
    echo " Class: ${CLASS}"
    echo "============================================================"

    # ── 학습 설정 저장 ──────────────────────────────────────────────
    mkdir -p "${OUTPUT_PATH}"
    cat > "${OUTPUT_PATH}/train_config.txt" <<EOF
motion_video        = ${CLASS}
SPATIAL_TARGETS     = ${SPATIAL_TARGETS}
TEMPORAL_TARGETS    = ${TEMPORAL_TARGETS}
num_temporal_heads  = ${num_temporal_heads}
stage1_start_epoch  = ${stage1_start_epoch}
learning_rate       = ${learning_rate}
max_epochs          = ${max_epochs}
lora_rank           = ${lora_rank}
lora_alpha          = ${lora_alpha}
lora_rank_spatial   = ${lora_rank_spatial}
lora_alpha_spatial  = ${lora_alpha_spatial}
lora_rank_temporal  = ${lora_rank_temporal}
lora_alpha_temporal = ${lora_alpha_temporal}
EOF

    # ── Step 1: Head analysis ────────────────────────────────────────
    CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/model_training/train_head_lora.py \
      --task head_analysis \
      --model_paths \
        "${CKPT_BASE}/diffusion_pytorch_model.safetensors" \
        "${CKPT_BASE}/models_t5_umt5-xxl-enc-bf16.pth" \
        "${CKPT_BASE}/Wan2.1_VAE.pth" \
      --dataset_base_path "${DATASET_BASE_PATH}" \
      --dataset_metadata_path "${DATASET_METADATA}" \
      --height 320 --width 576 --num_frames 17 \
      --num_temporal_heads ${num_temporal_heads} \
      --n_analysis_samples 1 \
      --output_path "${OUTPUT_PATH}" || { echo "[ERROR] head_analysis failed: ${CLASS}"; return 1; }

    # ── Step 2: Cache latents ────────────────────────────────────────
    CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/model_training/train_head_lora.py \
      --task cache_latents \
      --mode motion \
      --model_paths \
        "${CKPT_BASE}/diffusion_pytorch_model.safetensors" \
        "${CKPT_BASE}/models_t5_umt5-xxl-enc-bf16.pth" \
        "${CKPT_BASE}/Wan2.1_VAE.pth" \
      --dataset_base_path "${DATASET_BASE_PATH}" \
      --dataset_metadata_path "${DATASET_METADATA}" \
      --height 320 --width 576 --num_frames 49 \
      --output_path "${OUTPUT_PATH}" || { echo "[ERROR] cache_latents failed: ${CLASS}"; return 1; }

    # ── Step 3: Train ────────────────────────────────────────────────
    CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/model_training/train_head_lora.py \
      --task train \
      --mode motion \
      --model_paths \
        "${CKPT_BASE}/diffusion_pytorch_model.safetensors" \
        "${CKPT_BASE}/models_t5_umt5-xxl-enc-bf16.pth" \
        "${CKPT_BASE}/Wan2.1_VAE.pth" \
      --dataset_base_path "${DATASET_BASE_PATH}" \
      --dataset_metadata_path "${DATASET_METADATA}" \
      --head_types_path "${HEAD_TYPES_PATH}" \
      --output_path "${OUTPUT_PATH}" \
      --height 320 --width 576 --num_frames 49 \
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
      --training_strategy auto || { echo "[ERROR] train failed: ${CLASS}"; return 1; }

    echo "[DONE] ${CLASS}"
}

# run_one "bear_walking"
# run_one "boat_sailing"
# run_one "bus_traveling"
# run_one "dog_walking"
# run_one "mallard_flying"
# run_one "person_dancing"
# run_one "person_lifting_barbell"
# run_one "person_playing_cello"
# run_one "person_playing_flute"
# run_one "person_twirling"
# run_one "person_walking"
run_one "train_turning"

echo "============================================================"
echo " All motion training completed."
echo "============================================================"
