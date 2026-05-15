#!/bin/bash
# Run HeadLoRA subject training for all (subject, motion) pairs sequentially.
# cache_latents is run once per subject and reused across all motions.

export CUDA_LAUNCH_BLOCKING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_BUFFSIZE=2097152
export NCCL_P2P_LEVEL=0
export NCCL_SHM_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

SUBJECT_BASE="datasets/subject_customized"
MOTION_HEAD_BASE="./outputs/train/head_lora/motion/20260515"
OUTPUT_BASE="./outputs/train/head_lora/subject/20260515"
CKPT_BASE="./ckpts/Wan2.1-T2V-1.3B"

SPATIAL_TARGETS="q_spatial,k_spatial,v_spatial,o_spatial"
num_temporal_heads=6
learning_rate=5e-5
max_epochs=20
lora_rank=32
lora_alpha=32.0

# ── Subject 목록 (subject_customized 하위 디렉토리) ──────────────────────────
SUBJECTS=(
    "backpack"
    "bear_plushie"
    "book"
    "car"
    "cat"
    "clock"
    "dog"
    "instrument"
    "monster_toy"
    "pink_plushie"
    "sloth_plushie"
    "terracotta_warrior"
    "tortoise_plushie"
    "unicorn_toy"
    "wolf_plushie"
)

# ── Motion 목록 (motion_customized/video 하위 디렉토리) ──────────────────────
MOTIONS=(
    "bear_walking"
    "boat_sailing"
    "bus_traveling"
    "dog_walking"
    "mallard_flying"
    "person_dancing"
    "person_lifting_barbell"
    "person_playing_cello"
    "person_playing_flute"
    "person_twirling"
    "person_walking"
    "train_turning"
)

# ── Subject당 cache_latents 1회 실행 ─────────────────────────────────────────
cache_subject() {
    local SUBJECT="$1"
    local DATASET_BASE_PATH="${SUBJECT_BASE}/${SUBJECT}"
    local DATASET_METADATA_PATH="${DATASET_BASE_PATH}/metadata.csv"
    local CACHE_FLAG="${DATASET_BASE_PATH}/.cache_done"

    if [ -f "${CACHE_FLAG}" ]; then
        echo "[SKIP] cache already done: ${SUBJECT}"
        return 0
    fi

    echo "============================================================"
    echo " Cache latents: ${SUBJECT}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/model_training/train_head_lora.py \
      --task cache_latents \
      --mode subject \
      --model_paths \
        "${CKPT_BASE}/diffusion_pytorch_model.safetensors" \
        "${CKPT_BASE}/models_t5_umt5-xxl-enc-bf16.pth" \
        "${CKPT_BASE}/Wan2.1_VAE.pth" \
      --dataset_base_path "${DATASET_BASE_PATH}" \
      --dataset_metadata_path "${DATASET_METADATA_PATH}" \
      --height 512 --width 512 \
      --output_path "${DATASET_BASE_PATH}/cache_tmp" || { echo "[ERROR] cache_latents failed: ${SUBJECT}"; return 1; }

    touch "${CACHE_FLAG}"
}

# ── (Subject, Motion) 쌍 학습 ────────────────────────────────────────────────
train_one() {
    local SUBJECT="$1"
    local MOTION="$2"
    local DATASET_BASE_PATH="${SUBJECT_BASE}/${SUBJECT}"
    local DATASET_METADATA_PATH="${DATASET_BASE_PATH}/metadata.csv"
    local HEAD_TYPES_PATH="${MOTION_HEAD_BASE}/${MOTION}/head_types.pt"
    local OUTPUT_PATH="${OUTPUT_BASE}/${SUBJECT}/${SUBJECT}_${MOTION}"

    if [ ! -f "${HEAD_TYPES_PATH}" ]; then
        echo "[SKIP] head_types not found: ${HEAD_TYPES_PATH}"
        return 0
    fi

    echo "------------------------------------------------------------"
    echo " Train: ${SUBJECT} x ${MOTION}"
    echo "------------------------------------------------------------"

    mkdir -p "${OUTPUT_PATH}"
    cat > "${OUTPUT_PATH}/train_config.txt" <<EOF
subject_image      = ${SUBJECT}
motion_video       = ${MOTION}
HEAD_TYPES_PATH    = ${HEAD_TYPES_PATH}
SPATIAL_TARGETS    = ${SPATIAL_TARGETS}
learning_rate      = ${learning_rate}
max_epochs         = ${max_epochs}
lora_rank          = ${lora_rank}
lora_alpha         = ${lora_alpha}
EOF

    CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/model_training/train_head_lora.py \
      --task train \
      --mode subject \
      --model_paths \
        "${CKPT_BASE}/diffusion_pytorch_model.safetensors" \
        "${CKPT_BASE}/models_t5_umt5-xxl-enc-bf16.pth" \
        "${CKPT_BASE}/Wan2.1_VAE.pth" \
      --dataset_base_path "${DATASET_BASE_PATH}" \
      --dataset_metadata_path "${DATASET_METADATA_PATH}" \
      --head_types_path "${HEAD_TYPES_PATH}" \
      --output_path "${OUTPUT_PATH}" \
      --height 512 --width 512 \
      --learning_rate ${learning_rate} \
      --max_epochs ${max_epochs} \
      --steps_per_epoch 100 \
      --lora_rank ${lora_rank} \
      --lora_alpha ${lora_alpha} \
      --spatial_targets "${SPATIAL_TARGETS}" \
      --use_gradient_checkpointing \
      --dataloader_num_workers 2 \
      --training_strategy auto || { echo "[ERROR] train failed: ${SUBJECT} x ${MOTION}"; return 1; }

    echo "[DONE] ${SUBJECT} x ${MOTION}"
}

# ── 실행 ─────────────────────────────────────────────────────────────────────
for SUBJECT in "${SUBJECTS[@]}"; do
    cache_subject "${SUBJECT}" || continue

    for MOTION in "${MOTIONS[@]}"; do
        train_one "${SUBJECT}" "${MOTION}"
    done
done

echo "============================================================"
echo " All subject training completed."
echo "============================================================"
