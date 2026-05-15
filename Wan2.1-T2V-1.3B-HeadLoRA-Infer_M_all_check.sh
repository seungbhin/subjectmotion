#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_SHM_DISABLE=1

motion_video="bear_walking"

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
MOTION_HEAD_TYPES="./outputs/train/head_lora/motion/20260515/${motion_video}/head_types.pt"
CKPT_DIR="./outputs/train/head_lora/motion/20260515/${motion_video}"

MOTION_PROJECTIONS="q,k,v,o"
SEED=42
PROMPT="A dog is vvt walking on the beach"

# ── 체크포인트 탐색 ────────────────────────────────────────────────────────────
mapfile -t CKPTS < <(find "${CKPT_DIR}" -name "*.ckpt" | sort)

if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "❌ No checkpoints found in ${CKPT_DIR}"
    exit 1
fi

echo "Found ${#CKPTS[@]} checkpoint(s):"
for c in "${CKPTS[@]}"; do echo "  $c"; done
echo ""

# ── 각 체크포인트마다 추론 ─────────────────────────────────────────────────────
for MOTION_CKPT in "${CKPTS[@]}"; do
    ckpt_name=$(basename "${MOTION_CKPT}" .ckpt)
    OUTPUT_PATH="./outputs/inference/head_lora/motion/20260515/${motion_video}/${ckpt_name}"

    echo "============================================================"
    echo " CKPT : ${MOTION_CKPT}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/model_training/infer_head_lora.py \
      --model_paths \
        "./ckpts/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
        "./ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
        "./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
      --head_types_path "${MOTION_HEAD_TYPES}" \
      --lora_ckpt_path  "${MOTION_CKPT}" \
      --mode motion \
      --lora_rank_spatial  32 \
      --lora_rank_temporal 64 \
      --lora_scale 1.0 \
      ${MOTION_PROJECTIONS:+--motion_projections "${MOTION_PROJECTIONS}"} \
      --prompt "${PROMPT}" \
      --height 320 --width 576 --num_frames 49 \
      --num_inference_steps 50 --cfg_scale 6.0 \
      --seed ${SEED} \
      --output_path "${OUTPUT_PATH}"

    mkdir -p "${OUTPUT_PATH}"
    cat > "${OUTPUT_PATH}/infer_config.txt" <<EOF
motion_video       = ${motion_video}
MOTION_HEAD_TYPES  = ${MOTION_HEAD_TYPES}
MOTION_CKPT        = ${MOTION_CKPT}
MOTION_PROJECTIONS = ${MOTION_PROJECTIONS}
PROMPT             = ${PROMPT}
seed               = ${SEED}
height             = 320
width              = 576
num_frames         = 49
num_inference_steps= 50
cfg_scale          = 6.0
lora_scale         = 1.0
lora_rank_spatial  = 32
lora_rank_temporal = 64
EOF
    echo "[INFO] infer_config saved: ${OUTPUT_PATH}/infer_config.txt"
done

echo "============================================================"
echo " All checkpoints done."
echo "============================================================"