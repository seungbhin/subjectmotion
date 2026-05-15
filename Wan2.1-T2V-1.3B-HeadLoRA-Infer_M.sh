export CUDA_LAUNCH_BLOCKING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_SHM_DISABLE=1

motion_video="bear_walking"

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
MOTION_HEAD_TYPES="./outputs/train/head_lora/motion/20260515/${motion_video}/head_types.pt"
MOTION_CKPT="./outputs/train/head_lora/motion/20260515/${motion_video}/motion/lightning_logs/version_0/checkpoints/epoch=8-step=900.ckpt"
OUTPUT_PATH="./outputs/inference/head_lora/motion/20260515/${motion_video}"

# ── 적용할 projection 선택 ────────────────────────────────────────────────────
# 사용 가능: q, k, v, o (콤마로 나열)
# 비워두면("") 전체(q,k,v,o) 적용
MOTION_PROJECTIONS="q,k,v,o"         # motion(temporal) LoRA projection
SEED=42
PROMPT="A dog is vvt walking on the beach"

# ── 설정 출력 ─────────────────────────────────────────────────────────────────
echo "MOTION_CKPT : ${MOTION_CKPT}"
python3 -c "
import torch
sd = torch.load('${MOTION_CKPT}', map_location='cpu', weights_only=False)
if 'state_dict' in sd: sd = sd['state_dict']
# blocks.N 접두사 제거 후 고유 키만 출력
seen = set()
for k in sorted(sd):
    # blocks.0.ffn.0.lora_B.temporal.weight → ffn.0.lora_B.temporal.weight
    parts = k.split('.', 2)  # ['blocks', '0', 'rest']
    short = parts[2] if len(parts) == 3 else k
    if short not in seen:
        seen.add(short)
        print(f'  {short}')
"
echo ""

# ── Motion 추론 (단일) ─────────────────────────────────────────────────────────
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

# ── 추론 설정 저장 ─────────────────────────────────────────────────────────────
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