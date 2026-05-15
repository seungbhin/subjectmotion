export CUDA_LAUNCH_BLOCKING=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_SHM_DISABLE=1

motion_video="person_playing_flute_temp_4"
subject_image="bear_plushie"

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
MOTION_HEAD_TYPES="./outputs/train/head_lora/motion/${motion_video}/head_types.pt"
SUBJECT_CKPT="./outputs/train/head_lora/subject/${subject_image}/subject/lightning_logs/ \
              version_1/checkpoints/epoch=8-step=900.ckpt"

# ── 설정 출력 ─────────────────────────────────────────────────────────────────
echo "SUBJECT_CKPT : ${SUBJECT_CKPT}"
python3 -c "
import torch
sd = torch.load('${SUBJECT_CKPT}', map_location='cpu', weights_only=False)
if 'state_dict' in sd: sd = sd['state_dict']
seen = set()
for k in sorted(sd):
    parts = k.split('.', 2)
    short = parts[2] if len(parts) == 3 else k
    if short not in seen:
        seen.add(short)
        print(f'  {short}')
"
echo ""

# ── 적용할 projection 선택 ────────────────────────────────────────────────────
# 비워두면("") 전체(q,k,v,o) 적용
SUBJECT_PROJECTIONS="q,k,v,o" 
PROMPT="A sks monster toy is stomping along a foggy industrial dock at night, cranes looming overhead and ships moored in the dark water"

# ── Subject 추론 ───────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/model_training/infer_head_lora.py \
  --model_paths \
    "./ckpts/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
    "./ckpts/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
    "./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
  --head_types_path "${MOTION_HEAD_TYPES}" \
  --lora_ckpt_path  "${SUBJECT_CKPT}" \
  --mode subject \
  --lora_rank 32 \
  --lora_alpha 32.0 \
  --lora_scale 1.0 \
  --subject_projections "${SUBJECT_PROJECTIONS}" \
  --prompt "${PROMPT}" \
  --height 320 --width 576 \
  --num_inference_steps 50 --cfg_scale 5.0 \
  --seed 42 \
  --output_path "./outputs/inference/head_lora/subject/${subject_image}"

