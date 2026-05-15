"""
infer_head_lora.py

Inference for head-specific LoRA trained with train_head_lora.py.

Modes
-----
subject : spatial LoRA only (subject appearance)
motion  : spatial + temporal LoRA from one motion checkpoint
combine : spatial LoRA from subject ckpt + temporal LoRA from motion ckpt
          --lora_ckpt_path       subject checkpoint  (spatial keys only)
          --lora_ckpt_path_extra motion  checkpoint  (temporal keys only used)
          --lora_rank_spatial / --lora_rank_temporal for different ranks
"""

import os, sys, argparse, random

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_SCRIPT_DIR, "..", "..", "..")
sys.path.insert(0, os.path.abspath(_ROOT))

import torch
from diffsynth import save_video
from peft import LoraConfig, inject_adapter_in_model
from train_head_lora import _build_pipe, upgrade_self_attention, split_attention_heads, _enable_all_adapters

SPATIAL_TARGETS  = ["q_spatial",  "k_spatial",  "v_spatial",  "o_spatial"]
TEMPORAL_TARGETS = ["q_temporal", "k_temporal", "v_temporal", "o_temporal"]

_PROJ_MAP = {"q": ("q_spatial", "q_temporal"),
             "k": ("k_spatial", "k_temporal"),
             "v": ("v_spatial", "v_temporal"),
             "o": ("o_spatial", "o_temporal")}


_FFN_TARGETS = {"ffn.0", "ffn.2"}

def _parse_projections(raw: str, side: str) -> list:
    """'q,k,o,ffn.0' → ['q_spatial','k_spatial','o_spatial','ffn.0'] (side='spatial' or 'temporal')"""
    idx = 0 if side == "spatial" else 1
    result = []
    for p in raw.split(","):
        p = p.strip()
        if p in _PROJ_MAP:
            result.append(_PROJ_MAP[p][idx])
        elif p in _FFN_TARGETS:
            result.append(p)
    return result

DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


# ---------------------------------------------------------------------------
# LoRA loading
# ---------------------------------------------------------------------------

def _load_ckpt(path: str) -> dict:
    """Load a checkpoint saved by on_save_checkpoint (flat LoRA weight dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Lightning sometimes wraps inside 'state_dict' if not cleared properly
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]
    return ckpt


def _filter_keys(ckpt: dict, keep: str) -> dict:
    """Return only keys that contain `keep` (e.g. 'spatial' or 'temporal').

    With named adapters ('spatial'/'temporal'), ffn keys are naturally
    filtered: ffn.0.lora_A.temporal.weight contains 'temporal'.
    """
    return {k: v for k, v in ckpt.items() if keep in k}


def _apply_ckpt(model, ckpt: dict, label: str):
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    loaded = len(ckpt) - len(missing)
    print(f"[INFO] {label}: {loaded}/{len(ckpt)} params loaded "
          f"({len(missing)} missing, {len(unexpected)} unexpected)")


def build_and_load(
    model_paths: list,
    head_types_path: str,
    lora_ckpt_path: str,
    mode: str = "subject",
    lora_rank: int = 32,
    lora_rank_spatial: int = None,
    lora_rank_temporal: int = None,
    lora_ckpt_path_extra: str = None,
    lora_scale: float = 1.0,
    subject_projections: list = None,
    motion_projections: list = None,
) -> object:
    """Build pipeline, inject head-split LoRA, and load weights.
    lora_alpha is always set equal to lora_rank (scaling=1.0); use lora_scale to adjust strength.

    mode='combine': injects spatial LoRA (rank=lora_rank_spatial) and temporal
    LoRA (rank=lora_rank_temporal) separately, then loads spatial keys from
    lora_ckpt_path and temporal keys from lora_ckpt_path_extra.
    """
    pipe = _build_pipe(model_paths, device="cuda")
    upgrade_self_attention(pipe.dit)

    head_types = torch.load(head_types_path, map_location="cpu", weights_only=True)
    split_attention_heads(pipe.dit, head_types)

    # motion 모드: subject_projections 미지정 시 spatial 등록 안 함
    spa_default = [] if mode == "motion" else SPATIAL_TARGETS
    spa_targets = subject_projections if subject_projections is not None else spa_default
    tem_targets = motion_projections  if motion_projections  is not None else TEMPORAL_TARGETS
    print(f"[INFO] subject projections : {spa_targets}")
    print(f"[INFO] motion  projections : {tem_targets}")

    if mode == "subject":
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=float(lora_rank),
            init_lora_weights=True, target_modules=spa_targets,
        )
        pipe.dit = inject_adapter_in_model(lora_config, pipe.dit)
        _apply_ckpt(pipe.dit, _load_ckpt(lora_ckpt_path), "subject ckpt")

    elif mode == "motion":
        rank_spa = lora_rank_spatial  if lora_rank_spatial  is not None else lora_rank
        rank_tem = lora_rank_temporal if lora_rank_temporal is not None else lora_rank
        alpha_spa = float(rank_spa)
        alpha_tem = float(rank_tem)
        if spa_targets:
            pipe.dit = inject_adapter_in_model(LoraConfig(
                r=rank_spa, lora_alpha=alpha_spa,
                init_lora_weights=True, target_modules=spa_targets,
            ), pipe.dit, adapter_name="spatial")
        pipe.dit = inject_adapter_in_model(LoraConfig(
            r=rank_tem, lora_alpha=alpha_tem,
            init_lora_weights=True, target_modules=tem_targets,
        ), pipe.dit, adapter_name="temporal")
        _enable_all_adapters(pipe.dit)
        _apply_ckpt(pipe.dit, _load_ckpt(lora_ckpt_path), "motion ckpt")
        if lora_ckpt_path_extra is not None:
            _apply_ckpt(pipe.dit, _load_ckpt(lora_ckpt_path_extra), "motion extra ckpt")

    elif mode == "combine":
        rank_spa = lora_rank_spatial or lora_rank
        rank_tem = lora_rank_temporal or lora_rank
        alpha_spa = float(rank_spa)
        alpha_tem = float(rank_tem)

        # Inject spatial LoRA with "default" adapter (subject ckpt uses "default" key)
        spa_config = LoraConfig(
            r=rank_spa, lora_alpha=alpha_spa,
            init_lora_weights=True, target_modules=spa_targets,
        )
        pipe.dit = inject_adapter_in_model(spa_config, pipe.dit)

        # Inject temporal LoRA with "temporal" adapter name
        tem_config = LoraConfig(
            r=rank_tem, lora_alpha=alpha_tem,
            init_lora_weights=True, target_modules=tem_targets,
        )
        pipe.dit = inject_adapter_in_model(tem_config, pipe.dit, adapter_name="temporal")

        # Modules shared between spa/tem targets get two adapters; activate both
        _enable_all_adapters(pipe.dit)

        # Load spatial weights from subject checkpoint
        # subject ckpt keys: q_spatial.lora_A.default.weight → "spatial" in key ✓
        spa_ckpt = _filter_keys(_load_ckpt(lora_ckpt_path), "spatial")
        _apply_ckpt(pipe.dit, spa_ckpt, "subject ckpt (spatial keys)")

        # Load temporal weights from motion checkpoint
        # motion ckpt keys: q_temporal.lora_A.temporal.weight, ffn.0.lora_A.temporal.weight → "temporal" ✓
        if lora_ckpt_path_extra is None:
            raise ValueError("--lora_ckpt_path_extra (motion ckpt) required for combine mode")
        tem_ckpt = _filter_keys(_load_ckpt(lora_ckpt_path_extra), "temporal")
        _apply_ckpt(pipe.dit, tem_ckpt, "motion ckpt (temporal keys)")

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # Global LoRA scale
    if lora_scale != 1.0:
        for _, module in pipe.dit.named_modules():
            if hasattr(module, "scaling"):
                for k in module.scaling:
                    module.scaling[k] = lora_scale

    pipe.dit = pipe.dit.to(dtype=torch.bfloat16)
    return pipe


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Head-LoRA inference for Wan2.1")

    p.add_argument("--model_paths", nargs="+", required=True,
                   help="Paths to Wan2.1 model files (DiT, text encoder, VAE)")
    p.add_argument("--head_types_path", required=True,
                   help="Path to head_types.pt saved during head_analysis step")
    p.add_argument("--lora_ckpt_path", required=True,
                   help="Path to .ckpt saved by train_head_lora.py")
    p.add_argument("--lora_ckpt_path_extra", default=None,
                   help="combine/motion mode: motion checkpoint (temporal keys used)")
    p.add_argument("--mode", default="subject", choices=["subject", "motion", "combine"])
    p.add_argument("--lora_rank",          type=int,   default=32,
                   help="LoRA rank (fallback when spatial/temporal not specified)")
    p.add_argument("--lora_rank_spatial",  type=int,   default=None,
                   help="spatial LoRA rank")
    p.add_argument("--lora_rank_temporal", type=int,   default=None,
                   help="temporal LoRA rank")
    p.add_argument("--lora_scale", type=float, default=1.0,
                   help="Global LoRA scale multiplier applied at inference")
    p.add_argument("--subject_projections", default=None,
                   help="Comma-separated projections for subject (spatial) LoRA. "
                        "e.g. 'q,k,o'  Default: q,k,v,o")
    p.add_argument("--motion_projections", default=None,
                   help="Comma-separated projections for motion (temporal) LoRA. "
                        "e.g. 'v,o'  Default: q,k,v,o")

    p.add_argument("--prompt", required=True)
    p.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE)

    p.add_argument("--height",     type=int, default=320)
    p.add_argument("--width",      type=int, default=576)
    p.add_argument("--num_frames", type=int, default=49)

    p.add_argument("--num_inference_steps", type=int,   default=50)
    p.add_argument("--cfg_scale",           type=float, default=6.0)
    p.add_argument("--sigma_shift",         type=float, default=5.0)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--tiled",        action="store_true", default=True)
    p.add_argument("--output_path",  default="./outputs/inference/head_lora")
    p.add_argument("--fps",          type=int,   default=15)
    p.add_argument("--video_quality",type=int,   default=5)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    def _parse_proj(raw, side):
        if raw is None:
            return None
        return _parse_projections(raw, side)

    pipe = build_and_load(
        model_paths=args.model_paths,
        head_types_path=args.head_types_path,
        lora_ckpt_path=args.lora_ckpt_path,
        lora_ckpt_path_extra=args.lora_ckpt_path_extra,
        mode=args.mode,
        lora_rank=args.lora_rank,
        lora_rank_spatial=args.lora_rank_spatial,
        lora_rank_temporal=args.lora_rank_temporal,
        lora_scale=args.lora_scale,
        subject_projections=_parse_proj(args.subject_projections, "spatial"),
        motion_projections=_parse_proj(args.motion_projections, "temporal"),
    )
    pipe.enable_vram_management()

    seed = args.seed if args.seed is not None else random.randint(0, 10_000_000)
    print(f"[INFO] Seed: {seed}")

    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        sigma_shift=args.sigma_shift,
        seed=seed,
        tiled=args.tiled,
    )

    os.makedirs(args.output_path, exist_ok=True)
    ckpt_stem = os.path.splitext(os.path.basename(args.lora_ckpt_path))[0]
    out_file = os.path.join(args.output_path, f"{ckpt_stem}.mp4")
    save_video(video, out_file, fps=args.fps, quality=args.video_quality)
    print(f"[INFO] Saved: {out_file}")
