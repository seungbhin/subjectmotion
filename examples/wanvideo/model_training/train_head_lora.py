"""
train_head_lora.py

Subject-Motion Customization for Wan2.1 using head-specific LoRA.

Head classification:
    0 = temporal head  (high w_t / w_s ratio -> motion)
    1 = spatial  head  (low  w_t / w_s ratio -> appearance)

Training modes
--------------
subject mode:
    Injects spatial head LoRA only.
    Trains on single images or single frames (appearance learning).

motion mode  (2-stage per epoch, following SMRABooth):
    Stage 0  — spatial  head LoRA, single random frame  (appearance of the motion subject)
    Stage 1  — temporal head LoRA, full video clip      (motion pattern learning)
    spatial LoRA is frozen during Stage 1.

Loss: pure MSE diffusion loss for all stages (no feature-align, no flow loss).

Usage
-----
# Step 1: compute head types once
python train_head_lora.py --task head_analysis \\
    --model_paths /path/to/wan2.1/* \\
    --dataset_base_path ./data \\
    --dataset_metadata_path ./data/metadata.csv \\
    --num_temporal_heads 6 --output_path ./output

# Step 2a: train subject LoRA
python train_head_lora.py --task train --mode subject \\
    --model_paths /path/to/wan2.1/* \\
    --dataset_base_path ./data --dataset_metadata_path ./data/metadata.csv \\
    --head_types_path ./output/head_types.pt --output_path ./output/subject

# Step 2b: train motion LoRA  (2-stage per epoch)
python train_head_lora.py --task train --mode motion \\
    --model_paths /path/to/wan2.1/* \\
    --dataset_base_path ./data --dataset_metadata_path ./data/metadata.csv \\
    --head_types_path ./output/head_types.pt --output_path ./output/motion
"""

import os, sys, json, math, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import pandas as pd
import numpy as np
from PIL import Image
from einops import rearrange
from tqdm import tqdm
import lightning as pl
import lightning.pytorch.callbacks as pl_callbacks
import torchvision

# ── SMRABooth path ──────────────────────────────────────────────────────────
# 현재 파일 디렉토리 경로를 받아, 3단게 위 경로를 파이썬 경로에 추가 
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_SCRIPT_DIR, "..", "..", "..")
sys.path.insert(0, os.path.abspath(_ROOT))

from peft import LoraConfig, inject_adapter_in_model
from diffsynth.models.utils import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_dit import WanModel, SelfAttention as _OrigSelfAttention
from diffsynth.models.wan_video_dit_head_lora import (
    SelfAttentionHeadLora, # head별로 분리된 attention 클래스
    classify_heads, # temporal/spatial head 분류 함수
    get_attention_mask, # attention mask 계산 함수
    flash_attention as _flash_attn,
    rope_apply as _rope_apply,
)

# ============================================================================
# Dataset helpers
# ============================================================================

class VideoDataset(torch.utils.data.Dataset):
    """Loads videos or images for subject/motion training."""

    def __init__(
        self,
        base_path: str,
        metadata_path: str,
        num_frames: int = 17,
        height: int = 480,
        width: int = 832,
        mode: str = "motion",  # "subject" or "motion"
    ):
        if metadata_path.endswith(".json"):
            with open(metadata_path) as f:
                self.data = json.load(f)
        else:
            meta = pd.read_csv(metadata_path)
            self.data = meta.to_dict("records")

        self.base_path = base_path
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.mode = mode

        self.frame_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def _crop_resize(self, img: Image.Image) -> Image.Image:
        W, H = img.size
        scale = max(self.width / W, self.height / H)
        img = torchvision.transforms.functional.resize(
            img, (round(H * scale), round(W * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return torchvision.transforms.functional.center_crop(img, (self.height, self.width))

    def _load_video(self, path: str):
        reader = imageio.get_reader(path)
        total = reader.count_frames()
        n = min(self.num_frames, total)
        indices = np.linspace(0, total - 1, n, dtype=int)
        frames = []
        for idx in indices:
            frame = Image.fromarray(reader.get_data(int(idx)))
            frames.append(self.frame_transform(self._crop_resize(frame)))
        reader.close()
        frames = torch.stack(frames)                      # (T, C, H, W)
        return rearrange(frames, "T C H W -> C T H W")   # (C, T, H, W)

    def _load_image(self, path: str):
        img = Image.open(path).convert("RGB")
        frame = self.frame_transform(self._crop_resize(img))
        return rearrange(frame, "C H W -> C 1 H W")      # (C, 1, H, W)

    def _is_image(self, path: str) -> bool:
        return path.lower().split(".")[-1] in ("jpg", "jpeg", "png", "webp")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        prompt = row.get("prompt", row.get("text", ""))
        file_key = next((k for k in ("video", "image", "file_name") if k in row), None)
        path = os.path.join(self.base_path, row[file_key]) if file_key else None
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        if self._is_image(path):
            video = self._load_image(path)
        else:
            video = self._load_video(path)

        return {"video": video, "prompt": prompt}


# ============================================================================
# Cached latent dataset
# ============================================================================

def _metadata_file_paths(metadata_path: str, base_path: str) -> list:
    """Return a list of absolute file paths from a metadata CSV or JSON.

    Supports column names: file_name, video, image.
    """
    if metadata_path.endswith(".json"):
        with open(metadata_path) as f:
            meta = json.load(f)
        return [os.path.join(base_path, r.get("file_name", r.get("video", r.get("image", "")))) for r in meta]
    else:
        meta = pd.read_csv(metadata_path)
        file_col = next((c for c in ("file_name", "video", "image") if c in meta.columns), None)
        if file_col is None:
            raise ValueError(f"No file column found in metadata. Columns: {list(meta.columns)}")
        return [os.path.join(base_path, fn) for fn in meta[file_col]]


# VAE 인코딩 결과를 .cache.pt 파일로 저장한 후, 학습 시 해당 파일에서 직접 텐서를 로드하여 사용하는 Dataset 클래스
class CachedLatentDataset(torch.utils.data.Dataset):
    def __init__(self, base_path: str, metadata_path: str, steps_per_epoch: int = 500):
        paths = _metadata_file_paths(metadata_path, base_path)
        self.paths = [p + ".cache.pt" for p in paths if os.path.exists(p + ".cache.pt")]
        assert len(self.paths) > 0, "No cached tensors found. Run --task cache_latents first."
        self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        data_id = index % len(self.paths)
        return torch.load(self.paths[data_id], map_location="cpu", weights_only=True)


# ============================================================================
# Module swap helpers
# ============================================================================

def upgrade_self_attention(wan_model: WanModel) -> WanModel:
    """Replace every block's SelfAttention with SelfAttentionHeadLora in-place."""
    # DiT의 모든 블록을 순회
    for blk in wan_model.blocks:
        old: _OrigSelfAttention = blk.self_attn # 기존 SelfAttention 가져오기
        new = SelfAttentionHeadLora(old.dim, old.num_heads)
        state = {k: v for k, v in old.state_dict().items() if not k.startswith("attn.")}
        new.load_state_dict(state, strict=True)
        new = new.to(device=next(old.parameters()).device,
                     dtype=next(old.parameters()).dtype)
        blk.self_attn = new
    return wan_model


def split_attention_heads(wan_model: WanModel, all_heads_type: list):
    """Call split_QKV_O on every block's SelfAttentionHeadLora."""
    assert len(all_heads_type) == len(wan_model.blocks)
    for i, blk in enumerate(wan_model.blocks):
        blk.self_attn.split_QKV_O(all_heads_type[i])


# ============================================================================
# Pipeline builder
# ============================================================================

def _build_pipe(model_paths: list, device="cpu") -> WanVideoPipeline:
    """Build WanVideoPipeline from local model file paths.

    Auto-detects the tokenizer directory by looking for google/umt5-xxl
    next to the first model file.
    """
    tokenizer_path = None
    for p in model_paths:
        candidate = os.path.join(os.path.dirname(p), "google", "umt5-xxl")
        if os.path.isdir(candidate):
            tokenizer_path = candidate
            break

    model_configs = [ModelConfig(path=p) for p in model_paths]

    if tokenizer_path is not None:
        tokenizer_config = ModelConfig(path=tokenizer_path)
    else:
        # Fall back to default download (requires internet on first run)
        tokenizer_config = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/*",
        )

    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )


# ============================================================================
# Encode helpers
# ============================================================================

def _encode_video(pipe, video: torch.Tensor) -> torch.Tensor:
    """Encode (C, T, H, W) video tensor to VAE latents."""
    video = video.to(dtype=pipe.torch_dtype, device=pipe.device)
    latents = pipe.vae.encode(
        [video], device=pipe.device, tiled=True,
        tile_size=(34, 34), tile_stride=(18, 16),
    )[0]
    return latents.to(dtype=pipe.torch_dtype, device=pipe.device)


def _encode_prompt(pipe, prompt: str) -> torch.Tensor:
    """Encode a text prompt to context embeddings."""
    return pipe.prompter.encode_prompt(prompt, positive=True, device=pipe.device)


# ============================================================================
# Head-type analysis pipeline (Step 1)
# ============================================================================
# Attention map을 수집해 각 head가 temporal(움직임)인지 spatial(외형)인지 분류.


def _collect_attention_maps(pipe, latents, prompt_emb, device):
    scheduler = pipe.scheduler
    noise = torch.randn_like(latents)
    timestep_id = torch.tensor([1])
    timestep = scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=device)
    noisy = scheduler.add_noise(latents, noise, timestep)

    attn_maps = []

    def _make_hook(store):
        def _hook(module, inp, outp):
            q_in, freqs = inp[0], inp[1]
            with torch.no_grad():
                if getattr(module, "q", None) is None:
                    return
                q_ = _rope_apply(module.norm_q(module.q(q_in)), freqs, module.num_heads)
                k_ = _rope_apply(module.norm_k(module.k(q_in)), freqs, module.num_heads)
                q_ = rearrange(q_, "b s (n d) -> b n s d", n=module.num_heads)
                k_ = rearrange(k_, "b s (n d) -> b n s d", n=module.num_heads)
                scale = math.sqrt(q_.shape[-1])
                attn_w = F.softmax(
                    torch.matmul(q_.float(), k_.float().transpose(-2, -1)) / scale, dim=-1
                )
                store.append(attn_w.cpu())
        return _hook

    hooks = []
    for blk in pipe.dit.blocks:
        hooks.append(blk.self_attn.register_forward_hook(_make_hook(attn_maps)))

    with torch.no_grad():
        pipe.dit(noisy, timestep=timestep, **prompt_emb)

    for h in hooks:
        h.remove()

    return attn_maps


def analyse_head_types(pipe, dataset, device, num_temporal_heads=6, n_samples=4):
    pipe.dit.eval()
    pipe.device = device

    accumulated_attn = None
    last_latents = None
    count = 0

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in loader:
        if count >= n_samples:
            break
        video  = batch["video"][0]
        prompt = batch["prompt"][0]

        context = _encode_prompt(pipe, prompt).to(device)
        latents = _encode_video(pipe, video).unsqueeze(0)
        last_latents = latents

        attn_maps = _collect_attention_maps(pipe, latents, {"context": context}, device)

        if accumulated_attn is None:
            accumulated_attn = [m.clone() for m in attn_maps]
        else:
            for i, m in enumerate(attn_maps):
                accumulated_attn[i] = accumulated_attn[i] + m
        count += 1

    S = accumulated_attn[0].shape[-1]
    T_lat = last_latents.shape[2]
    frame_size_lat = S // T_lat

    all_heads_type = []
    for layer_attn in accumulated_attn:
        layer_attn = layer_attn / count
        head_types = classify_heads(layer_attn, frame_size=frame_size_lat, num_temporal=num_temporal_heads)
        all_heads_type.append(head_types)

    return all_heads_type


# ============================================================================
# Cache latents (Step 2 for train)
# ============================================================================
# VAE + 텍스트 인코더 결과를 디스크에 미리 저장하는 모듈.

class LightningModelForCache(pl.LightningModule):
    """Pre-encode VAE + text embeddings to disk for faster training."""

    def __init__(self, model_paths, mode="motion"):
        super().__init__()
        self.pipe = _build_pipe(model_paths, device="cpu")
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.mode = mode

    def test_step(self, batch, batch_idx):
        self.pipe.device = self.device
        video  = batch["video"][0]   # (C, T, H, W)
        prompt = batch["prompt"][0]
        path   = batch.get("path", ["unknown"])[0]

        context = _encode_prompt(self.pipe, prompt)  # (1, L, D)

        if self.mode == "subject":
            # Subject: cache per-frame latents (single-frame training)
            frames_latents = []
            T = video.shape[1]
            for t in range(T):
                frame = video[:, t:t + 1, :, :]
                lat = _encode_video(self.pipe, frame)
                frames_latents.append(lat.unsqueeze(0))
            data = {"context": context, "frames_latents": frames_latents}

        else:
            # Motion: cache full-clip latent (Stage 1) AND per-frame latents (Stage 0)
            latents = _encode_video(self.pipe, video).unsqueeze(0)  # (1, C_lat, T_lat, ...)
            frames_latents = []
            T = video.shape[1]
            for t in range(T):
                frame = video[:, t:t + 1, :, :]
                lat = _encode_video(self.pipe, frame)
                frames_latents.append(lat.unsqueeze(0))
            data = {"latents": latents, "frames_latents": frames_latents, "context": context}

        torch.save(data, path + ".cache.pt")


def _fmt_sec(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def _enable_all_adapters(model):
    """Restore each LoRA module's active adapter(s) to whatever it actually has.

    After two inject_adapter_in_model calls (adapter_name="spatial" then
    "temporal"), PEFT sets every module's active_adapters to ["temporal"].
    Modules that only have a "spatial" adapter then silently skip LoRA during
    the forward pass, breaking the computation graph.  This helper resets each
    module to use its own adapter(s) regardless of how many it has.
    """
    for module in model.modules():
        if hasattr(module, "lora_A") and module.lora_A:
            module.set_adapter(list(module.lora_A.keys()))


# ============================================================================
# Lightning training module
# ============================================================================

class LightningModelForTrain(pl.LightningModule):
    """Head-specific LoRA training with pure MSE diffusion loss.

    subject mode : spatial  head LoRA only.
    motion  mode : spatial head LoRA (Stage 0, single frame)
                   → freeze → temporal head LoRA (Stage 1, full clip).
                   Stages alternate every Lightning epoch; pass max_epochs = 2 * N
                   to get N full (Stage-0 + Stage-1) cycles.
    """

    SPATIAL_TARGETS  = ["q_spatial",  "k_spatial",  "v_spatial",  "o_spatial"]
    TEMPORAL_TARGETS = ["q_temporal", "k_temporal", "v_temporal", "o_temporal"]

    @staticmethod
    def _print_registered_lora(model, label: str = ""):
        registered = sorted({
            name.rsplit(".", 2)[0]          # strip .lora_A / .lora_B
            for name, _ in model.named_modules()
            if hasattr(_, "lora_A")
        })
        suffix_counts: dict = {}
        for full in registered:
            suffix = full.rsplit(".", 1)[-1]
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        print(f"[INFO] {label} registered LoRA modules: {len(registered)} total")
        for suffix, cnt in sorted(suffix_counts.items()):
            print(f"       {suffix}: {cnt} layers")

    def __init__(
        self,
        model_paths: list,
        mode: str = "subject",
        head_types_path: str = None,
        all_heads_type: list = None,
        lora_rank: int = 4,
        lora_alpha: float = 4.0,
        lora_rank_spatial: int = None,
        lora_rank_temporal: int = None,
        lora_alpha_spatial: float = None,
        lora_alpha_temporal: float = None,
        learning_rate: float = 1e-5,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
        pretrained_lora_path: str = None,
        spatial_targets: list = None,
        temporal_targets: list = None,
        stage1_start_epoch: int = None,
    ):
        super().__init__()
        assert mode in ("subject", "motion"), f"mode must be 'subject' or 'motion', got {mode!r}"
        self.mode = mode
        self.stage1_start_epoch = stage1_start_epoch
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        # ── Load model ──────────────────────────────────────────────────────
        self.pipe = _build_pipe(model_paths, device="cpu")
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # ── Upgrade SelfAttention → SelfAttentionHeadLora ───────────────────
        upgrade_self_attention(self.pipe.dit)

        # ── Load / build head type list ─────────────────────────────────────
        if all_heads_type is not None:
            head_types = all_heads_type
        elif head_types_path is not None and os.path.exists(head_types_path):
            head_types = torch.load(head_types_path, map_location="cpu")
            print(f"[INFO] Loaded head types from {head_types_path}")
        else:
            num_layers = len(self.pipe.dit.blocks)
            num_heads  = self.pipe.dit.blocks[0].num_heads
            half = num_heads // 2
            head_types = [[0] * half + [1] * (num_heads - half)] * num_layers
            print("[WARN] No head_types found. Using deterministic 50/50 split.")

        # ── Split attention heads ────────────────────────────────────────────
        split_attention_heads(self.pipe.dit, head_types)

        # ── Freeze all parameters first ──────────────────────────────────────
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.dit.train()

        # ── Resolve LoRA target modules ──────────────────────────────────────
        spa_targets = spatial_targets  if spatial_targets  is not None else self.SPATIAL_TARGETS
        tem_targets = temporal_targets if temporal_targets is not None else self.TEMPORAL_TARGETS
        print(f"[INFO] spatial  LoRA targets : {spa_targets}")
        print(f"[INFO] temporal LoRA targets : {tem_targets}")

        # ── Inject LoRA ──────────────────────────────────────────────────────
        rank_spa  = lora_rank_spatial   if lora_rank_spatial   is not None else lora_rank
        rank_tem  = lora_rank_temporal  if lora_rank_temporal  is not None else lora_rank
        alpha_spa = lora_alpha_spatial  if lora_alpha_spatial  is not None else lora_alpha
        alpha_tem = lora_alpha_temporal if lora_alpha_temporal is not None else lora_alpha

        if mode == "subject":
            lora_config = LoraConfig(
                r=rank_spa, lora_alpha=alpha_spa,
                init_lora_weights=True,
                target_modules=spa_targets,
            )
            self.pipe.dit = inject_adapter_in_model(lora_config, self.pipe.dit)
            self._print_registered_lora(self.pipe.dit, "subject(spatial)")

        else:  # motion: named adapters so shared targets (e.g. ffn.0) get parallel LoRA
            if spa_targets:
                spa_config = LoraConfig(
                    r=rank_spa, lora_alpha=alpha_spa,
                    init_lora_weights=True,
                    target_modules=spa_targets,
                )
                self.pipe.dit = inject_adapter_in_model(spa_config, self.pipe.dit, adapter_name="spatial")
                self._print_registered_lora(self.pipe.dit, "motion(spatial)")
            tem_config = LoraConfig(
                r=rank_tem, lora_alpha=alpha_tem,
                init_lora_weights=True,
                target_modules=tem_targets,
            )
            self.pipe.dit = inject_adapter_in_model(tem_config, self.pipe.dit, adapter_name="temporal")
            self._print_registered_lora(self.pipe.dit, "motion(spatial+temporal)")
            # Modules in both targets (e.g. ffn.0, ffn.2) receive two adapters;
            # activate both so their outputs are summed in the forward pass.
            _enable_all_adapters(self.pipe.dit)
            print(f"[INFO] LoRA rank/alpha — spatial: {rank_spa}/{alpha_spa}, temporal: {rank_tem}/{alpha_tem}")

        # Upcast LoRA params to fp32
        for param in self.pipe.dit.parameters():
            if param.requires_grad:
                param.data = param.to(torch.float32)

        if pretrained_lora_path is not None:
            sd = load_state_dict(pretrained_lora_path)
            missing, unexpected = self.pipe.dit.load_state_dict(sd, strict=False)
            print(f"[INFO] Loaded pretrained LoRA: {len(sd) - len(missing)} params loaded, "
                  f"{len(unexpected)} unexpected.")

        # ── Cache param lists for fast stage switching (motion only) ─────────
        if mode == "motion":
            self._spatial_params  = [
                p for n, p in self.pipe.dit.named_parameters()
                if "lora" in n and "spatial" in n
            ]
            self._temporal_params = [
                p for n, p in self.pipe.dit.named_parameters()
                if "lora" in n and "temporal" in n
            ]
            # Start at stage 0: spatial trainable, temporal frozen
            self._set_stage(0)

        total_trainable = sum(p.numel() for p in self.pipe.dit.parameters() if p.requires_grad)
        print(f"[INFO] Mode={mode}  initial trainable params: {total_trainable:,}")

    # ── Stage switching (motion mode) ────────────────────────────────────────

    def _set_stage(self, stage: int):
        """stage 0 = spatial trainable; stage 1 = temporal trainable."""
        for p in self._spatial_params:
            p.requires_grad_(stage == 0)
        for p in self._temporal_params:
            p.requires_grad_(stage == 1)
        label = "Stage 0 (spatial LoRA)"  if stage == 0 else "Stage 1 (temporal LoRA)"
        print(f"[INFO] Motion training: {label}")

    def on_train_start(self):
        self._train_start_time = time.time()
        self._epoch_start_time = time.time()

    def on_train_epoch_start(self):
        self._epoch_start_time = time.time()

        cur  = self.trainer.current_epoch
        total = self.trainer.max_epochs
        if cur > 0 and hasattr(self, "_train_start_time"):
            elapsed   = time.time() - self._train_start_time
            avg_epoch = elapsed / cur
            remaining = avg_epoch * (total - cur)
            print(f"[TIME] Elapsed: {_fmt_sec(elapsed)}  "
                  f"ETA: {_fmt_sec(remaining)}  "
                  f"(avg {avg_epoch:.1f}s/epoch)")

        if self.mode == "motion":
            threshold = self.stage1_start_epoch if self.stage1_start_epoch is not None \
                        else self.trainer.max_epochs // 2
            stage = 0 if cur < threshold else 1
            self._set_stage(stage)

    # ── Forward (training step) ──────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        self.pipe.device = self.device

        context = batch["context"]
        if isinstance(context, (list, tuple)):
            context = context[0]
        # DataLoader collation adds an extra batch dim on top of the saved batch dim=1
        context = context.squeeze(0).to(device=self.device, dtype=self.pipe.torch_dtype)

        if self.mode == "subject":
            # Pick a random pre-encoded single-frame latent
            frames_latents = batch["frames_latents"]
            frame_idx = torch.randint(0, len(frames_latents), (1,)).item()
            latents = frames_latents[frame_idx].squeeze(0).to(self.device)

        else:  # motion
            threshold = self.stage1_start_epoch if self.stage1_start_epoch is not None \
                        else self.trainer.max_epochs // 2
            stage = 0 if self.trainer.current_epoch < threshold else 1
            if stage == 0:
                # Stage 0: single random frame for spatial head appearance learning
                frames_latents = batch["frames_latents"]
                frame_idx = torch.randint(0, len(frames_latents), (1,)).item()
                latents = frames_latents[frame_idx].squeeze(0).to(self.device)
            else:
                # Stage 1: full video clip for temporal head motion learning
                latents = batch["latents"].squeeze(0).to(self.device)

        # Add noise
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(
            dtype=self.pipe.torch_dtype, device=self.device,
        )
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # DiT forward
        noise_pred = self.pipe.dit(
            noisy_latents,
            timestep=timestep,
            context=context,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
        )

        # Pure MSE loss
        loss = F.mse_loss(noise_pred.float(), target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        stage_label = "" if self.mode == "subject" else f"_stage{stage if self.mode == 'motion' else 0}"
        self.log(f"train_loss{stage_label}", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Include all LoRA params (both spatial and temporal for motion mode)
        # Optimizer holds state for all; requires_grad controls which get updated.
        all_lora = [p for n, p in self.pipe.dit.named_parameters() if "lora" in n]
        return torch.optim.AdamW(all_lora, lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_names = {n for n, p in self.pipe.dit.named_parameters() if "lora" in n}
        checkpoint.update({
            n: p for n, p in self.pipe.dit.state_dict().items() if n in trainable_names
        })


# ============================================================================
# Argument parser
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Head-LoRA spatial/motion customization for Wan2.1")

    p.add_argument("--task", required=True,
                   choices=["head_analysis", "cache_latents", "train"])
    p.add_argument("--mode", default="subject", choices=["subject", "motion"])

    p.add_argument("--model_paths", nargs="+", required=True)
    p.add_argument("--dataset_base_path", required=True)
    p.add_argument("--dataset_metadata_path", required=True)
    p.add_argument("--head_types_path", default=None)
    p.add_argument("--output_path", default="./output")

    p.add_argument("--num_frames", type=int, default=17)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)

    p.add_argument("--num_temporal_heads", type=int, default=6)
    p.add_argument("--n_analysis_samples", type=int, default=4)

    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_rank_spatial",  type=int,   default=None,
                   help="motion mode: spatial LoRA rank. Defaults to --lora_rank.")
    p.add_argument("--lora_rank_temporal", type=int,   default=None,
                   help="motion mode: temporal LoRA rank. Defaults to --lora_rank.")
    p.add_argument("--lora_alpha", type=float, default=4.0)
    p.add_argument("--lora_alpha_spatial",  type=float, default=None,
                   help="motion mode: spatial LoRA alpha. Defaults to --lora_alpha.")
    p.add_argument("--lora_alpha_temporal", type=float, default=None,
                   help="motion mode: temporal LoRA alpha. Defaults to --lora_alpha.")
    p.add_argument("--stage1_start_epoch", type=int, default=None,
                   help="motion mode: Lightning epoch at which to switch from Stage 0 to Stage 1. "
                        "Defaults to max_epochs // 2.")
    p.add_argument("--pretrained_lora_path", default=None)
    p.add_argument("--spatial_targets", default=None,
                   help="Comma-separated spatial LoRA targets. "
                        "Default: q_spatial,k_spatial,v_spatial,o_spatial")
    p.add_argument("--temporal_targets", default=None,
                   help="Comma-separated temporal LoRA targets. "
                        "Default: q_temporal,k_temporal,v_temporal,o_temporal")

    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--max_epochs", type=int, default=1,
                   help="Number of full training epochs. "
                        "For motion mode each epoch internally runs Stage0 + Stage1, "
                        "so Lightning max_epochs will be 2x this value.")
    p.add_argument("--steps_per_epoch", type=int, default=500)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    p.add_argument("--use_gradient_checkpointing", action="store_true")
    p.add_argument("--use_gradient_checkpointing_offload", action="store_true")
    p.add_argument("--training_strategy", default="auto",
                   choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"])

    return p.parse_args()


# ============================================================================
# Task implementations
# ============================================================================

def task_head_analysis(args):
    pipe = _build_pipe(args.model_paths, device="cpu")
    pipe.scheduler.set_timesteps(1000, training=True)
    pipe.dit = pipe.dit.cuda()
    pipe.vae = pipe.vae.cuda()
    if hasattr(pipe, 'text_encoder'):
        pipe.text_encoder = pipe.text_encoder.cuda()
    if hasattr(pipe, 'prompter') and hasattr(pipe.prompter, 'text_encoder'):
        pipe.prompter.text_encoder = pipe.prompter.text_encoder.cuda()
    pipe.device = torch.device("cuda")
    pipe.dit.eval()

    dataset = VideoDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        mode="motion",
    )

    device = torch.device("cuda")
    all_heads_type = analyse_head_types(
        pipe=pipe, dataset=dataset, device=device,
        num_temporal_heads=args.num_temporal_heads,
        n_samples=args.n_analysis_samples,
    )

    os.makedirs(args.output_path, exist_ok=True)
    out_path = os.path.join(args.output_path, "head_types.pt")
    torch.save(all_heads_type, out_path)

    n_layers = len(all_heads_type)
    n_heads  = len(all_heads_type[0])
    n_temporal = sum(t == 0 for t in all_heads_type[0])
    print(f"\n[HEAD ANALYSIS] {n_layers} layers, {n_heads} heads/layer, "
          f"{n_temporal} temporal / {n_heads - n_temporal} spatial per layer")
    print(f"[HEAD ANALYSIS] Saved to {out_path}")
    for i, ht in enumerate(all_heads_type[:3]):
        print(f"  block[{i:2d}]: {ht}")


def task_cache_latents(args):
    raw_dataset = VideoDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        mode=args.mode,
    )

    class _DatasetWithPath(torch.utils.data.Dataset):
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            d = self.ds[i]
            row = self.ds.data[i]
            fname = row.get("file_name", row.get("video", row.get("image", "")))
            d["path"] = os.path.join(self.ds.base_path, fname)
            return d

    dataloader = torch.utils.data.DataLoader(
        _DatasetWithPath(raw_dataset), batch_size=1,
        num_workers=args.dataloader_num_workers, shuffle=False,
    )
    model = LightningModelForCache(model_paths=args.model_paths, mode=args.mode)
    trainer = pl.Trainer(accelerator="gpu", devices="auto",
                         default_root_dir=args.output_path)
    trainer.test(model, dataloader)
    print(f"[INFO] Latents cached. Now run --task train.")


def task_train(args):
    dataset = CachedLatentDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        steps_per_epoch=args.steps_per_epoch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    def _parse_targets(raw):
        if raw is None:
            return None
        return [t.strip() for t in raw.split(",") if t.strip()]

    model = LightningModelForTrain(
        model_paths=args.model_paths,
        mode=args.mode,
        head_types_path=args.head_types_path,
        lora_rank=args.lora_rank,
        lora_rank_spatial=args.lora_rank_spatial,
        lora_rank_temporal=args.lora_rank_temporal,
        lora_alpha=args.lora_alpha,
        lora_alpha_spatial=args.lora_alpha_spatial,
        lora_alpha_temporal=args.lora_alpha_temporal,
        stage1_start_epoch=args.stage1_start_epoch,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
        spatial_targets=_parse_targets(args.spatial_targets),
        temporal_targets=_parse_targets(args.temporal_targets),
    )

    out_dir = os.path.join(args.output_path, args.mode)
    os.makedirs(out_dir, exist_ok=True)

    lightning_epochs = args.max_epochs

    trainer = pl.Trainer(
        max_epochs=lightning_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16-mixed",
        strategy=args.training_strategy,
        default_root_dir=out_dir,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl_callbacks.ModelCheckpoint(save_top_k=-1)],
    )
    trainer.fit(model, dataloader)
    print(f"[INFO] Training complete. Checkpoints saved in {out_dir}")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    # head 분류
    if args.task == "head_analysis":
        task_head_analysis(args)

    # VAE 인코딩 캐시
    elif args.task == "cache_latents":
        task_cache_latents(args)
    
    # LoRA 훈련
    elif args.task == "train":
        task_train(args)
