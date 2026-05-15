"""
wan_video_dit_head_lora.py

Wan2.1 DiT with head-specific split for subject/motion customization.

Extends FollowYourMotion's split_QKV approach to also split the output
projection `o`, enabling head-specific LoRA on q, k, v, AND o.

Head classification (from FYM / Collect_attn_map.py):
    0 = temporal head  (high w_t / w_s ratio -> motion)
    1 = subject head   (low  w_t / w_s ratio -> appearance)

After split_QKV_O():
    Spatial  heads live in: self_attn.q_spatial,  k_spatial,  v_spatial,  o_spatial
    Temporal heads live in: self_attn.q_temporal, k_temporal, v_temporal, o_temporal

LoRA target_modules for spatial  training: ["q_spatial",  "k_spatial",  "v_spatial",  "o_spatial"]
LoRA target_modules for temporal training: ["q_temporal", "k_temporal", "v_temporal", "o_temporal"]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List
from einops import rearrange
from .utils import hash_state_dict_keys

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


def flash_attention(q, k, v, num_heads, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64,
                                       device=position.device).div(dim // 2))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim, end=1024, theta=10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim, end=1024, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


def _head_flat_indices(heads_pos: List[int], head_type: int, head_dim: int) -> List[int]:
    """Return flat indices in the concatenated (B, S, num_heads*head_dim) tensor
    corresponding to heads of the given type.
    head_type=0 -> temporal, head_type=1 -> spatial."""
    indices = []
    for i, t in enumerate(heads_pos):
        if t == head_type:
            indices.extend(range(i * head_dim, (i + 1) * head_dim))
    return indices


class SelfAttentionHeadLora(nn.Module):
    """SelfAttention with support for splitting Q, K, V, O by head type.

    Before split_QKV_O() is called, the module behaves identically to the
    original Wan2.1 SelfAttention.

    After split_QKV_O(heads_pos) is called:
      - q, k, v, o are deleted and replaced by typed sub-layers:
          q_spatial,  k_spatial,  v_spatial,  o_spatial   (type=1)
          q_temporal, k_temporal, v_temporal, o_temporal   (type=0)
      - PEFT LoRA can then be injected on exactly the desired sub-layers.
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self._split_done = False

    # ------------------------------------------------------------------
    # Split helpers
    # ------------------------------------------------------------------

    def split_QKV_O(self, heads_pos: List[int]):
        """Split q, k, v, o by head type and register typed sub-layers.

        heads_pos: list of ints, length = num_heads
                   0 = temporal head, 1 = spatial head
        """
        head_dim = self.head_dim
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Gather row slices of weight (and bias) for each type
        def _split_rows(W, b=None):
            # W: (out_features, in_features), b: (out_features,) or None
            rows_spa, rows_tem = [], []
            bias_spa, bias_tem = [], []
            for i, t in enumerate(heads_pos):
                chunk_W = W[i * head_dim:(i + 1) * head_dim]
                if t == 1:
                    rows_spa.append(chunk_W)
                    if b is not None:
                        bias_spa.append(b[i * head_dim:(i + 1) * head_dim])
                else:
                    rows_tem.append(chunk_W)
                    if b is not None:
                        bias_tem.append(b[i * head_dim:(i + 1) * head_dim])
            w_spa = torch.cat(rows_spa)
            w_tem = torch.cat(rows_tem)
            b_spa = torch.cat(bias_spa) if bias_spa else None
            b_tem = torch.cat(bias_tem) if bias_tem else None
            return w_spa, w_tem, b_spa, b_tem

        # Gather column slices of o weight (o.weight: (dim, dim))
        def _split_o_cols(W):  # W shape: (out_features, in_features=dim)
            cols_spa, cols_tem = [], []
            for i, t in enumerate(heads_pos):
                start, end = i * head_dim, (i + 1) * head_dim
                if t == 1:
                    cols_spa.append(W[:, start:end])
                else:
                    cols_tem.append(W[:, start:end])
            return torch.cat(cols_spa, dim=1), torch.cat(cols_tem, dim=1)

        q_w_spa, q_w_tem, q_b_spa, q_b_tem = _split_rows(self.q.weight.data, self.q.bias.data if self.q.bias is not None else None)
        k_w_spa, k_w_tem, k_b_spa, k_b_tem = _split_rows(self.k.weight.data, self.k.bias.data if self.k.bias is not None else None)
        v_w_spa, v_w_tem, v_b_spa, v_b_tem = _split_rows(self.v.weight.data, self.v.bias.data if self.v.bias is not None else None)
        o_w_spa, o_w_tem = _split_o_cols(self.o.weight.data)

        in_dim = q_w_spa.shape[1]
        dim_spa = q_w_spa.shape[0]
        dim_tem = q_w_tem.shape[0]
        out_dim = o_w_spa.shape[0]

        has_qkv_bias = self.q.bias is not None

        def _make_linear(out_f, in_f, bias=False):
            lin = nn.Linear(in_f, out_f, bias=bias)
            return lin.to(device=device, dtype=dtype)

        self.q_spatial  = _make_linear(dim_spa, in_dim, bias=has_qkv_bias)
        self.q_temporal = _make_linear(dim_tem, in_dim, bias=has_qkv_bias)
        self.k_spatial  = _make_linear(dim_spa, in_dim, bias=has_qkv_bias)
        self.k_temporal = _make_linear(dim_tem, in_dim, bias=has_qkv_bias)
        self.v_spatial  = _make_linear(dim_spa, in_dim, bias=has_qkv_bias)
        self.v_temporal = _make_linear(dim_tem, in_dim, bias=has_qkv_bias)
        # o_spatial carries the o bias; o_temporal has no bias
        self.o_spatial  = _make_linear(out_dim, dim_spa, bias=True)
        self.o_temporal = _make_linear(out_dim, dim_tem, bias=False)

        self.q_spatial.weight.data  = q_w_spa
        self.q_temporal.weight.data = q_w_tem
        self.k_spatial.weight.data  = k_w_spa
        self.k_temporal.weight.data = k_w_tem
        self.v_spatial.weight.data  = v_w_spa
        self.v_temporal.weight.data = v_w_tem
        self.o_spatial.weight.data  = o_w_spa
        self.o_temporal.weight.data = o_w_tem

        if has_qkv_bias:
            self.q_spatial.bias.data  = q_b_spa
            self.q_temporal.bias.data = q_b_tem
            self.k_spatial.bias.data  = k_b_spa
            self.k_temporal.bias.data = k_b_tem
            self.v_spatial.bias.data  = v_b_spa
            self.v_temporal.bias.data = v_b_tem

        if self.o.bias is not None:
            self.o_spatial.bias.data = self.o.bias.data.clone()

        # Store head ordering info for mix_heads and output splitting
        self.heads_pos = heads_pos
        # Flat indices in the full (B, S, dim) attention output tensor
        self._spatial_out_idx  = _head_flat_indices(heads_pos, 1, head_dim)
        self._temporal_out_idx = _head_flat_indices(heads_pos, 0, head_dim)
        # Number of spatial / temporal heads
        self.n_spatial_heads  = sum(1 for t in heads_pos if t == 1)
        self.n_temporal_heads = sum(1 for t in heads_pos if t == 0)

        # Remove original projections
        del self.q, self.k, self.v, self.o
        self.requires_grad_(False)
        self._split_done = True

    def _mix_heads(self, temporal, spatial):
        """Interleave temporal and spatial head tensors to restore original head ordering."""
        head_dim = self.head_dim
        out = []
        it, is_ = 0, 0
        for t in self.heads_pos:
            if t == 0:
                out.append(temporal[..., it * head_dim:(it + 1) * head_dim])
                it += 1
            else:
                out.append(spatial[..., is_ * head_dim:(is_ + 1) * head_dim])
                is_ += 1
        return torch.cat(out, dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, freqs):
        if not self._split_done:
            # Original (pre-split) path
            q = self.norm_q(self.q(x))
            k = self.norm_k(self.k(x))
            v = self.v(x)
            attn_out = flash_attention(
                rope_apply(q, freqs, self.num_heads),
                rope_apply(k, freqs, self.num_heads),
                v, self.num_heads,
            )
            return self.o(attn_out)

        # Post-split path: spatial and temporal sub-layers
        q_spa = self.q_spatial(x)   # (B, S, dim_spa)
        q_tem = self.q_temporal(x)  # (B, S, dim_tem)
        k_spa = self.k_spatial(x)
        k_tem = self.k_temporal(x)
        v_spa = self.v_spatial(x)
        v_tem = self.v_temporal(x)

        # Merge back to original head order for RMSNorm + RoPE
        q_full = self.norm_q(self._mix_heads(q_tem, q_spa))
        k_full = self.norm_k(self._mix_heads(k_tem, k_spa))
        v_full = self._mix_heads(v_tem, v_spa)

        attn_out = flash_attention(
            rope_apply(q_full, freqs, self.num_heads),
            rope_apply(k_full, freqs, self.num_heads),
            v_full, self.num_heads,
        )  # (B, S, num_heads * head_dim) in original head order

        # Split attention output by head type and apply typed o projections
        device = attn_out.device
        spa_idx = torch.tensor(self._spatial_out_idx,  device=device)
        tem_idx = torch.tensor(self._temporal_out_idx, device=device)

        x_spa = attn_out[..., spa_idx]   # (B, S, dim_spa)
        x_tem = attn_out[..., tem_idx]   # (B, S, dim_tem)

        return self.o_spatial(x_spa) + self.o_temporal(x_tem)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, eps=1e-6, has_image_input=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

    def forward(self, x, y):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = flash_attention(q, k, v, num_heads=self.num_heads)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn  = SelfAttentionHeadLora(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, x, context, t_mod, freqs):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.self_attn(input_x, freqs)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps):
        super().__init__()
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, t_mod):
        shift, scale = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)


class WanModelHeadLora(nn.Module):
    """Wan2.1 DiT with head-specific LoRA support.

    Call split_attention(all_heads_type) once (with pre-computed or
    analysed head type lists) before injecting LoRA adapters.
    """

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim)

    def split_attention(self, all_heads_type: List[List[int]]):
        """Split all DiT blocks' self-attention by head type.

        all_heads_type: list of length num_layers, each element is a list of
                        length num_heads with values 0 (temporal) or 1 (spatial).
        """
        self.all_heads_type = all_heads_type
        for i, blk in enumerate(self.blocks):
            blk.self_attn.split_QKV_O(all_heads_type[i])

    def patchify(self, x):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size

    def unpatchify(self, x, grid_size):
        return rearrange(
            x, 'b (f h w) (xp yp zp c) -> b c (f xp) (h yp) (w zp)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            xp=self.patch_size[0], yp=self.patch_size[1], zp=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        x, (f, h, w) = self.patchify(x)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        def _checkpoint_forward(module):
            def _fn(*inputs):
                return module(*inputs)
            return _fn

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            _checkpoint_forward(block), x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        _checkpoint_forward(block), x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        from .wan_video_dit import WanModelStateDictConverter
        return WanModelStateDictConverter()


# ---------------------------------------------------------------------------
# Head analysis
# ---------------------------------------------------------------------------

import math as _math


def get_attention_mask(mask_name: str, num_frame: int, frame_size: int) -> torch.Tensor:
    """Build spatial or temporal block-diagonal attention mask.

    Returns a binary mask of shape (num_frame * frame_size, num_frame * frame_size).
    spatial  mask: 1 where query and key are in nearby frames (same or adjacent)
    temporal mask: 1 where query and key share the same spatial position across frames
    """
    N = num_frame * frame_size
    block_size = 128
    num_block = _math.ceil(N / block_size)

    pixel_mask = torch.zeros(N, N, dtype=torch.float)
    for i in range(num_block):
        for j in range(num_block):
            if abs(i - j) < frame_size // block_size:
                r0, r1 = i * block_size, min((i + 1) * block_size, N)
                c0, c1 = j * block_size, min((j + 1) * block_size, N)
                pixel_mask[r0:r1, c0:c1] = 1.0

    if mask_name == "temporal":
        pixel_mask = (
            pixel_mask
            .reshape(frame_size, num_frame, frame_size, num_frame)
            .permute(1, 0, 3, 2)
            .reshape(N, N)
        )
    return pixel_mask


def classify_heads(
    attn: torch.Tensor,
    frame_size: int,
    num_temporal: Optional[int] = None,
    temporal_ratio_threshold: float = 1.3,
) -> List[int]:
    """Classify attention heads as subject (1) or temporal (0).

    attn: (B, num_heads, seq_len, seq_len) attention weight tensor
    frame_size: number of tokens per frame (H_lat * W_lat / patch²)
    num_temporal: if given, fix this many heads as temporal (top-k by score);
                  otherwise use temporal_ratio_threshold.

    Returns list of ints, length = num_heads.
        0 = temporal head
        1 = subject head
    """
    B, n, s, _ = attn.shape
    num_frame = s // frame_size
    spatial_mask  = get_attention_mask("spatial",  num_frame, frame_size).to(attn.device)
    temporal_mask = get_attention_mask("temporal", num_frame, frame_size).to(attn.device)

    scores = []
    for i in range(n):
        attn_h = attn[:, i]                         # (B, s, s)
        w_s = (attn_h * spatial_mask).sum()
        w_t = (attn_h * temporal_mask).sum()
        scores.append((w_t / (w_s + 1e-8)).item())

    if num_temporal is None:
        return [1 if scores[i] < temporal_ratio_threshold else 0 for i in range(n)]
    else:
        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
        types = [1] * n
        for idx in ranked[:num_temporal]:
            types[idx] = 0
        return types
