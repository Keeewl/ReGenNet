# cnet_v2.py
# V2.1: Joint-token ST blocks + (time_pos + joint_id) embeddings + CFG-style actor dropout + no-clone delta updates
# - x:      [B, J, F, T]
# - cmotion:[B, J, F, T]  (actor condition)
# - tokens: [B, T, J, D]
# - supports:
#     arch="offline": default (non-causal temporal attn)
#     arch="online": causal temporal attn inside STSelfAttnBlock (reactor past-only in temporal attn)
#   Note: ACT cross-attn is not time-masked here (keeps logic simple and same as offline).

import math
import torch
import torch.nn as nn

from model.rotation2xyz import Rotation2xyz, Rotation2xyz_x


def _resolve_joint_splits(
    njoints,
    body_joint_ids=None,
    hand_joint_ids=None,
    body_joints=None,
    hand_joints=None,
):
    """
    Fixed SMPL-X split used in your v1:
      body: 0..24 plus transl(55)
      hand: 25..54
    """
    body = torch.tensor(list(range(0, 25)) + [55], dtype=torch.long)
    hand = torch.tensor(list(range(25, 55)), dtype=torch.long)
    if njoints is not None:
        body = body[body < njoints]
        hand = hand[hand < njoints]
    return body, hand


def _causal_mask(T: int, device):
    # nn.MultiheadAttention supports bool attn_mask where True means "blocked"
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


class JointTokenInput(nn.Module):
    def __init__(self, nfeats, latent_dim):
        super().__init__()
        self.proj = nn.Linear(nfeats, latent_dim)

    def forward(self, x):
        # x: [B, J, F, T] -> [B, T, J, D]
        x = x.permute(0, 3, 1, 2)  # [B,T,J,F]
        return self.proj(x)        # [B,T,J,D]


class JointTokenOutput(nn.Module):
    def __init__(self, nfeats, latent_dim):
        super().__init__()
        self.proj = nn.Linear(latent_dim, nfeats)

    def forward(self, x):
        # x: [B, T, J, D] -> [B, J, F, T]
        x = self.proj(x)                 # [B,T,J,F]
        return x.permute(0, 2, 3, 1)      # [B,J,F,T]


class TimestepEmbedder(nn.Module):
    """
    Diffusion timestep embedding (not motion frame position embedding).
    """
    def __init__(self, latent_dim, max_len=2000):
        super().__init__()
        self.latent_dim = latent_dim

        pe = torch.zeros(max_len, latent_dim, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, latent_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / latent_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, timesteps):
        # timesteps: [B] long/int
        t_vec = self.pe[timesteps]           # [B,D]
        return self.time_embed(t_vec)        # [B,D]


class AdaLN(nn.Module):
    """
    AdaLN(t): LayerNorm(x) then apply per-channel (1+gamma(t))*x + beta(t)
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim * 2),
        )

    def forward(self, x, t_emb):
        x_norm = self.norm(x)
        gamma, beta = self.mlp(t_emb).chunk(2, dim=-1)  # [B,D], [B,D]

        # broadcast to x shape
        while gamma.dim() < x_norm.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return x_norm * (1.0 + gamma) + beta


def _build_ffn(latent_dim, dropout, mult=4):
    hidden = latent_dim * mult
    return nn.Sequential(
        nn.Linear(latent_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, latent_dim),
    )


class STSelfAttnBlock(nn.Module):
    """
    ST self-attention over [B,T,J,D]:
      - temporal attn per joint: reshape to [B*J, T, D]
      - spatial  attn per time : reshape to [B*T, J, D]
    Both are conditioned by AdaLN(t_emb).
    """
    def __init__(self, latent_dim, num_heads=4, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.ada_ln_attn = AdaLN(latent_dim)
        self.temporal_attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.spatial_attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)

        self.ada_ln_ffn = AdaLN(latent_dim)
        self.ffn = _build_ffn(latent_dim, dropout, mult=ffn_mult)

    def forward(self, x, t_emb, causal=False):
        # x: [B,T,J,D]
        bsz, T, J, D = x.shape
        x_norm = self.ada_ln_attn(x, t_emb)

        attn_mask = _causal_mask(T, x.device) if causal else None

        # temporal per joint: [B*J, T, D]
        x_t = x_norm.permute(0, 2, 1, 3).contiguous().view(bsz * J, T, D)
        y_t = self.temporal_attn(x_t, x_t, x_t, attn_mask=attn_mask)[0]
        y_t = y_t.view(bsz, J, T, D).permute(0, 2, 1, 3).contiguous()  # [B,T,J,D]

        # spatial per time: [B*T, J, D]
        x_s = x_norm.contiguous().view(bsz * T, J, D)
        y_s = self.spatial_attn(x_s, x_s, x_s)[0]
        y_s = y_s.view(bsz, T, J, D)

        x = x + self.drop(y_t + y_s)
        x = x + self.drop(self.ffn(self.ada_ln_ffn(x, t_emb)))
        return x


class PET(nn.Module):
    """
    Part-wise (body/hand) STSelfAttn updates.
    Returns delta (updated - x) without cloning x (performance-friendly).
    """
    def __init__(self, latent_dim, num_heads, dropout, body_ids, hand_ids, ffn_mult=4):
        super().__init__()
        self.register_buffer("body_ids", torch.as_tensor(body_ids, dtype=torch.long))
        self.register_buffer("hand_ids", torch.as_tensor(hand_ids, dtype=torch.long))
        self.body_block = STSelfAttnBlock(latent_dim, num_heads, dropout, ffn_mult=ffn_mult)
        self.hand_block = STSelfAttnBlock(latent_dim, num_heads, dropout, ffn_mult=ffn_mult)

    def forward(self, x, t_emb, causal=False):
        # x: [B,T,J,D]
        delta = torch.zeros_like(x)

        if self.body_ids.numel() > 0:
            xb = x.index_select(2, self.body_ids)              # [B,T,Jb,D]
            ub = self.body_block(xb, t_emb, causal=causal)     # updated
            delta.index_copy_(2, self.body_ids, ub - xb)

        if self.hand_ids.numel() > 0:
            xh = x.index_select(2, self.hand_ids)
            uh = self.hand_block(xh, t_emb, causal=causal)
            delta.index_copy_(2, self.hand_ids, uh - xh)

        return delta


class CrossAttnBlock(nn.Module):
    """
    Cross-attn from q (subset joints) to kv (all joints), within the current window.
    We flatten [T,J] into tokens for simplicity: [B, T*J, D].
    """
    def __init__(self, latent_dim, num_heads=4, dropout=0.1, ffn_mult=4):
        super().__init__()
        self.q_norm = AdaLN(latent_dim)
        self.kv_norm = AdaLN(latent_dim)
        self.attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)

        self.ffn_norm = AdaLN(latent_dim)
        self.ffn = _build_ffn(latent_dim, dropout, mult=ffn_mult)

    def forward(self, q, kv, t_emb):
        # q:  [B,T,Jp,D]
        # kv: [B,T,J ,D]
        bsz, T, Jp, D = q.shape

        q_flat = q.contiguous().view(bsz, T * Jp, D)                     # [B, T*Jp, D]
        kv_flat = kv.contiguous().view(bsz, kv.shape[1] * kv.shape[2], D)  # [B, T*J, D]

        qn = self.q_norm(q_flat, t_emb)
        kvn = self.kv_norm(kv_flat, t_emb)

        out = self.attn(qn, kvn, kvn)[0]
        q_flat = q_flat + self.drop(out)
        q_flat = q_flat + self.drop(self.ffn(self.ffn_norm(q_flat, t_emb)))

        return q_flat.view(bsz, T, Jp, D)


class ACT(nn.Module):
    """
    Actor-conditioned cross-attn updates (body/hand separately).
    Returns delta without cloning x.
    """
    def __init__(self, latent_dim, num_heads, dropout, body_ids, hand_ids, ffn_mult=4):
        super().__init__()
        self.register_buffer("body_ids", torch.as_tensor(body_ids, dtype=torch.long))
        self.register_buffer("hand_ids", torch.as_tensor(hand_ids, dtype=torch.long))
        self.body_block = CrossAttnBlock(latent_dim, num_heads, dropout, ffn_mult=ffn_mult)
        self.hand_block = CrossAttnBlock(latent_dim, num_heads, dropout, ffn_mult=ffn_mult)

    def forward(self, x, a_tok, t_emb):
        # x: [B,T,J,D], a_tok: [B,T,J,D]
        delta = torch.zeros_like(x)

        if self.body_ids.numel() > 0:
            xb = x.index_select(2, self.body_ids)
            ub = self.body_block(xb, a_tok, t_emb)
            delta.index_copy_(2, self.body_ids, ub - xb)

        if self.hand_ids.numel() > 0:
            xh = x.index_select(2, self.hand_ids)
            uh = self.hand_block(xh, a_tok, t_emb)
            delta.index_copy_(2, self.hand_ids, uh - xh)

        return delta


class WET(nn.Module):
    """
    Whole-body STSelfAttn update. Returns delta.
    """
    def __init__(self, latent_dim, num_heads, dropout, ffn_mult=4):
        super().__init__()
        self.block = STSelfAttnBlock(latent_dim, num_heads, dropout, ffn_mult=ffn_mult)

    def forward(self, x, t_emb, causal=False):
        updated = self.block(x, t_emb, causal=causal)
        return updated - x


class CNetV2Block(nn.Module):
    """
    x = x + PET(x)
    x = x + ACT(x, actor)
    x = x + WET(x)
    """
    def __init__(self, latent_dim, num_heads, dropout, body_ids, hand_ids, ffn_mult=4):
        super().__init__()
        self.pet = PET(latent_dim, num_heads, dropout, body_ids, hand_ids, ffn_mult=ffn_mult)
        self.act = ACT(latent_dim, num_heads, dropout, body_ids, hand_ids, ffn_mult=ffn_mult)
        self.wet = WET(latent_dim, num_heads, dropout, ffn_mult=ffn_mult)

    def forward(self, x, a_tok, t_emb, causal=False):
        x = x + self.pet(x, t_emb, causal=causal)
        x = x + self.act(x, a_tok, t_emb)
        x = x + self.wet(x, t_emb, causal=causal)
        return x


class CNetV2(nn.Module):
    """
    V2 joint-token diffusion denoiser (reactor conditioned on actor motion).
    """
    def __init__(
        self,
        modeltype,
        njoints,
        nfeats,
        num_actions,
        translation,
        pose_rep,
        glob,
        glob_rot,
        num_frames=60,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        ablation=None,
        activation="gelu",
        legacy=False,
        data_rep="rot6d",
        dataset="chi3d",
        clip_dim=512,
        arch="offline",
        cm_mode="concat",
        body_model="smplx",
        wo_pos_emb=False,
        emb_trans_dec=False,
        clip_version=None,
        **kargs
    ):
        super().__init__()

        # keep CMDM-like attrs for compatibility
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.arch = arch
        self.cm_mode = cm_mode
        self.cond_mode = kargs.get("cond_mode", "no_cond")

        # CFG-style condition dropout (applies to actor motion tokens)
        self.cond_mask_prob = float(kargs.get("cond_mask_prob", 0.0))

        # position embeddings for motion frames + joint identity
        self.wo_pos_emb = wo_pos_emb
        self.max_frames = int(kargs.get("max_frames", num_frames if num_frames is not None else 2000))
        self.time_pos = nn.Parameter(torch.zeros(self.max_frames, self.latent_dim))
        self.joint_emb = nn.Embedding(self.njoints, self.latent_dim)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.trunc_normal_(self.joint_emb.weight, std=0.02)

        # joint split
        body_joint_ids, hand_joint_ids = _resolve_joint_splits(
            self.njoints,
            kargs.get("body_joint_ids"),
            kargs.get("hand_joint_ids"),
            kargs.get("body_joints"),
            kargs.get("hand_joints"),
        )
        self.register_buffer("body_joint_ids", body_joint_ids)
        self.register_buffer("hand_joint_ids", hand_joint_ids)

        # token in/out
        self.joint_input = JointTokenInput(self.nfeats, self.latent_dim)
        self.joint_output = JointTokenOutput(self.nfeats, self.latent_dim)

        # diffusion timestep embedder (for AdaLN)
        # set max_len to be safe; you can pass kargs["diffusion_steps"] to adjust
        tmax = int(kargs.get("diffusion_steps", 2000))
        self.timestep_embedder = TimestepEmbedder(self.latent_dim, max_len=tmax)

        # ffn multiplier (optional)
        self.ffn_mult = int(kargs.get("ffn_mult", 4))

        # blocks
        self.blocks = nn.ModuleList(
            [
                CNetV2Block(
                    self.latent_dim,
                    self.num_heads,
                    self.dropout,
                    self.body_joint_ids,
                    self.hand_joint_ids,
                    ffn_mult=self.ffn_mult,
                )
                for _ in range(self.num_layers)
            ]
        )

        # rot2xyz (kept for compatibility with training/eval pipeline)
        self.body_model = body_model
        if body_model == "smpl":
            self.rot2xyz = Rotation2xyz(device="cpu", dataset=self.dataset)
        elif body_model == "smplx":
            self.rot2xyz = Rotation2xyz_x(device="cpu", dataset=self.dataset)
        else:
            raise ValueError("CNetV2 only supports body_model='smpl' or 'smplx'.")

    def parameters_wo_clip(self):
        # no clip in v2; kept for compatibility
        return [p for name, p in self.named_parameters() if not name.startswith("clip_model.")]

    def forward(self, x, timesteps, y=None):
        """
        x: [B, J, F, T] reactor noise sequence
        timesteps: [B]
        y: dict with actor motion in y['cmotion'] (or y['actor'])
        """
        y = y or {}
        actor = y.get("cmotion", y.get("actor", None))
        if actor is None:
            raise ValueError("Missing actor motion in y['cmotion'] (or y['actor']).")

        # [B,T,J,D]
        x_tok = self.joint_input(x)
        a_tok = self.joint_input(actor)

        B, T, J, D = x_tok.shape

        # CFG-style training-time drop of actor condition
        if self.training and self.cond_mask_prob > 0.0:
            drop = torch.bernoulli(
                torch.full((B, 1, 1, 1), self.cond_mask_prob, device=x_tok.device)
            )
            a_tok = a_tok * (1.0 - drop)

        # explicit uncond
        if y.get("uncond", False):
            a_tok = torch.zeros_like(a_tok)

        # motion position embeddings (time) + joint identity embeddings (joint)
        if not self.wo_pos_emb:
            if T > self.max_frames:
                raise ValueError(
                    f"T={T} exceeds max_frames={self.max_frames}. "
                    f"Set kargs['max_frames'] larger or increase num_frames."
                )
            tpos = self.time_pos[:T].view(1, T, 1, D)            # [1,T,1,D]
            jpos = self.joint_emb.weight.view(1, 1, J, D)        # [1,1,J,D]
            x_tok = x_tok + tpos + jpos
            a_tok = a_tok + tpos + jpos

        # diffusion timestep embedding for AdaLN
        t_emb = self.timestep_embedder(timesteps)  # [B,D]

        # online: causal temporal attention inside ST blocks
        causal = (self.arch == "online")

        for blk in self.blocks:
            x_tok = blk(x_tok, a_tok, t_emb, causal=causal)

        return self.joint_output(x_tok)

    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, "rot2xyz"):
            self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if hasattr(self, "rot2xyz"):
            self.rot2xyz.smpl_model.train(*args, **kwargs)