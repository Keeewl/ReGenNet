import math

import torch
import torch.nn as nn

from diffusion import gaussian_diffusion as gd
from diffusion.nn import timestep_embedding
from diffusion.respace import SpacedDiffusion, space_timesteps


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class ShapeEncoder(nn.Module):
    def __init__(self, beta_dim, hidden_dim, gender_num_embeddings=3, gender_embed_dim=16, dropout=0.1):
        super().__init__()
        self.gender_embed = nn.Embedding(gender_num_embeddings, gender_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(beta_dim + gender_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, betas, gender_id):
        in_dim = self.net[0].in_features - self.gender_embed.embedding_dim
        if betas.shape[-1] < in_dim:
            pad = betas.new_zeros(betas.shape[0], in_dim - betas.shape[-1])
            betas = torch.cat([betas, pad], dim=-1)
        elif betas.shape[-1] > in_dim:
            betas = betas[:, :in_dim]
        gender_feat = self.gender_embed(gender_id.long())
        feat = torch.cat([betas, gender_feat], dim=-1)
        return self.net(feat).unsqueeze(1)


class RelativeShapeEncoder(nn.Module):
    def __init__(self, beta_dim, hidden_dim, gender_num_embeddings=3, gender_embed_dim=16, dropout=0.1):
        super().__init__()
        self.gender_embed = nn.Embedding(gender_num_embeddings, gender_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(beta_dim + gender_embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, actor_betas, reactor_betas, actor_gender_id, reactor_gender_id):
        in_dim = self.net[0].in_features - self.gender_embed.embedding_dim * 2
        delta = actor_betas - reactor_betas
        if delta.shape[-1] < in_dim:
            pad = delta.new_zeros(delta.shape[0], in_dim - delta.shape[-1])
            delta = torch.cat([delta, pad], dim=-1)
        elif delta.shape[-1] > in_dim:
            delta = delta[:, :in_dim]
        actor_gender = self.gender_embed(actor_gender_id.long())
        reactor_gender = self.gender_embed(reactor_gender_id.long())
        feat = torch.cat([delta, actor_gender, reactor_gender], dim=-1)
        return self.net(feat).unsqueeze(1)


class TemporalSelfAttentionBlock(nn.Module):
    """
    Temporal self-attention per token.

    Inputs:
        x: [B, T, N, D]
        time_mask: [B, T] or None
    Outputs:
        out: [B, T, N, D]
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, time_mask=None):
        bsz, num_frames, num_tokens, hidden = x.shape
        y = self.ln1(x).permute(0, 2, 1, 3).reshape(bsz * num_tokens, num_frames, hidden)
        key_padding_mask = None
        if time_mask is not None:
            key_padding_mask = (~time_mask).repeat_interleave(num_tokens, dim=0)
        attn_out, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        attn_out = attn_out.reshape(bsz, num_tokens, num_frames, hidden).permute(0, 2, 1, 3)
        x = x + attn_out
        z = self.ln2(x).permute(0, 2, 1, 3).reshape(bsz * num_tokens, num_frames, hidden)
        mlp_out = self.mlp(z)
        mlp_out = mlp_out.reshape(bsz, num_tokens, num_frames, hidden).permute(0, 2, 1, 3)
        return x + mlp_out


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention from reactor tokens to context tokens.

    Inputs:
        reactor: [B, T, J, D]
        ctx: [B, T, P, D]
        ctx_mask: [B, T, P] or [B, P] or None
    Outputs:
        out: [B, T, J, D]
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, reactor, ctx, ctx_mask=None):
        bsz, num_frames, num_joints, hidden = reactor.shape
        q = reactor.reshape(bsz * num_frames, num_joints, hidden)
        kv = ctx.reshape(bsz * num_frames, ctx.shape[2], hidden)
        key_padding_mask = None
        if ctx_mask is not None:
            if ctx_mask.dim() == 2:
                ctx_mask = ctx_mask[:, None, :].expand(bsz, num_frames, -1)
            key_padding_mask = (~ctx_mask).reshape(bsz * num_frames, ctx_mask.shape[2])
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        attn_out = attn_out.reshape(bsz, num_frames, num_joints, hidden)
        out = reactor + attn_out
        out = out + self.mlp(self.ln(out))
        return out


class SpatialSelfAttentionBlock(nn.Module):
    """
    Spatial self-attention across joint tokens per frame.

    Inputs:
        x: [B, T, J, D]
    Outputs:
        out: [B, T, J, D]
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        bsz, num_frames, num_joints, hidden = x.shape
        q = x.reshape(bsz * num_frames, num_joints, hidden)
        attn_out, _ = self.attn(q, q, q, need_weights=False)
        attn_out = attn_out.reshape(bsz, num_frames, num_joints, hidden)
        out = x + attn_out
        out = out + self.mlp(self.ln(out))
        return out


class MeshConditionalDiffusionRefiner(nn.Module):
    """
    Mesh-aware conditional residual diffusion refiner.

    Inputs:
        x_t: [B, T, J, 6]
        timesteps: [B]
        coarse_local: [B, T, J, 6]
        actor_tokens: [B, T, P, Fa]
        mesh_tokens: [B, T, M, Fm]
        mesh_token_type: [B, M]
        cond_feat: [B, T, C]
        mesh_relation_feat: [B, T, Fr]
        time_mask: [B, T] or None
        actor_mask: [B, T, P] or [B, P] or None
        mesh_mask: [B, T, M] or [B, M] or None
    Outputs:
        pred_eps: [B, T, J, 6]
    """

    def __init__(
        self,
        joint_ids=None,
        hidden_dim=128,
        num_temporal_blocks=2,
        num_cross_blocks=2,
        num_spatial_blocks=1,
        dropout=0.1,
        cond_dim=18,
        actor_dim=6,
        mesh_dim=12,
        mesh_rel_dim=22,
        geometry_dim=13,
        target_summary_dim=10,
        mesh_type_vocab=16,
        time_embed_dim=128,
        use_spatial_attn=True,
        shape_dim=10,
        gender_num_embeddings=3,
        use_shape_condition=False,
    ):
        super().__init__()
        self.joint_ids = joint_ids
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.use_spatial_attn = bool(use_spatial_attn)
        self.use_shape_condition = bool(use_shape_condition)

        self.input_encoder = FeatureEncoder(6, hidden_dim, dropout=dropout)
        self.coarse_encoder = FeatureEncoder(6, hidden_dim, dropout=dropout)
        self.cond_encoder = FeatureEncoder(cond_dim, hidden_dim, dropout=dropout)
        self.rel_encoder = FeatureEncoder(mesh_rel_dim, hidden_dim, dropout=dropout)
        self.geometry_encoder = FeatureEncoder(geometry_dim, hidden_dim, dropout=dropout)
        self.target_summary_encoder = FeatureEncoder(target_summary_dim, hidden_dim, dropout=dropout)

        self.actor_encoder = FeatureEncoder(actor_dim, hidden_dim, dropout=dropout)
        self.mesh_encoder = FeatureEncoder(mesh_dim, hidden_dim, dropout=dropout)
        self.mesh_type_embed = nn.Embedding(mesh_type_vocab, hidden_dim)
        if self.use_shape_condition:
            self.actor_shape_encoder = ShapeEncoder(
                beta_dim=shape_dim,
                hidden_dim=hidden_dim,
                gender_num_embeddings=gender_num_embeddings,
                dropout=dropout,
            )
            self.reactor_shape_encoder = ShapeEncoder(
                beta_dim=shape_dim,
                hidden_dim=hidden_dim,
                gender_num_embeddings=gender_num_embeddings,
                dropout=dropout,
            )
            self.relative_shape_encoder = RelativeShapeEncoder(
                beta_dim=shape_dim,
                hidden_dim=hidden_dim,
                gender_num_embeddings=gender_num_embeddings,
                dropout=dropout,
            )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.temporal_blocks = nn.ModuleList(
            [TemporalSelfAttentionBlock(hidden_dim, num_heads=4, dropout=dropout) for _ in range(num_temporal_blocks)]
        )
        self.cross_blocks = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads=4, dropout=dropout) for _ in range(num_cross_blocks)]
        )
        self.spatial_blocks = nn.ModuleList(
            [SpatialSelfAttentionBlock(hidden_dim, num_heads=4, dropout=dropout) for _ in range(num_spatial_blocks)]
        )

        self.out_head = nn.Linear(hidden_dim, 6)
        self.contact_conf_head = nn.Linear(hidden_dim, 1)
        self.target_distance_head = nn.Linear(hidden_dim, 1)
        self.clearance_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x_t,
        timesteps,
        coarse_local=None,
        actor_tokens=None,
        actor_mask=None,
        mesh_tokens=None,
        mesh_token_type=None,
        mesh_mask=None,
        cond_feat=None,
        mesh_relation_feat=None,
        geometry_state_feat=None,
        target_geometry_summary=None,
        time_mask=None,
        actor_shape_tokens=None,
        reactor_shape_tokens=None,
        relative_shape_tokens=None,
        shape_mask=None,
        return_aux=False,
    ):
        if x_t.dim() != 4:
            raise ValueError("x_t must be [B,T,J,6]")

        h = self.input_encoder(x_t)
        if coarse_local is not None:
            h = h + self.coarse_encoder(coarse_local)
        if cond_feat is not None:
            h = h + self.cond_encoder(cond_feat).unsqueeze(2)
        if mesh_relation_feat is not None:
            h = h + self.rel_encoder(mesh_relation_feat).unsqueeze(2)
        if geometry_state_feat is not None:
            h = h + self.geometry_encoder(geometry_state_feat).unsqueeze(2)
        if target_geometry_summary is not None:
            h = h + self.target_summary_encoder(target_geometry_summary).unsqueeze(2)

        time_emb = timestep_embedding(timesteps, self.time_embed_dim)
        time_emb = self.time_mlp(time_emb).view(x_t.shape[0], 1, 1, self.hidden_dim)
        h = h + time_emb

        actor_ctx = None
        if actor_tokens is not None:
            actor_ctx = self.actor_encoder(actor_tokens)

        mesh_ctx = None
        if mesh_tokens is not None:
            mesh_ctx = self.mesh_encoder(mesh_tokens)
            if mesh_token_type is not None:
                if mesh_token_type.dim() == 2:
                    mesh_token_type = mesh_token_type[:, None, :].expand(mesh_ctx.shape[0], mesh_ctx.shape[1], -1)
                mesh_ctx = mesh_ctx + self.mesh_type_embed(mesh_token_type)

        shape_ctx = None
        if self.use_shape_condition:
            shape_tokens = [tok for tok in (actor_shape_tokens, reactor_shape_tokens, relative_shape_tokens) if tok is not None]
            if shape_tokens:
                base = []
                for tok in shape_tokens:
                    if tok.dim() == 3:
                        tok = tok[:, None, :, :].expand(x_t.shape[0], x_t.shape[1], -1, -1)
                    base.append(tok)
                shape_ctx = torch.cat(base, dim=2)
                if shape_mask is None:
                    shape_mask = torch.ones(shape_ctx.shape[0], shape_ctx.shape[2], device=x_t.device, dtype=torch.bool)

        for block in self.temporal_blocks:
            h = block(h, time_mask=time_mask)

        for block in self.cross_blocks:
            if actor_ctx is not None:
                h = block(h, actor_ctx, ctx_mask=actor_mask)
            if mesh_ctx is not None:
                h = block(h, mesh_ctx, ctx_mask=mesh_mask)
            if shape_ctx is not None:
                h = block(h, shape_ctx, ctx_mask=shape_mask)

        if self.use_spatial_attn:
            for block in self.spatial_blocks:
                h = block(h)

        pred_eps = self.out_head(h)
        if not return_aux:
            return pred_eps
        geom_token = h.mean(dim=2)
        return {
            "pred_eps": pred_eps,
            "contact_conf": self.contact_conf_head(geom_token),
            "target_distance": self.target_distance_head(geom_token),
            "clearance": self.clearance_head(geom_token),
        }

    def encode_shape_tokens(self, actor_betas, reactor_betas, actor_gender_id, reactor_gender_id):
        if not self.use_shape_condition:
            return {}
        return {
            "actor_shape_tokens": self.actor_shape_encoder(actor_betas, actor_gender_id),
            "reactor_shape_tokens": self.reactor_shape_encoder(reactor_betas, reactor_gender_id),
            "relative_shape_tokens": self.relative_shape_encoder(
                actor_betas, reactor_betas, actor_gender_id, reactor_gender_id
            ),
            "shape_mask": torch.ones(actor_betas.shape[0], 3, device=actor_betas.device, dtype=torch.bool),
        }


def create_spaced_diffusion(
    diffusion_steps=1000,
    noise_schedule="cosine",
    timestep_respacing=None,
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
    )


def predict_xstart_from_eps(diffusion, x_t, t, eps):
    scale = torch.from_numpy(diffusion.sqrt_alphas_cumprod).to(x_t.device)[t]
    scale = scale.view(-1, *([1] * (x_t.dim() - 1)))
    sigma = torch.from_numpy(diffusion.sqrt_one_minus_alphas_cumprod).to(x_t.device)[t]
    sigma = sigma.view(-1, *([1] * (x_t.dim() - 1)))
    return (x_t - sigma * eps) / scale.clamp(min=1e-6)
