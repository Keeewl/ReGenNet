import numpy as np
import torch
import torch.nn as nn
import clip

from model.rotation2xyz import Rotation2xyz, Rotation2xyz_x


class CNetV3(nn.Module):
    """
    two-stream conditional denoiser
    = body stream + hand stream + actor-conditioned fusion + ParCo coordination
    """
    def __init__(
        self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
        num_frames=60, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
        ablation=None, activation="gelu", legacy=False, data_rep="rot6d", dataset="chi3d",
        clip_dim=512, arch="offline", cm_mode="concat", body_model="smplx", wo_pos_emb=False,
        clip_version=None, **kargs
    ):
        super().__init__()

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
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.clip_dim = clip_dim
        self.arch = arch
        self.cm_mode = cm_mode
        self.wo_pos_emb = wo_pos_emb

        self.cond_mode = kargs.get("cond_mode", "no_cond")
        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.0)

        self.input_feats = self.njoints * self.nfeats

        body_joint_ids, hand_joint_ids = _resolve_joint_splits(
            self.njoints,
            kargs.get("body_joint_ids"),
            kargs.get("hand_joint_ids"),
            kargs.get("body_joints"),
            kargs.get("hand_joints"),
        )
        self.register_buffer("body_joint_ids", body_joint_ids)
        self.register_buffer("hand_joint_ids", hand_joint_ids)

        if self.arch not in ["offline", "online"]:
            raise ValueError("CNet v3 only supports arch='offline' or 'online'.")
        if self.cm_mode != "concat":
            raise ValueError("CNet v3 only supports cm_mode='concat'.")
        if self.dataset != "chi3d":
            raise ValueError("CNet v3 only supports dataset='chi3d'.")

        self.actor_input_process = InputProcess(
            self.data_rep, self.input_feats, self.latent_dim
        )
        self.body_input_process = InputProcess(
            self.data_rep, int(self.body_joint_ids.numel()) * self.nfeats, self.latent_dim
        )
        self.hand_input_process = InputProcess(
            self.data_rep, int(self.hand_joint_ids.numel()) * self.nfeats, self.latent_dim
        )

        self.body_fuse_process = nn.Linear(self.latent_dim * 2, self.latent_dim)
        self.hand_fuse_process = nn.Linear(self.latent_dim * 2, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(
            self.latent_dim, dropout=self.dropout
        )
        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
        )

        if self.cond_mode != "no_cond":
            if "text" in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)

        self.transformer_layers = nn.ModuleList(
            [
                ParCoTransformerLayer(
                    dim=self.latent_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.ff_size / self.latent_dim,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.body_output_process = OutputProcess(
            self.data_rep,
            int(self.body_joint_ids.numel()) * self.nfeats,
            self.latent_dim,
            int(self.body_joint_ids.numel()),
            self.nfeats,
        )
        self.hand_output_process = OutputProcess(
            self.data_rep,
            int(self.hand_joint_ids.numel()) * self.nfeats,
            self.latent_dim,
            int(self.hand_joint_ids.numel()),
            self.nfeats,
        )

        self.body_model = body_model
        if body_model == "smpl":
            self.rot2xyz = Rotation2xyz(device="cpu", dataset=self.dataset)
        elif body_model == "smplx":
            self.rot2xyz = Rotation2xyz_x(device="cpu", dataset=self.dataset)
        else:
            raise ValueError("CNet v3 only supports body_model='smpl' or 'smplx'.")

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith("clip_model.")]

    def load_and_freeze_clip(self, clip_version):
        clip_model, _ = clip.load(clip_version, device="cpu", jit=False)
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        if self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(bs, 1)
            return cond * (1.0 - mask)
        return cond

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ["humanml", "kit"] else None
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2
            assert context_length < default_context_length
            texts = clip.tokenize(
                raw_text, context_length=context_length, truncate=True
            ).to(device)
            zero_pad = torch.zeros(
                [texts.shape[0], default_context_length - context_length],
                dtype=texts.dtype,
                device=texts.device,
            )
            texts = torch.cat([texts, zero_pad], dim=1)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device)
        return self.clip_model.encode_text(texts).float()

    def _split_reactor(self, x):
        body = x[:, self.body_joint_ids]
        hand = x[:, self.hand_joint_ids]
        return body, hand

    def _merge_outputs(self, body, hand):
        bs, _, _, nframes = body.shape
        out = torch.zeros(
            bs, self.njoints, self.nfeats, nframes, device=body.device, dtype=body.dtype
        )
        out[:, self.body_joint_ids] = body
        out[:, self.hand_joint_ids] = hand
        return out

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames] reactor noise sequence
        timesteps: [batch_size]
        y: dict with actor motion and optional text condition
        """
        bs, njoints, nfeats, nframes = x.shape
        y = y or {}
        actor = y.get("actor", y.get("cmotion", None))
        if actor is None:
            raise ValueError("Missing actor motion in y['actor'] or y['cmotion'].")

        actor_embed = self.actor_input_process(actor)

        body, hand = self._split_reactor(x)
        body_embed = self.body_input_process(body)
        hand_embed = self.hand_input_process(hand)

        body_embed = torch.cat([body_embed, actor_embed], dim=-1)
        body_embed = self.body_fuse_process(body_embed)
        hand_embed = torch.cat([hand_embed, actor_embed], dim=-1)
        hand_embed = self.hand_fuse_process(hand_embed)

        cond = self.embed_timestep(timesteps)
        force_mask = y.get("uncond", False)
        if "text" in self.cond_mode:
            enc_text = self.encode_text(y["text"])
            cond = cond + self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))

        cond = cond.unsqueeze(1)
        body_seq = torch.cat([cond, body_embed], dim=1)  # [B, 1+T, D]
        hand_seq = torch.cat([cond, hand_embed], dim=1)  # [B, 1+T, D]

        if not self.wo_pos_emb:
            body_seq = self.sequence_pos_encoder(body_seq)
            hand_seq = self.sequence_pos_encoder(hand_seq)

        online = self.arch == "online"
        attn_mask = None
        if online:
            attn_mask = _build_causal_mask(body_seq.shape[1], body_seq.device)

        for layer in self.transformer_layers:
            body_seq, hand_seq = layer(body_seq, hand_seq, attn_mask=attn_mask, online=online)

        body_seq = body_seq[:, 1:, :]
        hand_seq = hand_seq[:, 1:, :]

        body_out = self.body_output_process(body_seq)
        hand_out = self.hand_output_process(hand_seq)
        output = self._merge_outputs(body_out, hand_out)
        return output

    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, "rot2xyz"):
            self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if hasattr(self, "rot2xyz"):
            self.rot2xyz.smpl_model.train(*args, **kwargs)


class ParCoTransformerLayer(nn.Module):
    """
    body -> LN -> self-attn -> residual
    hand -> LN -> self-attn -> residual
    body, hand -> ParCoCoord
    body -> LN -> MLP -> residual
    hand -> LN -> MLP -> residual

    Block: 
    SelfAttn_body ​→ ParCoCoord(hand2body) → FFN_body
    SelfAttn_hand ​→ ParCoCoord(body2hand) → FFN_hand
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn_body = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_hand = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm_body_attn = nn.LayerNorm(dim)
        self.norm_hand_attn = nn.LayerNorm(dim)

        self.coord = ParCoCoord(dim, mlp_ratio, dropout, activation)

        self.norm_body_ffn = nn.LayerNorm(dim)
        self.norm_hand_ffn = nn.LayerNorm(dim)
        self.mlp_body = Mlp(dim, int(dim * mlp_ratio), dropout, activation)
        self.mlp_hand = Mlp(dim, int(dim * mlp_ratio), dropout, activation)

        self.drop = nn.Dropout(dropout)

    def forward(self, body, hand, attn_mask=None, online=False):
        body_norm = self.norm_body_attn(body)
        body = body + self.drop(
            self.self_attn_body(body_norm, body_norm, body_norm, attn_mask=attn_mask)[0]
        )
        hand_norm = self.norm_hand_attn(hand)
        hand = hand + self.drop(
            self.self_attn_hand(hand_norm, hand_norm, hand_norm, attn_mask=attn_mask)[0]
        )

        body, hand = self.coord(body, hand, online=online)

        body = body + self.drop(self.mlp_body(self.norm_body_ffn(body)))
        hand = hand + self.drop(self.mlp_hand(self.norm_hand_ffn(hand)))
        return body, hand


class ParCoCoord(nn.Module):
    """
    part-aware global context exchange
    """
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.1, activation="gelu"):
        super().__init__()
        self.mlp_body_from_hand = Mlp(dim, int(dim * mlp_ratio), dropout, activation)
        self.mlp_hand_from_body = Mlp(dim, int(dim * mlp_ratio), dropout, activation)
        self.norm_body = nn.LayerNorm(dim)
        self.norm_hand = nn.LayerNorm(dim)

    def forward(self, body, hand, online=False):
        # body/hand: [B, 1+T, D], exclude cond token for summaries
        body_tokens = body[:, 1:, :]
        hand_tokens = hand[:, 1:, :]

        if online:
            # Prefix mean enforces strict causality for ParCoCoord.
            body_cumsum = body_tokens.cumsum(dim=1)
            hand_cumsum = hand_tokens.cumsum(dim=1)
            denom = torch.arange(
                1, body_tokens.shape[1] + 1, device=body_tokens.device, dtype=body_tokens.dtype
            ).view(1, -1, 1)
            prefix_body = body_cumsum / denom
            prefix_hand = hand_cumsum / denom
            m_body = self.mlp_body_from_hand(prefix_hand)
            m_hand = self.mlp_hand_from_body(prefix_body)
            body_tokens = body_tokens + m_body
            hand_tokens = hand_tokens + m_hand
        else:
            s_body = body_tokens.mean(dim=1)
            s_hand = hand_tokens.mean(dim=1)
            m_body = self.mlp_body_from_hand(s_hand)
            m_hand = self.mlp_hand_from_body(s_body)
            body_tokens = body_tokens + m_body[:, None, :]
            hand_tokens = hand_tokens + m_hand[:, None, :]

        body = torch.cat([body[:, :1, :], body_tokens], dim=1)
        hand = torch.cat([hand[:, :1, :], hand_tokens], dim=1)
        body = self.norm_body(body)
        hand = self.norm_hand(hand)
        return body, hand


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[0, timesteps])


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == "rot_vel":
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((0, 3, 1, 2)).reshape(bs, nframes, njoints * nfeats)

        if self.data_rep in ["rot6d", "xyz", "hml_vec"]:
            x = self.poseEmbedding(x)
            return x
        if self.data_rep == "rot_vel":
            first_pose = x[:, [0]]
            first_pose = self.poseEmbedding(first_pose)
            vel = x[:, 1:]
            vel = self.velEmbedding(vel)
            return torch.cat((first_pose, vel), dim=1)
        raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == "rot_vel":
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        bs, nframes, d = output.shape
        if self.data_rep in ["rot6d", "xyz", "hml_vec"]:
            output = self.poseFinal(output)
        elif self.data_rep == "rot_vel":
            first_pose = output[:, [0]]
            first_pose = self.poseFinal(first_pose)
            vel = output[:, 1:]
            vel = self.velFinal(vel)
            output = torch.cat((first_pose, vel), dim=1)
        else:
            raise ValueError
        output = output.reshape(bs, nframes, self.njoints, self.nfeats)
        output = output.permute(0, 2, 3, 1)
        return output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, dropout, activation):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError("Unsupported activation for Mlp.")
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _build_causal_mask(seq_len, device):
    # True entries are masked (blocked). Includes cond token at position 0.
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


def _resolve_joint_splits(
    njoints, body_joint_ids, hand_joint_ids, body_joints, hand_joints
):
    # Fixed v1 split: transl stays in body.
    body = torch.tensor(list(range(0, 25)) + [55], dtype=torch.long)
    hand = torch.tensor(list(range(25, 55)), dtype=torch.long)
    return body, hand
