import torch
import torch.nn as nn

from model.cnet.cnet_v5 import (
    CNetV5,
    InputProcess,
    Mlp,
    OutputProcess,
    PIACrossAttention,
    _build_actor_causal_mask,
    _build_causal_mask,
    _build_cross_causal_mask,
)


class CNetV5SingleStreamTransformerLayer(nn.Module):
    """Single-stream conditional transformer block for the reactor ablation."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.pia = PIACrossAttention(dim, num_heads, dropout=dropout)
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_pia = nn.LayerNorm(dim)
        self.norm_actor_pia = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout, activation)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        reactor,
        actor_memory,
        attn_mask=None,
        cross_attn_mask=None,
        return_attn=False,
    ):
        reactor_norm = self.norm_attn(reactor)
        reactor = reactor + self.drop(
            self.self_attn(
                reactor_norm, reactor_norm, reactor_norm, attn_mask=attn_mask
            )[0]
        )

        reactor_cond = reactor[:, :1, :]
        reactor_motion = reactor[:, 1:, :]
        actor_norm = self.norm_actor_pia(actor_memory)
        reactor_motion_norm = self.norm_pia(reactor_motion)

        if return_attn:
            pia_out, pia_weights = self.pia(
                reactor_motion_norm,
                actor_norm,
                attn_mask=cross_attn_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            pia_out = self.pia(
                reactor_motion_norm,
                actor_norm,
                attn_mask=cross_attn_mask,
                need_weights=False,
            )

        reactor_motion = reactor_motion + self.drop(pia_out)
        reactor = torch.cat([reactor_cond, reactor_motion], dim=1)
        reactor = reactor + self.drop(self.mlp(self.norm_ffn(reactor)))

        if return_attn:
            return reactor, pia_weights
        return reactor


class CNetV5ReactorSingleStream(CNetV5):
    """Ablation of CNetV5 with a holistic single-stream reactor generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        del self.body_input_process
        del self.hand_input_process
        del self.transformer_layers
        del self.body_output_process
        del self.hand_output_process

        self.reactor_input_process = InputProcess(
            self.data_rep,
            self.input_feats,
            self.latent_dim,
        )
        self.transformer_layers = nn.ModuleList(
            [
                CNetV5SingleStreamTransformerLayer(
                    dim=self.latent_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.ff_size / self.latent_dim,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.reactor_output_process = OutputProcess(
            self.data_rep,
            self.input_feats,
            self.latent_dim,
            self.njoints,
            self.nfeats,
        )

    def forward(self, x, timesteps, y=None, return_attn=False):
        y = y or {}
        actor = y.get("actor", y.get("cmotion", None))
        if actor is None:
            raise ValueError("Missing actor motion in y['actor'] or y['cmotion'].")

        actor_tokens, token_type_ids = self._build_actor_tokens(actor)
        nframes = x.shape[-1]

        online = self.arch == "online"
        attn_mask = None
        actor_attn_mask = None
        cross_attn_mask = None
        if online:
            num_actor_tokens_per_frame = 1 + len(self.part_names)
            attn_mask = _build_causal_mask(nframes + 1, x.device)
            actor_attn_mask = _build_actor_causal_mask(
                nframes, num_actor_tokens_per_frame, actor_tokens.device
            )
            cross_attn_mask = _build_cross_causal_mask(
                nframes, num_actor_tokens_per_frame, actor_tokens.device
            )

        actor_memory = self.actor_encoder(
            actor_tokens, token_type_ids, attn_mask=actor_attn_mask
        )
        actor_memory = self.actor_to_pia(actor_memory)

        reactor_embed = self.reactor_input_process(x)

        cond = self.embed_timestep(timesteps)
        force_mask = y.get("uncond", False)
        if "text" in self.cond_mode:
            enc_text = self.encode_text(y["text"])
            cond = cond + self.embed_text(
                self.mask_cond(enc_text, force_mask=force_mask)
            )

        cond = cond.unsqueeze(1)
        reactor_seq = torch.cat([cond, reactor_embed], dim=1)
        if not self.wo_pos_emb:
            reactor_seq = self.sequence_pos_encoder(reactor_seq)

        pia_weights = []
        for layer in self.transformer_layers:
            if return_attn:
                reactor_seq, weights = layer(
                    reactor_seq,
                    actor_memory,
                    attn_mask=attn_mask,
                    cross_attn_mask=cross_attn_mask,
                    return_attn=True,
                )
                pia_weights.append(weights)
            else:
                reactor_seq = layer(
                    reactor_seq,
                    actor_memory,
                    attn_mask=attn_mask,
                    cross_attn_mask=cross_attn_mask,
                    return_attn=False,
                )

        output = self.reactor_output_process(reactor_seq[:, 1:, :])
        if return_attn:
            return output, {"reactor_pia_weights": pia_weights}
        return output
