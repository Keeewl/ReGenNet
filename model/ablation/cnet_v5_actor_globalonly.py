import torch

from model.cnet.cnet_v5 import (
    ActorEncoder,
    CNetV5,
    _build_actor_causal_mask,
    _build_causal_mask,
    _build_cross_causal_mask,
)


class CNetV5ActorGlobalOnly(CNetV5):
    """Ablation of CNetV5 with actor tokens encoded as global-only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        del self.actor_part_input_processes

        self.actor_token_names = ["global"]
        self.actor_num_tokens_per_frame = 1

        self.actor_encoder = ActorEncoder(
            dim=self.actor_dim,
            num_heads=self.num_heads,
            num_layers=self.actor_num_layers,
            num_token_types=self.actor_num_tokens_per_frame,
            mlp_ratio=self.ff_size / self.actor_dim,
            dropout=self.dropout,
            activation=self.activation,
        )

    def _build_actor_tokens(self, actor):
        actor_global = self.actor_global_input_process(actor)
        token_type_ids = actor_global.new_zeros(
            (actor_global.shape[1],), dtype=self.body_joint_ids.dtype
        )
        return actor_global, token_type_ids

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
            attn_mask = _build_causal_mask(nframes + 1, x.device)
            actor_attn_mask = _build_actor_causal_mask(
                nframes, self.actor_num_tokens_per_frame, actor_tokens.device
            )
            cross_attn_mask = _build_cross_causal_mask(
                nframes, self.actor_num_tokens_per_frame, actor_tokens.device
            )

        actor_memory = self.actor_encoder(
            actor_tokens, token_type_ids, attn_mask=actor_attn_mask
        )
        actor_memory = self.actor_to_pia(actor_memory)

        body, hand = self._split_reactor(x)
        body_embed = self.body_input_process(body)
        hand_embed = self.hand_input_process(hand)

        cond = self.embed_timestep(timesteps)
        force_mask = y.get("uncond", False)
        if "text" in self.cond_mode:
            enc_text = self.encode_text(y["text"])
            cond = cond + self.embed_text(
                self.mask_cond(enc_text, force_mask=force_mask)
            )

        cond = cond.unsqueeze(1)
        body_seq = torch.cat([cond, body_embed], dim=1)
        hand_seq = torch.cat([cond, hand_embed], dim=1)
        if not self.wo_pos_emb:
            body_seq = self.sequence_pos_encoder(body_seq)
            hand_seq = self.sequence_pos_encoder(hand_seq)

        body_pia_weights = []
        hand_pia_weights = []
        for layer in self.transformer_layers:
            if return_attn:
                body_seq, hand_seq, body_weights, hand_weights = layer(
                    body_seq,
                    hand_seq,
                    actor_memory,
                    attn_mask=attn_mask,
                    cross_attn_mask=cross_attn_mask,
                    online=online,
                    return_attn=True,
                )
                body_pia_weights.append(body_weights)
                hand_pia_weights.append(hand_weights)
            else:
                body_seq, hand_seq = layer(
                    body_seq,
                    hand_seq,
                    actor_memory,
                    attn_mask=attn_mask,
                    cross_attn_mask=cross_attn_mask,
                    online=online,
                    return_attn=False,
                )

        body_out = self.body_output_process(body_seq[:, 1:, :])
        hand_out = self.hand_output_process(hand_seq[:, 1:, :])
        output = self._merge_outputs(body_out, hand_out)

        if return_attn:
            return output, {
                "body_pia": body_pia_weights,
                "hand_pia": hand_pia_weights,
            }
        return output
