import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from functools import partial
from einops import rearrange
from model.mlp import DiffMLP
from model.rotation2xyz import Rotation2xyz, Rotation2xyz_x
from model.transformer_utils import Block, trunc_normal_, positional_encoding

class CMDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 num_frames=60, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', cm_mode='add', body_model='smpl', wo_pos_emb=False, emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()

        # model and data configuration
        self.legacy = legacy
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

        # network hyperparameters
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.clip_dim = clip_dim

        # ablation switch?
        self.ablation = ablation

        # conditional patten
        self.action_emb = kargs.get('action_emb', None)
        self.normalize_output = kargs.get('normalize_encoder_output', False)
        self.cond_mode = kargs.get('cond_mode', 'no_cond') # conditional mode: text, action, ...
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.) # classifier-free guidance
        self.arch = arch # online(decoder)/ offline(encoder)/ gru/ mlp
        self.cm_mode = cm_mode # the way to fuse x and cmotion: add/ concat

        # InputProcess
        self.input_feats = self.njoints * self.nfeats
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
        self.cmo_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        # PositionEmbedding and TimestepEmb
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec # for online setting
        self.wo_pos_emb = wo_pos_emb

        # concat + fuse
        if self.cm_mode == 'concat':
            self.fuse_process = nn.Linear(self.latent_dim*2, self.latent_dim)

        # offline
        if self.arch == 'trans_enc' or self.arch == 'offline':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,num_layers=self.num_layers)
        
        # online: emb as the memory
        elif self.arch == 'trans_dec' or self.arch == 'online':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,num_layers=self.num_layers)
        
        # gru
        elif self.arch == 'gru':
            print("GRU init")
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        
        # mlp
        elif self.arch == 'mlp':
            print("MLP init")
            self.mlp = DiffMLP(self.latent_dim, seq=num_frames, num_layers=self.num_layers)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        # TimestepEmbed
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Conditional patten
        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                # CLIP + Linear
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            if 'action' in self.cond_mode:
                # lookup table in action label
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

        # OutputProcess
        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        # Recovering 3D joints/vertices from rotational representation during training/evaluation.
        self.body_model = body_model
        if body_model == 'smpl':
            self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)
        elif body_model == 'smplx':
            self.rot2xyz = Rotation2xyz_x(device='cpu', dataset=self.dataset)

    # CLIP is not updated during training. The optimizer only takes non-CLIP parameters.
    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    # Using CLIP as a frozen text encoder
    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    # classifier-free guidance, input: [B, D]
    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            # 1-> use null_cond, 0-> use real cond
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  
            return cond * (1. - mask)
        else:
            return cond

    # classifier-free guidance, input: [B, N, D], not use
    def mask_cond_sparse(self, cond, force_mask=True):
        bs, n, c = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(
                bs, 1, 1
            )  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond

    # 
    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    # for online
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [1, B, D]

        # CFG
        force_mask = y.get('uncond', False)

        # emb = emb_t + emb_text + emb_action
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action']) # [B, D]
            emb += self.mask_cond(action_emb, force_mask=force_mask) # [1, 64, 512] + [64, 512]
        
        # actor motion
        cmx = y['cmotion']

        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints*nfeats, 1, nframes)
            cmx_reshaped = cmx.reshape(bs, njoints*nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)     #[#frames, bs, d]
            emb_gru = emb_gru.permute(1, 2, 0)      #[bs, d, #frames]
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)  #[bs, d, 1, #frames]

            x = torch.cat((x_reshaped, emb_gru), axis=1)  #[bs, d+joints*feat, 1, #frames]
            cmx = torch.cat((cmx_reshaped, emb_gru), axis=1)

        # InputProcess
        x = self.input_process(x)
        cmx = self.cmo_process(cmx)

        # online
        if self.arch == 'online':
            # concat the condition motion feature
            if self.cm_mode == 'add':
                xseq = x + cmx
            elif self.cm_mode == 'concat':
                xseq = torch.cat((x, cmx), axis=-1) # [T, B, 2D]
                xseq = self.fuse_process(xseq) # [T, B, D]
            # emb in first token and memory
            if self.emb_trans_dec:
                xseq = torch.cat((emb, xseq), axis=0) # [T+1, B, D]
            else:
                xseq = xseq # [T, B, D]

            if not self.wo_pos_emb:
                xseq = self.sequence_pos_encoder(xseq)  # [T+1, B, D]

            # causal mask
            mask = self.generate_square_subsequent_mask(xseq.shape[0])
            mask = mask.to(xseq.device)

            # causal self-attention
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb, tgt_mask=mask)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb, tgt_mask=mask)
        
        # offline
        elif self.arch == 'offline':
            if self.cm_mode == 'add':
                xseq = x + cmx
            elif self.cm_mode == 'concat':
                xseq = torch.cat((x, cmx), axis=-1)
                xseq = self.fuse_process(xseq)
            else:
                raise NotImplementedError
            xseq = torch.cat((emb, xseq), axis=0)  # [T+1, B, D]
            xseq = self.sequence_pos_encoder(xseq)  # [T+1, B, D]
            output = self.seqTransEncoder(xseq)[1:]
        
        # mlp
        elif self.arch == 'mlp':
            xseq = torch.cat((cmx, x), axis=-1).permute(1, 0, 2)
            emb = emb.permute(1, 0, 2)
            output = self.mlp(xseq, emb).permute(1, 0, 2)
        
        # gru
        elif self.arch == 'gru':
            if self.cm_mode == 'add':
                xseq = cmx + x
            else:
                raise NotImplementedError
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        # OutputProcess
        output = self.output_process(output)  # [bs, njoints, nfeats, nframes] denoted x_0^ in the paper
        return output

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class PositionalEncoding(nn.Module):
    """
    Adds position to the motion token of [T,B,D].
    The pre-stored pe is also used by the TimestepEmbedder 
    as the initial embedding for the diffusion timestep.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # [5000, 1, 512]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    """
    timesteps       [B]
    lookup pe[t]    [B, 1, D]
    MLP             [B, 1, D]
    permute         [1, B, D]
    """
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder # PositionalEncoding

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        """
        timesteps:  [B]
        return:     [1,B,D]
        """
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    """
    Input:  [B, J, F, T]
    Output: [T, B, D]
    """
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, np, d]
            return x
        elif self.data_rep == 'rot_vel':
            #TODO not implemented
            # first_pose is pose in first frame, the following frames is velocity token, need two embed
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    """
    Input:  [T, B, D]
    Output: [B, J, F, T]
    """
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, input_feats]
        elif self.data_rep == 'rot_vel':
            # not implement
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0) # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    """
    Embedding of action category labels.
    num_actions: The number of action categories
    latent_dim: The dimension of the embedding vector for each action category
    """
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        # Learnable action category embedding table
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim)) # [num_actions, latent_dim]

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx] # [B, latent_dim] or [B, D]
        return output