1) 深度拆解 cmdm.py：它在做什么、哪些必须复用、哪些要替换成 v1
1.1 数据流主干（offline 版本）

x: [B, J, F, T]（reactor, noised）
cmx: [B, J, F, T]（actor full, clean condition, y['cmotion']）
t: [B]

核心流程（offline）：

emb = embed_timestep(t) → [1, B, D]

x_embed = input_process(x) → [T, B, D]

cmx_embed = cmo_process(cmx) → [T, B, D]

fuse（add 或 concat+Linear）：xseq = fuse(x_embed, cmx_embed) → [T, B, D]

xseq = cat([emb, xseq], dim=0) → [T+1, B, D]

xseq = PE(xseq)

output = TransformerEncoder(xseq)[1:] → [T, B, D]

x0_hat = output_process(output) → [B, J, F, T]

✅ 这条链条里：PositionalEncoding / TimestepEmbedder / InputProcess / OutputProcess / offline 的拼 emb + PE + Encoder 都非常稳定、值得复用。