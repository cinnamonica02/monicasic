"""Compare tiny_moe/ref/qwen3_ref.py against transformers' Qwen3DecoderLayer (random small config, CPU).

  python -m tiny_moe.ref.check_qwen3_layer      (from the repo root)
Needs torch + transformers (CPU is enough at this size).
"""
import numpy as np
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

from .qwen3_ref import Cfg, LAYER_KEYS, decoder_layer, random_weights


def main(T=12, seed=0):
    c = Cfg(hidden=64, n_heads=4, n_kv_heads=2, head_dim=32, inter=128, rope_theta=1e6)
    hf_cfg = Qwen3Config(
        hidden_size=c.hidden, num_attention_heads=c.n_heads, num_key_value_heads=c.n_kv_heads,
        head_dim=c.head_dim, intermediate_size=c.inter, num_hidden_layers=1,
        rope_theta=c.rope_theta, rms_norm_eps=c.eps, attention_bias=False,
        max_position_embeddings=128, vocab_size=100, attn_implementation="eager",
    )
    layer = Qwen3DecoderLayer(hf_cfg, layer_idx=0).float().eval()
    w = random_weights(c, seed)
    sd = layer.state_dict()
    assert set(LAYER_KEYS) == set(sd), (set(LAYER_KEYS) ^ set(sd))
    layer.load_state_dict({k: torch.from_numpy(v) for k, v in w.items()})

    x = np.random.default_rng(seed + 1).standard_normal((T, c.hidden)).astype(np.float32)
    xt = torch.from_numpy(x)[None]
    pos = torch.arange(T)[None]
    pe = Qwen3RotaryEmbedding(hf_cfg)(xt, pos)
    mask = torch.full((T, T), float("-inf")).triu(1)[None, None]
    with torch.no_grad():
        out = layer(xt, attention_mask=mask, position_ids=pos, position_embeddings=pe)
    out = (out[0] if isinstance(out, tuple) else out)[0].numpy()

    mine = decoder_layer(x, w, c)
    err = np.abs(out - mine).max()
    print(f"max abs diff vs transformers: {err:.3e}  (output scale {np.abs(out).max():.2f})")
    assert err < 1e-4, "reference does not match transformers"
    print("OK")


if __name__ == "__main__":
    main()
