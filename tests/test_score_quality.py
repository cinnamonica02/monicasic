"""head_stats (streamed LM head) against a brute-force computation on the tiny fake checkpoint."""
import numpy as np
from test_qwen3_full import NL, VOCAB, _make

from tiny_moe.ref.check_qwen3_full import Checkpoint, cfg_from_json, final_hidden, forward
from tiny_moe.ref.score_quality import head_stats


def test_head_stats_matches_bruteforce(tmp_path):
    _make(tmp_path)
    ck = Checkpoint(tmp_path)
    cfg, n = cfg_from_json(tmp_path / "config.json")
    rng = np.random.default_rng(3)
    seqs = [rng.integers(0, VOCAB, 12), rng.integers(0, VOCAB, 12)]
    take = [np.arange(11)] * 2
    targets = np.concatenate([s[1:] for s in seqs])
    quiet = lambda *_: None
    h = final_hidden(ck, cfg, n, seqs, take, log=quiet)
    s = head_stats(ck, h, targets, head_chunk=7)                       # 7 does not divide 50: tests the ragged last chunk
    lg = np.concatenate(forward(ck, cfg, n, seqs, take, log=quiet)).astype(np.float64)
    np.testing.assert_array_equal(s["argmax"], lg.argmax(1))
    srt = np.sort(lg, 1)
    np.testing.assert_allclose(s["top1"], srt[:, -1], rtol=1e-6)
    np.testing.assert_allclose(s["top2"], srt[:, -2], rtol=1e-6)
    np.testing.assert_allclose(s["target_logit"], lg[np.arange(len(targets)), targets], rtol=1e-6)
    want_lse = np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1)) + lg.max(1)
    np.testing.assert_allclose(s["lse"], want_lse, rtol=1e-9)
