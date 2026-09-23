"""Validate the SDPA attention compat against upstream's structure and against torch MHA.

Upstream's `MultiheadFlashAttention` cannot run here (flash_attn is CUDA-only and its
inner forward asserts `q.is_cuda`), so it cannot be diffed numerically on this machine.
Two checks stand in:

  1. **Structural** — our state_dict keys and shapes must match upstream's exactly, so
     the official checkpoint loads 0 missing / 0 unexpected. Upstream's module can be
     CONSTRUCTED on Mac (only its forward touches CUDA), so this is a real comparison.
  2. **Numerical** — against `torch.nn.MultiheadAttention` with the same packed weights,
     which computes the same softmax(QK^T/sqrt(d))V. Flash attention is a kernel, not a
     different algorithm, so agreeing with torch MHA is the meaningful check.

Run: PYTHONPATH=.:diffusiondrive_planner/upstream-nusc PYTORCH_ENABLE_MPS_FALLBACK=1 \
       conda run -n uniad2.0 python -m pytest diffusiondrive_planner/tests/test_attention_compat.py -q
"""

import pytest
import torch

from diffusiondrive_planner.attention_compat import (
    FlashMHACompat,
    MultiheadAttentionCompat,
    _in_projection_packed,
)

BS, SQ, SK, EMBED, HEADS = 2, 7, 5, 64, 8


def upstream_attention():
    return pytest.importorskip(
        "projects.mmdet3d_plugin.models.attention",
        reason="run with PYTHONPATH including diffusiondrive_planner/upstream-nusc",
    )


# ------------------------------------------------------------------ structural

def test_state_dict_keys_match_upstream():
    """The whole point: the official checkpoint must load into our module."""
    mod = upstream_attention()
    ref = mod.MultiheadFlashAttention(embed_dims=EMBED, num_heads=HEADS, batch_first=True)
    ours = MultiheadAttentionCompat(embed_dims=EMBED, num_heads=HEADS, batch_first=True)

    ref_sd, our_sd = ref.state_dict(), ours.state_dict()
    assert set(ref_sd) == set(our_sd), (
        f"key mismatch\n  only upstream: {set(ref_sd) - set(our_sd)}"
        f"\n  only ours:     {set(our_sd) - set(ref_sd)}"
    )
    for k in ref_sd:
        assert ref_sd[k].shape == our_sd[k].shape, f"shape mismatch at {k}"


def test_loads_upstream_state_dict_0_0():
    mod = upstream_attention()
    ref = mod.MultiheadFlashAttention(embed_dims=EMBED, num_heads=HEADS, batch_first=True)
    ours = MultiheadAttentionCompat(embed_dims=EMBED, num_heads=HEADS, batch_first=True)
    missing, unexpected = ours.load_state_dict(ref.state_dict(), strict=False)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"


def test_expected_key_names():
    ours = MultiheadAttentionCompat(embed_dims=EMBED, num_heads=HEADS)
    keys = set(ours.state_dict())
    assert keys == {
        "attn.in_proj_weight", "attn.in_proj_bias",
        "attn.out_proj.weight", "attn.out_proj.bias",
    }


def test_in_projection_packed_matches_upstream():
    mod = upstream_attention()
    torch.manual_seed(0)
    q, k, v = (torch.randn(BS, SQ, EMBED) for _ in range(3))
    w = torch.randn(3 * EMBED, EMBED)
    b = torch.randn(3 * EMBED)
    for got, want in zip(
        _in_projection_packed(q, k, v, w, b), mod._in_projection_packed(q, k, v, w, b)
    ):
        torch.testing.assert_close(got, want, atol=0, rtol=0)


# ------------------------------------------------------------------- numerical

def torch_mha_reference(mha: FlashMHACompat):
    """An nn.MultiheadAttention carrying the same packed weights."""
    ref = torch.nn.MultiheadAttention(
        EMBED, HEADS, dropout=0.0, bias=True, batch_first=True
    )
    with torch.no_grad():
        ref.in_proj_weight.copy_(mha.in_proj_weight)
        ref.in_proj_bias.copy_(mha.in_proj_bias)
        ref.out_proj.weight.copy_(mha.out_proj.weight)
        ref.out_proj.bias.copy_(mha.out_proj.bias)
    return ref.eval()


def test_self_attention_matches_torch_mha():
    torch.manual_seed(0)
    ours = FlashMHACompat(EMBED, HEADS).eval()
    ref = torch_mha_reference(ours)
    x = torch.randn(BS, SQ, EMBED)

    with torch.no_grad():
        got, _ = ours(x, x, x)
        want, _ = ref(x, x, x, need_weights=False)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_cross_attention_matches_torch_mha():
    """Different key/value length — the cross-attention case the loop actually uses."""
    torch.manual_seed(1)
    ours = FlashMHACompat(EMBED, HEADS).eval()
    ref = torch_mha_reference(ours)
    q = torch.randn(BS, SQ, EMBED)
    kv = torch.randn(BS, SK, EMBED)

    with torch.no_grad():
        got, _ = ours(q, kv, kv)
        want, _ = ref(q, kv, kv, need_weights=False)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_uses_default_softmax_scale():
    """Scale must be 1/sqrt(head_dim), not 1/sqrt(embed_dim).

    Using the wrong one keeps every shape valid and quietly changes the temperature.
    Verified implicitly by matching torch MHA, and explicitly here on a rank-1 case.
    """
    torch.manual_seed(2)
    ours = FlashMHACompat(EMBED, HEADS).eval()
    with torch.no_grad():
        ours.in_proj_weight.copy_(torch.eye(EMBED).repeat(3, 1))
        ours.in_proj_bias.zero_()
        ours.out_proj.weight.copy_(torch.eye(EMBED))
        ours.out_proj.bias.zero_()
        x = torch.randn(1, 3, EMBED)
        got, _ = ours(x, x, x)

        q = x.view(1, 3, HEADS, EMBED // HEADS).transpose(1, 2)
        scale = (EMBED // HEADS) ** -0.5
        attn = torch.softmax(q @ q.transpose(-2, -1) * scale, dim=-1)
        want = (attn @ q).transpose(1, 2).reshape(1, 3, EMBED)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


# -------------------------------------------------------------------- wrapper

def test_residual_uses_pre_pos_query():
    """identity defaults to the ORIGINAL query, before query_pos is added.

    Adding pos first and then using that as the residual is the natural mistake and
    shifts every output by query_pos.
    """
    torch.manual_seed(3)
    m = MultiheadAttentionCompat(EMBED, HEADS).eval()
    q = torch.randn(BS, SQ, EMBED)
    pos = torch.randn(BS, SQ, EMBED)

    with torch.no_grad():
        out = m(q, query_pos=pos)
        attn_only, _ = m.attn(q + pos, q + pos, q)  # value must NOT get pos
    torch.testing.assert_close(out, q + attn_only, atol=1e-5, rtol=1e-5)


def test_pos_is_not_added_to_value():
    """mmcv convention: pos goes to q and k, never v."""
    torch.manual_seed(4)
    m = MultiheadAttentionCompat(EMBED, HEADS).eval()
    q = torch.randn(BS, SQ, EMBED)
    kv = torch.randn(BS, SK, EMBED)
    kpos = torch.randn(BS, SK, EMBED)

    with torch.no_grad():
        out = m(q, key=kv, value=kv, key_pos=kpos)
        expected, _ = m.attn(q, kv + kpos, kv)
    torch.testing.assert_close(out, q + expected, atol=1e-5, rtol=1e-5)


def test_query_pos_reused_as_key_pos_only_when_shapes_match():
    torch.manual_seed(5)
    m = MultiheadAttentionCompat(EMBED, HEADS).eval()
    q = torch.randn(BS, SQ, EMBED)
    pos = torch.randn(BS, SQ, EMBED)

    with torch.no_grad():
        # Self-attention: shapes match, so query_pos is reused for the key.
        same = m(q, key=q, value=q, query_pos=pos)
        want_same, _ = m.attn(q + pos, q + pos, q)
        torch.testing.assert_close(same, q + want_same, atol=1e-5, rtol=1e-5)

        # Cross-attention with a different key length: pos must NOT reach the key.
        kv = torch.randn(BS, SK, EMBED)
        cross = m(q, key=kv, value=kv, query_pos=pos)
        want_cross, _ = m.attn(q + pos, kv, kv)
        torch.testing.assert_close(cross, q + want_cross, atol=1e-5, rtol=1e-5)


def test_explicit_identity_overrides_default():
    torch.manual_seed(6)
    m = MultiheadAttentionCompat(EMBED, HEADS).eval()
    q = torch.randn(BS, SQ, EMBED)
    ident = torch.randn(BS, SQ, EMBED)
    with torch.no_grad():
        out = m(q, identity=ident)
        attn_only, _ = m.attn(q, q, q)
    torch.testing.assert_close(out, ident + attn_only, atol=1e-5, rtol=1e-5)


def test_runs_on_mps():
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS")
    m = MultiheadAttentionCompat(EMBED, HEADS).eval().to("mps")
    x = torch.randn(BS, SQ, EMBED, device="mps")
    with torch.no_grad():
        out = m(x)
    assert out.shape == (BS, SQ, EMBED) and torch.isfinite(out).all()


def test_gradients_flow():
    m = MultiheadAttentionCompat(EMBED, HEADS).train()
    x = torch.randn(BS, SQ, EMBED, requires_grad=True)
    m(x).pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert m.attn.in_proj_weight.grad is not None
    assert m.attn.out_proj.weight.grad is not None
