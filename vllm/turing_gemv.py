# SPDX-License-Identifier: Apache-2.0
"""Opt-in Triton streaming GEMV (M=1) for fp16 dense linears on Turing (sm75).

At decode (single row) every fp16 dense linear is a pure bandwidth-bound GEMV.
cuBLAS picks a split-K GEMV whose reduction pass is pure overhead at M=1; this
kernel streams the weight once, with no split-K. Measured on RTX 2080 Ti vs
cuBLAS: ~1.06-1.24x on the large shapes and up to ~1.6x on the small,
latency-bound ones. fp16 in, fp32 accumulate, fp16 out -- no precision change.

Gated by ``VLLM_TURING_GEMV=1``; only CUDA compute capability 7.5, fp16 weight,
and a 2D input with a single row take this path. Everything else falls back to
the stock unquantized GEMM. The op is capture-safe (no host sync), so it can be
recorded into the CUDA graph.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_ENABLED: bool | None = None

# Fixed launch config (see the Turing GEMV probe): BLOCK_N=8 keeps enough
# programs for the medium shapes while BLOCK_K=512/num_warps=4 matches the
# best-measured streaming tile.
_BLOCK_N = 8
_BLOCK_K = 512
_NUM_WARPS = 4


@triton.jit
def _turing_gemv_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    K,
    N,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    ns = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = ns < N
    k_offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + k_offs
        kmask = ks < K
        xv = tl.load(x_ptr + ks, mask=kmask, other=0.0).to(tl.float32)
        wv = tl.load(
            w_ptr + ns[:, None] * K + ks[None, :],
            mask=n_mask[:, None] & kmask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(wv * xv[None, :], axis=1)
    tl.store(y_ptr + ns, acc.to(y_ptr.dtype.element_ty), mask=n_mask)


def _turing_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x: [1, K] fp16 contiguous; w: [N, K] fp16 contiguous -> y: [1, N]."""
    K = x.shape[1]
    N = w.shape[0]
    y = torch.empty((x.shape[0], N), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(N, _BLOCK_N),)
    _turing_gemv_kernel[grid](
        x,
        w,
        y,
        K,
        N,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=_NUM_WARPS,
    )
    return y


def _turing_gemv_fake(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        try:
            from vllm.platforms import current_platform

            _ENABLED = (
                os.environ.get("VLLM_TURING_GEMV", "0") == "1"
                and current_platform.is_cuda()
                and current_platform.get_device_capability() == (7, 5)
            )
        except Exception:
            _ENABLED = False
        if _ENABLED:
            logger.info_once(
                "VLLM_TURING_GEMV: Triton streaming GEMV active for fp16 "
                "dense linears (sm75)."
            )
    return _ENABLED


def _register_op() -> None:
    if hasattr(torch.ops.vllm, "turing_gemv"):
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="turing_gemv",
        op_func=_turing_gemv,
        mutates_args=[],
        fake_impl=_turing_gemv_fake,
    )


def apply_patch() -> None:
    """Route M=1 fp16 unquantized linears through the Triton GEMV."""
    if not enabled():
        return
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if getattr(UnquantizedLinearMethod, "_turing_gemv_patched", False):
        return
    _register_op()
    orig_apply = UnquantizedLinearMethod.apply

    def apply(self, layer, x, bias=None):
        if x.dim() == 2 and x.shape[0] == 1 and x.dtype == torch.float16:
            w = layer.weight
            if w.dtype == torch.float16 and w.dim() == 2:
                if not x.is_contiguous():
                    x = x.contiguous()
                if not w.is_contiguous():
                    w = w.contiguous()
                out = torch.ops.vllm.turing_gemv(x, w)
                if bias is not None:
                    out = out + bias
                return out
        return orig_apply(self, layer, x, bias)

    UnquantizedLinearMethod.apply = apply
    UnquantizedLinearMethod._turing_gemv_patched = True
    logger.info_once("VLLM_TURING_GEMV: patched UnquantizedLinearMethod.apply.")
