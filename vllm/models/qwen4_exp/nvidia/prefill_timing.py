# SPDX-License-Identifier: Apache-2.0
"""Opt-in CUDA-event timing for the Qwen4Exp prefill path (SM75 tuning).

Enable with VLLM_QWEN4_PREFILL_TIMING=1. Brackets submodules with CUDA events
and logs a breakdown for model steps whose GPU time exceeds _REPORT_MIN_MS
(prefill chunks; decode replays graphs). All activity is skipped during CUDA
graph capture (recording/synchronizing there invalidates the capture).
"""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED = os.environ.get("VLLM_QWEN4_PREFILL_TIMING", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

_TARGETS = {
    "Qwen4ExpQSAAttention": "qsa_total",
    "QSAIndexer": "qsa_indexer",
    "QwenGatedDeltaNetAttention": "gdn",
    "Qwen4ExpSparseMoeBlock": "moe",
    "GateLinear": "moe_gate",
    "SharedExperts": "moe_shared",
    "Qwen3NextMLP": "moe_shared_mlp",
    "Qwen4ExpPLELayer": "ple",
}
_ROOT_NAMES = ("Qwen4ExpModel",)
_REPORT_MIN_MS = 20.0

_events: dict[str, list] = {}
_stack: dict[str, list] = {}
_state: dict = {"num_tokens": 0, "root_start": None}
_dbg = {"root": 0, "target": 0}


def _capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _num_tokens() -> int:
    try:
        from vllm.forward_context import get_forward_context

        fc = get_forward_context()
        for attr in ("num_tokens", "num_actual_tokens"):
            nt = getattr(fc, attr, None)
            if nt is not None:
                return int(nt)
    except Exception:
        pass
    return 0


def _gpre(module, args):
    if not _ENABLED or _capturing():
        return
    name = type(module).__name__
    if name in _ROOT_NAMES:
        _events.clear()
        _stack.clear()
        _state["num_tokens"] = _num_tokens()
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        _state["root_start"] = e
        if _dbg["root"] == 0:
            _dbg["root"] = 1
            logger.info("[prefill-timing] root hook fired: %s num_tokens=%d",
                        name, _state["num_tokens"])
        return
    lbl = _TARGETS.get(name)
    if lbl is None:
        return
    if _dbg["target"] == 0:
        _dbg["target"] = 1
        logger.info("[prefill-timing] target hook fired: %s -> %s", name, lbl)
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    _stack.setdefault(lbl, []).append(e)


def _gpost(module, args, output):
    if not _ENABLED or _capturing():
        return
    name = type(module).__name__
    if name in _ROOT_NAMES:
        if _state["root_start"] is not None:
            _report()
        return
    lbl = _TARGETS.get(name)
    if lbl is None:
        return
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    s = _stack.get(lbl)
    if s:
        _events.setdefault(lbl, []).append((s.pop(), e))


def _report() -> None:
    torch.cuda.synchronize()
    tot = {lbl: sum(s.elapsed_time(e) for s, e in evs) for lbl, evs in _events.items()}
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    end.synchronize()
    step = _state["root_start"].elapsed_time(end)
    if step < _REPORT_MIN_MS:
        return
    qsa_total = tot.get("qsa_total", 0.0)
    idx = tot.get("qsa_indexer", 0.0)
    qsa_attn = max(qsa_total - idx, 0.0)
    gdn = tot.get("gdn", 0.0)
    moe = tot.get("moe", 0.0)
    ep = tot.get("moe_ep", 0.0)
    moe_compute = max(moe - ep, 0.0)
    gate = tot.get("moe_gate", 0.0)
    align = tot.get("moe_align", 0.0)
    marlin = tot.get("moe_marlin", 0.0)
    shared = tot.get("moe_shared", 0.0)
    shared_mlp = tot.get("moe_shared_mlp", 0.0)
    select = tot.get("moe_select", 0.0)
    modular = tot.get("moe_modular", 0.0)
    monolithic = tot.get("moe_monolithic", 0.0)
    ple = tot.get("ple", 0.0)
    ar = tot.get("allreduce", 0.0)
    other = max(step - (qsa_total + gdn + moe + ple + ar), 0.0)
    pc = lambda v: (100.0 * v / step) if step else 0.0
    logger.info(
        "[prefill-timing] nt=%d step=%.1fms | indexer=%.1f(%.0f%%) "
        "qsa_attn=%.1f(%.0f%%) | gdn=%.1f(%.0f%%) "
        "moe=%.1f(%.0f%%)(compute %.1f ep %.1f gate %.1f align %.1f marlin %.1f shared %.1f(+mlp %.1f) select %.1f modular %.1f monolithic %.1f) ple=%.1f(%.0f%%) "
        "allreduce=%.1f(%.0f%%) other=%.1f(%.0f%%)",
        _state["num_tokens"], step, idx, pc(idx), qsa_attn, pc(qsa_attn),
        gdn, pc(gdn), moe, pc(moe), moe_compute, ep, gate, align, marlin, shared, shared_mlp, select, modular, monolithic, ple, pc(ple), ar, pc(ar), other, pc(other),
    )


def _patch_allreduce() -> None:
    """Wrap the TP all-reduce so its GPU time is billed separately."""
    try:
        from vllm.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator,
        )
    except Exception as e:  # noqa: BLE001
        logger.info("[prefill-timing] cannot patch all_reduce: %s", e)
        return
    if getattr(CudaCommunicator.all_reduce, "_prefill_timed", False):
        return
    orig = CudaCommunicator.all_reduce

    def timed_all_reduce(self, input_):
        if not _ENABLED or _capturing():
            return orig(self, input_)
        s = torch.cuda.Event(enable_timing=True)
        s.record()
        try:
            return orig(self, input_)
        finally:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            _events.setdefault("allreduce", []).append((s, e))

    timed_all_reduce._prefill_timed = True  # type: ignore[attr-defined]
    CudaCommunicator.all_reduce = timed_all_reduce


def _timed_fn(orig, label):
    """Wrap a plain function with CUDA-event timing into ``label``."""

    def wrapper(*args, **kwargs):
        if not _ENABLED or _capturing():
            return orig(*args, **kwargs)
        s = torch.cuda.Event(enable_timing=True)
        s.record()
        try:
            return orig(*args, **kwargs)
        finally:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            _events.setdefault(label, []).append((s, e))

    wrapper._prefill_timed = True  # type: ignore[attr-defined]
    return wrapper


def _patch_moe_fns() -> None:
    """Time MoE compute pieces: align kernel and Marlin grouped GEMMs."""
    try:
        from vllm.model_executor.layers.fused_moe.experts import marlin_moe
    except Exception as e:  # noqa: BLE001
        logger.info("[prefill-timing] cannot patch moe fns: %s", e)
        return
    if getattr(marlin_moe._fused_marlin_moe, "_prefill_timed", False):
        return
    marlin_moe._fused_marlin_moe = _timed_fn(
        marlin_moe._fused_marlin_moe, "moe_marlin"
    )
    # NOTE: wrap the use-site reference (marlin_moe imported the name with
    # `from ... import`); patching the source module alone would not fire.
    try:
        from vllm.model_executor.layers.fused_moe import routed_experts as _rexp
        for _mname, _mlabel in (("forward_modular", "moe_modular"),
                                ("forward_monolithic", "moe_monolithic")):
            _m = getattr(_rexp.RoutedExperts, _mname, None)
            if _m is not None and not getattr(_m, "_prefill_timed", False):
                setattr(_rexp.RoutedExperts, _mname,
                        _timed_fn(_m, _mlabel))
    except (ImportError, AttributeError) as e:  # noqa: BLE001
        logger.info("[prefill-timing] cannot patch routed experts: %s", e)
    try:
        from vllm.model_executor.layers.fused_moe.router import fused_moe_router as _fmr
        if not getattr(_fmr.FusedMoERouter.select_experts, "_prefill_timed", False):
            _fmr.FusedMoERouter.select_experts = _timed_fn(
                _fmr.FusedMoERouter.select_experts, "moe_select"
            )
    except (ImportError, AttributeError) as e:  # noqa: BLE001
        logger.info("[prefill-timing] cannot patch moe select: %s", e)
    if getattr(marlin_moe.moe_align_block_size, "_prefill_timed", False):
        return
    marlin_moe.moe_align_block_size = _timed_fn(
        marlin_moe.moe_align_block_size, "moe_align"
    )
    try:
        marlin_moe.batched_moe_align_block_size = _timed_fn(
            marlin_moe.batched_moe_align_block_size, "moe_align"
        )
    except AttributeError:
        pass


def _patch_moe_ep() -> None:
    """Wrap EP dispatch/combine (AgRs allgather/reduce-scatter) separately.

    On pre-NVLink boxes the EP manager is AgRsAll2AllManager: dispatch =
    all_gatherv, combine = reduce_scatterv. moe_compute = moe module time
    minus this bucket (serial data dependency, no overlap expected).
    """
    try:
        from vllm.distributed.device_communicators.all2all import (
            AgRsAll2AllManager,
        )
    except Exception as e:  # noqa: BLE001
        logger.info("[prefill-timing] cannot patch moe_ep: %s", e)
        return
    if getattr(AgRsAll2AllManager.dispatch, "_prefill_timed", False):
        return
    orig_d = AgRsAll2AllManager.dispatch
    orig_c = AgRsAll2AllManager.combine

    def timed_ep(self, orig, *args, **kwargs):
        if not _ENABLED or _capturing():
            return orig(self, *args, **kwargs)
        s = torch.cuda.Event(enable_timing=True)
        s.record()
        try:
            return orig(self, *args, **kwargs)
        finally:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            _events.setdefault("moe_ep", []).append((s, e))

    def timed_dispatch(self, *args, **kwargs):
        return timed_ep(self, orig_d, *args, **kwargs)

    def timed_combine(self, *args, **kwargs):
        return timed_ep(self, orig_c, *args, **kwargs)

    timed_dispatch._prefill_timed = True  # type: ignore[attr-defined]
    timed_combine._prefill_timed = True  # type: ignore[attr-defined]
    AgRsAll2AllManager.dispatch = timed_dispatch
    AgRsAll2AllManager.combine = timed_combine


if _ENABLED:
    from torch.nn.modules.module import (
        register_module_forward_hook,
        register_module_forward_pre_hook,
    )

    register_module_forward_pre_hook(_gpre)
    register_module_forward_hook(_gpost)
    _patch_allreduce()
    _patch_moe_ep()
    _patch_moe_fns()
    logger.info("[prefill-timing] enabled (VLLM_QWEN4_PREFILL_TIMING=1)")
