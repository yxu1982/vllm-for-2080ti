# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in per-shape autotuning for Qwen4Exp Triton kernels on pre-Ampere GPUs.

Turing (sm75) has 64 KiB of shared memory per block, 255 registers per
thread and no cp.async, so the GB300-tuned tiles of the QSA, indexer and HC
kernels were replaced by fixed profiles chosen to *fit*, not for speed. This
module adds an opt-in layer that benchmarks a short, smem-checked candidate
list per coarse shape bucket and caches the winner. It never changes the
default launches:

* ``VLLM_QWEN4_TURING_AUTOTUNE=1`` enables it and only on CUDA devices below
  compute capability 8.0. Default off: every launch stays byte-identical to
  the fixed profile.
* Only the prefill-shaped path of each kernel is tuned. Decode launches keep
  the fixed profile because they are CUDA-graph captured.
* Nothing is benchmarked while a CUDA graph is being captured; the fixed
  profile is used (and not cached) in that case.
* Candidates whose static shared-memory estimate exceeds ``SMEM_BUDGET``
  (60 KiB) are dropped before compilation; candidates that still fail to
  compile or launch score +inf, like ``triton.autotune`` does.
* ``TRITON_PRINT_AUTOTUNING=1`` prints per-candidate timings in Triton's
  format; the winner is always logged once per key at INFO.

The benchmark deliberately avoids ``triton.testing.do_bench``: that helper
allocates a 256 MiB L2-flush buffer, which is not available once vLLM has
sized its KV cache (the first tuning event happens on the first real prefill,
after allocation). A 16 MiB flush buffer (RTX 8000 L2 is 6 MiB) is allocated
per tuning event and released afterwards.
"""

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any, TypeVar

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.triton_utils import triton

logger = init_logger(__name__)

ENV_FLAG = "VLLM_QWEN4_TURING_AUTOTUNE"

#: Static shared-memory budget for a candidate config. Turing reports
#: ``max_shared_mem=65536``; 60 KiB leaves headroom for Triton's scratch.
SMEM_BUDGET = 60 * 1024

_L2_FLUSH_BYTES = 16 * 1024 * 1024
_BENCH_BUDGET_MS = 40.0
_BENCH_MIN_REPS = 3
_BENCH_MAX_REPS = 20
# Correctness gate tolerances for candidate outputs vs the fixed profile (fp16
# kernel tests upstream use atol 2e-2).
_CHECK_ATOL = 2e-2
_CHECK_RTOL = 2e-2
# The correctness gate compares at most this many values per output tensor:
# a strided sample of the flattened output. The indexer prefill logits are
# num_queries x context wide (0.8 GB fp32 per copy at a 100K-token prompt);
# cloning, casting and boolean-indexing them whole OOMed the worker at 100K+
# (2026-09-25, RTX 8000 with the 5.07 GiB KV pool), taking the engine down.
_CHECK_MAX_ELEMS = 1 << 22


def _check_sample(t: torch.Tensor) -> torch.Tensor:
    """A bounded fp32 copy of ``t`` for the gate: every k-th value of the
    flattened tensor so that at most ``_CHECK_MAX_ELEMS`` are kept."""
    flat = t.detach().reshape(-1)
    n = flat.numel()
    if n > _CHECK_MAX_ELEMS:
        flat = flat[:: -(-n // _CHECK_MAX_ELEMS)]
    return flat.float().clone()


_PRE_AMPERE: bool | None = None
_ENABLED: bool | None = None

C = TypeVar("C", bound=Hashable)


def is_pre_ampere() -> bool:
    """True on CUDA GPUs below compute capability 8.0 (64 KiB smem/block)."""
    global _PRE_AMPERE
    if _PRE_AMPERE is None:
        try:
            from vllm.platforms import current_platform

            _PRE_AMPERE = bool(
                current_platform.is_cuda()
                and not current_platform.has_device_capability(80)
            )
        except Exception:
            _PRE_AMPERE = False
    return _PRE_AMPERE


def turing_autotune_enabled() -> bool:
    """``VLLM_QWEN4_TURING_AUTOTUNE=1`` on a pre-Ampere CUDA device."""
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = bool(envs.VLLM_QWEN4_TURING_AUTOTUNE) and is_pre_ampere()
        if _ENABLED:
            logger.debug(
                "%s=1: Qwen4Exp Turing per-shape autotuning is ON "
                "(prefill paths only; smem budget %d KiB).",
                ENV_FLAG,
                SMEM_BUDGET // 1024,
            )
    return _ENABLED


def in_graph_capture() -> bool:
    """True while the current CUDA stream is being captured into a graph."""
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def fits_smem(estimate_bytes: int) -> bool:
    return estimate_bytes <= SMEM_BUDGET


def bucket_pow2(n: int, lo: int, hi: int) -> int:
    """Round ``n`` up to a power of two, clamped to ``[lo, hi]``.

    Keys built from this are coarse on purpose: every distinct key costs one
    benchmarking event (a few seconds) on its first use.
    """
    return min(max(triton.next_power_of_2(max(int(n), 1)), lo), hi)


def bucket_steps(n: int, steps: Iterable[int]) -> int:
    """Smallest of ``steps`` that is >= ``n``; the last step if none is."""
    last = None
    for step in steps:
        last = step
        if n <= step:
            return step
    assert last is not None
    return last


def _print_autotuning() -> bool:
    return os.environ.get("TRITON_PRINT_AUTOTUNING", "0") == "1"


def _is_resource_error(exc: BaseException) -> bool:
    """Compile/launch failures that mean "this config does not fit"."""
    try:
        from triton.compiler.errors import CompilationError
        from triton.runtime.errors import OutOfResources

        if isinstance(exc, (OutOfResources, CompilationError)):
            return True
    except Exception:  # pragma: no cover - triton layout differences
        pass
    return isinstance(exc, (RuntimeError, torch.cuda.OutOfMemoryError))


class ShapeTuner:
    """Hand-rolled per-shape config cache for kernels launched via wrappers.

    ``triton.autotune`` cannot be used when the wrapper derives the grid,
    workspace or constexprs from the config, so the wrapper instead calls
    :meth:`choose` with a ``run(candidate)`` closure that performs the full
    launch (and is idempotent: every candidate writes the same outputs).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.cache: dict[Hashable, Any] = {}
        self._flush: torch.Tensor | None = None

    def choose(
        self,
        key: Hashable,
        candidates: list[C],
        fixed: C,
        run: Callable[[C], None],
        describe: Callable[[C], str] = str,
        outputs: Callable[[], Sequence[torch.Tensor]] | None = None,
    ) -> C:
        """Return the cached winner for ``key`` or benchmark ``candidates``.

        ``fixed`` is returned without caching when tuning is disabled, a CUDA
        graph is being captured, or there is nothing to choose between.

        ``outputs`` (optional) returns the tensors the launch writes. When given,
        the fixed profile is run once and its outputs kept as the reference;
        every candidate whose outputs differ from it beyond
        ``_CHECK_ATOL``/``_CHECK_RTOL`` (or carry non-finite values the
        reference does not) is rejected regardless of speed. A fast-but-wrong
        tile configuration is the documented failure mode of speed-only
        sweeps (vllm-project/vllm#55557's bf16 BLOCK_N=128), so the gate makes
        the selection safe by construction at a one-time cost per key.
        """
        if key in self.cache:
            return self.cache[key]
        if not turing_autotune_enabled() or in_graph_capture():
            return fixed
        if len(candidates) <= 1:
            best = candidates[0] if candidates else fixed
            self.cache[key] = best
            return best

        t0 = time.perf_counter()
        timings: dict[C, float] = {}
        try:
            self._flush = torch.empty(
                _L2_FLUSH_BYTES, dtype=torch.int8, device="cuda"
            )
            reference: list[torch.Tensor] | None = None
            # Sites without a fixed profile (fixed=None) take the first
            # candidate as the reference.
            ref_cfg = fixed if fixed is not None else candidates[0]
            if outputs is not None:
                try:
                    run(ref_cfg)
                    torch.cuda.synchronize()
                    reference = [
                        (tuple(t.shape), _check_sample(t)) for t in outputs()
                    ]
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Turing autotune %s key=%s: reference profile %s failed "
                        "the reference run (%s); candidates are not checked",
                        self.name,
                        key,
                        describe(ref_cfg),
                        exc,
                    )
            for candidate in candidates:
                timings[candidate] = self._bench(candidate, run, describe)
                if reference is not None and timings[candidate] != float("inf"):
                    if not self._matches(candidate, run, outputs, reference):
                        logger.warning(
                            "Turing autotune %s key=%s: candidate %s rejected on "
                            "correctness (output differs from the reference profile "
                            "beyond atol %.0e / rtol %.0e)",
                            self.name,
                            key,
                            describe(candidate),
                            _CHECK_ATOL,
                            _CHECK_RTOL,
                        )
                        timings[candidate] = float("inf")
            if reference is not None:
                # leave the buffers holding the reference profile's result
                try:
                    run(ref_cfg)
                except Exception:  # noqa: BLE001
                    pass
        finally:
            self._flush = None
        best = min(timings, key=timings.__getitem__)
        if timings[best] == float("inf"):
            best = fixed
        self.cache[key] = best
        elapsed = time.perf_counter() - t0
        logger.debug(
            "Triton autotuning for function %s finished after %.2fs; "
            "best config selected: %s;",
            self.name,
            elapsed,
            describe(best),
        )
        for candidate, ms in timings.items():
            logger.debug("  %s: %.4f ms", describe(candidate), ms)
        logger.debug(
            "Turing autotune %s key=%s: selected %s (%.4f ms; fixed %s at "
            "%s ms; %d candidates in %.1fs)",
            self.name,
            key,
            describe(best),
            timings[best],
            describe(fixed),
            (
                f"{timings[fixed]:.4f}"
                if fixed in timings and timings[fixed] != float("inf")
                else "n/a"
            ),
            len(candidates),
            elapsed,
        )
        return best

    def _matches(
        self,
        candidate: C,
        run: Callable[[C], None],
        outputs: Callable[[], Sequence[torch.Tensor]],
        reference: list[tuple[tuple[int, ...], torch.Tensor]],
    ) -> bool:
        try:
            run(candidate)
            torch.cuda.synchronize()
            got = list(outputs())
        except Exception:  # noqa: BLE001
            return False
        if len(got) != len(reference):
            return False
        for a, (shape, bf) in zip(got, reference):
            if tuple(a.shape) != shape:
                return False
            af = _check_sample(a)
            finite = torch.isfinite(bf)
            if not torch.equal(torch.isfinite(af), finite):
                return False
            if not torch.allclose(
                af[finite], bf[finite], atol=_CHECK_ATOL, rtol=_CHECK_RTOL
            ):
                return False
        return True

    def _bench(self, candidate: C, run: Callable[[C], None], describe) -> float:
        try:
            run(candidate)  # compile + warm
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run(candidate)
            end.record()
            end.synchronize()
            once_ms = max(start.elapsed_time(end), 1e-3)
            reps = int(_BENCH_BUDGET_MS / once_ms)
            reps = min(max(reps, _BENCH_MIN_REPS), _BENCH_MAX_REPS)
            samples: list[float] = []
            for _ in range(reps):
                if self._flush is not None:
                    self._flush.zero_()
                start.record()
                run(candidate)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            return statistics.median(samples)
        except Exception as exc:  # noqa: BLE001 - scored, not raised
            if not _is_resource_error(exc):
                raise
            logger.warning(
                "Turing autotune %s: candidate %s skipped (%s: %s)",
                self.name,
                describe(candidate),
                type(exc).__name__,
                str(exc).splitlines()[0][:120] if str(exc) else "",
            )
            return float("inf")


__all__ = [
    "ENV_FLAG",
    "SMEM_BUDGET",
    "ShapeTuner",
    "bucket_pow2",
    "bucket_steps",
    "fits_smem",
    "in_graph_capture",
    "is_pre_ampere",
    "turing_autotune_enabled",
]
