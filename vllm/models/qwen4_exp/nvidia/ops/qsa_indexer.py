# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for Qwen4Exp QSA index selection."""

import os

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .turing_autotune import (
    ShapeTuner,
    bucket_steps,
    fits_smem,
    is_pre_ampere,
    turing_autotune_enabled,
)

logger = init_logger(__name__)

_TOPK_WORKSPACE_BYTES = 1024 * 1024
_DECODE_BLOCK_N = 64

# Opt-in Turing autotune (VLLM_QWEN4_TURING_AUTOTUNE=1) for the prefill scoring
# kernel _qsa_mqa_paged_prefill_kernel. Candidates are (TILE_R, BLOCK_N,
# num_warps); STAGES and K_TILES keep the fixed profile's values (the RTX 8000
# probe shows STAGES changes neither smem nor registers on sm75, and K_TILES
# only re-splits the grid). TILE_R=32 never fits (80-96 KiB measured) and is
# dropped by the estimate; TILE_R=16/BLOCK_N=64 is exactly 64 KiB with
# 494-814 spills and is dropped by the 60 KiB budget.
_TURING_PREFILL_TILE_R = (8, 16, 32)
_TURING_PREFILL_BLOCK_N = (32, 64)
_TURING_PREFILL_WARPS = (4, 8)
# Coarse key buckets: rows per request tile-loop and logits width (columns).
_TURING_PREFILL_ROW_BUCKETS = (128, 512, 2048, 1 << 30)
_TURING_PREFILL_WIDTH_BUCKETS = (1024, 4096, 16384, 1 << 30)
_TURING_PREFILL_TUNER = ShapeTuner("_qsa_mqa_paged_prefill_kernel")


def _turing_prefill_smem_bytes(
    tile_r: int, block_n: int, num_heads: int, head_dim: int, elem_size: int
) -> int:
    """Static shared-memory estimate of the prefill scoring kernel on sm75.

    Fitted to all 36 rows of the RTX 8000 probe (Triton 3.7, 4 heads, D=128,
    fp16): 2 KiB * TILE_R + 512 B * BLOCK_N, i.e. both dot operands (the
    resident Q tile [D, TILE_R*heads] and the K tile [BLOCK_N, D]) are staged
    twice, independent of STAGES.
    """
    heads_padded = triton.next_power_of_2(num_heads)
    block_d = max(16, triton.next_power_of_2(head_dim))
    q_tile = block_d * tile_r * heads_padded * elem_size
    k_tile = block_n * block_d * elem_size
    return 2 * (q_tile + k_tile)


def _turing_prefill_candidates(
    num_heads: int, head_dim: int, elem_size: int, fixed: tuple[int, int, int]
) -> list[tuple[int, int, int]]:
    """(TILE_R, BLOCK_N, num_warps) candidates within budget; fixed first."""
    candidates = [fixed]
    for tile_r in _TURING_PREFILL_TILE_R:
        for block_n in _TURING_PREFILL_BLOCK_N:
            if not fits_smem(
                _turing_prefill_smem_bytes(
                    tile_r, block_n, num_heads, head_dim, elem_size
                )
            ):
                continue
            for num_warps in _TURING_PREFILL_WARPS:
                candidate = (tile_r, block_n, num_warps)
                if candidate not in candidates:
                    candidates.append(candidate)
    return candidates


def _describe_prefill(candidate: tuple[int, int, int]) -> str:
    return (
        f"TILE_R={candidate[0]}, BLOCK_N={candidate[1]}, "
        f"num_warps={candidate[2]}"
    )


@triton.jit
def _qsa_mqa_paged_uniform_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DECODE_QUERY_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
) -> None:
    NUM_COLUMNS: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE
    DECODE_QUERY_LEN_PADDED: tl.constexpr = triton.next_power_of_2(DECODE_QUERY_LEN)
    NUM_HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    # tl.dot requires a reduction dimension of at least 16.
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    request = tl.program_id(0)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    query_offsets = tl.arange(0, DECODE_QUERY_LEN_PADDED)
    valid_query_offsets = query_offsets < DECODE_QUERY_LEN
    rows = request * DECODE_QUERY_LEN + query_offsets
    visible = tl.load(
        visible_blocks_ptr + rows,
        mask=valid_query_offsets,
        other=0,
    )
    max_visible = tl.max(visible, axis=0)
    if tile_start * BLOCK_N >= max_visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(max_visible, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(NUM_COLUMNS, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    n = tl.arange(0, DECODE_QUERY_LEN_PADDED * NUM_HEADS_PADDED)
    query_offset = n // NUM_HEADS_PADDED
    head = n % NUM_HEADS_PADDED
    valid_query = (query_offset < DECODE_QUERY_LEN) & (head < NUM_HEADS)
    query = tl.load(
        q_ptr
        + (request * DECODE_QUERY_LEN + query_offset)[None, :] * stride_q_row
        + head[None, :] * stride_q_head
        + dims[:, None],
        mask=valid_query[None, :] & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr + request * stride_table_req + logical_page,
            mask=live,
            other=0,
        )
        keys = tl.load(
            k_cache_ptr
            + physical_page[:, None].to(tl.int64) * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :],
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(valid_query[None, :], tl.maximum(scores, 0.0), 0.0)
        scores = tl.reshape(
            scores,
            (BLOCK_N, DECODE_QUERY_LEN_PADDED, NUM_HEADS_PADDED),
        )
        score = tl.sum(scores, axis=2)
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            score,
            mask=valid_query_offsets[None, :]
            & (columns[:, None] < NUM_COLUMNS)
            & (columns[:, None] < visible[None, :]),
        )


@triton.jit(do_not_specialize=["num_rows", "query_offset"])
def _qsa_mqa_paged_prefill_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    query_start_loc_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    num_rows,
    query_offset,
    page_table_width,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
) -> None:
    num_columns = page_table_width * PAGE_SIZE
    NUM_HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    # tl.dot requires a reduction dimension of at least 16.
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    request = tl.program_id(0)
    query_base = tl.load(query_start_loc_ptr)
    query_end = query_offset + num_rows
    request_start = tl.maximum(
        tl.load(query_start_loc_ptr + request) - query_base, query_offset
    )
    request_end = tl.minimum(
        tl.load(query_start_loc_ptr + request + 1) - query_base, query_end
    )
    absolute_row_start = request_start + tl.program_id(1) * TILE_R
    if absolute_row_start >= request_end:
        return

    lanes = tl.arange(0, TILE_R)
    absolute_rows = absolute_row_start + lanes
    rows = absolute_rows - query_offset
    valid_rows = absolute_rows < request_end
    visible = tl.load(
        visible_blocks_ptr + absolute_rows,
        mask=valid_rows,
        other=0,
    )
    max_visible = tl.max(visible, axis=0)
    k_tile_start = tl.program_id(2) * K_TILES
    if k_tile_start * BLOCK_N >= max_visible:
        return
    k_tile_end = tl.minimum(k_tile_start + K_TILES, tl.cdiv(max_visible, BLOCK_N))
    k_tile_end = tl.minimum(k_tile_end, tl.cdiv(num_columns, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    m = tl.arange(0, TILE_R * NUM_HEADS_PADDED)
    q_row_offsets = m // NUM_HEADS_PADDED
    q_rows = absolute_row_start + q_row_offsets
    heads = m % NUM_HEADS_PADDED
    query = tl.load(
        q_ptr
        + q_rows[None, :] * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None],
        mask=(heads[None, :] < NUM_HEADS)
        & (absolute_row_start + q_row_offsets[None, :] < request_end)
        & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(k_tile_start, k_tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, page_table_width - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr + request * stride_table_req + logical_page,
            mask=live,
            other=0,
        )
        keys = tl.load(
            k_cache_ptr
            + physical_page[:, None].to(tl.int64) * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :],
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        # With an fp8 cache, SM90 lowers this dot to reduced-precision wgmma
        # accumulation; SM100 tcgen05 accumulates in fp32. The SM90 error
        # (~1e-4 relative) is far below the e4m3 quantization noise floor.
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.reshape(scores, (BLOCK_N, TILE_R, NUM_HEADS_PADDED))
        score = tl.sum(tl.maximum(scores, 0.0), axis=2)
        store_mask = (
            valid_rows[None, :]
            & (columns[:, None] < visible[None, :])
            & (columns[:, None] < num_columns)
        )
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            score,
            mask=store_mask,
        )


@triton.jit
def _expand_qsa_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    visible_blocks_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
) -> None:
    OUTPUT_WIDTH: tl.constexpr = BLOCK_TOPK * COMPRESS_RATIO + COMPRESS_RATIO - 1
    row = tl.program_id(0)
    columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
    query_position = tl.load(query_positions_ptr + row)
    visible_blocks = tl.load(visible_blocks_ptr + row)
    complete_blocks = tl.minimum(visible_blocks, BLOCK_TOPK)
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_rank * stride_blocks_column,
        mask=is_expanded,
        other=-1,
    )
    expanded = block * COMPRESS_RATIO + offset
    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count)
        & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    token = tl.where(is_expanded, expanded, tail_start + tail_offset)
    valid = (columns < OUTPUT_WIDTH) & (is_expanded | is_tail) & (token >= 0)
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        tl.where(valid, token, -1),
        mask=columns < OUTPUT_WIDTH,
    )
    # TRAILING COUNT COLUMN: the packed buffer is OUTPUT_WIDTH+1 wide; the
    # last column holds this row's valid-entry count (expanded blocks plus
    # causal tail). It is NOT a token index — the sparse attention kernel
    # reads it as the row's tile-loop bound.
    if tl.program_id(1) == 0:
        tl.store(
            output_ptr + row * stride_output_row + OUTPUT_WIDTH * stride_output_column,
            expanded_count + tail_count,
        )


# Pre-Ampere (sm75) profile of the uniform decode scoring kernel. The default
# profile below (BLOCK_N=64, STAGES=2, num_warps=2, one 64-column tile per
# program in the small-batch region) was tuned on GB300; on the RTX 8000 it is
# latency-bound. tools/decode_probe2.py (4 heads, D=128, page 400, page-table
# width 164, MTP K=2 -> 3 query rows per request, 20 candidates checked against
# the default's output within 2e-2) measures, in us, default -> BLOCK_N=32 with
# one warp and 4 tiles per program: 1 request 226 -> 53 (64K) / 332 -> 79 (128K);
# 2 requests 285 -> 83 / 524 -> 144; 4 requests 506 -> 139 / 993 -> 256;
# 8 requests 1042 -> 262 / 2054 -> 494; 12 requests 1536 -> 440 / 2974 -> 705.
# 4 tiles is best or within 5% of the best at every (requests, context) row
# from 16K to 128K and for 1 or 3 query rows; 2 tiles wins only by a few us at
# 16K. BLOCK_N=128 does not compile on sm75. The kernel runs once per QSA
# layer per decode step (13 launches with the MTP draft), so the default
# costs ~2.5 ms of a 27 ms two-stream decode step and ~5 ms at four streams.
_TURING_DECODE_BLOCK_N = 32
_TURING_DECODE_TILES = 4
_TURING_DECODE_WARPS = 1


def _decode_profile(
    num_requests: int, columns: int, cache_dtype: torch.dtype
) -> tuple[int, int, int]:
    """(BLOCK_N, TILES_PER_PROG, num_warps) of the uniform decode kernel."""
    if is_pre_ampere():
        return _TURING_DECODE_BLOCK_N, _TURING_DECODE_TILES, _TURING_DECODE_WARPS
    return (
        _DECODE_BLOCK_N,
        _decode_tiles_per_program(num_requests, columns),
        # tuned on GB300
        1 if cache_dtype == torch.float8_e4m3fn else 2,
    )


def _decode_tiles_per_program(num_requests: int, columns: int) -> int:
    programs = num_requests * triton.cdiv(columns, _DECODE_BLOCK_N)
    if programs < 16384:
        return 1
    if programs < 32768:
        return 2
    if programs < 131072:
        return 4
    return 8


def _qsa_decode_warmup_profiles(
    max_dql: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
    columns: int,
) -> tuple[tuple[int, int], ...]:
    profiles: list[tuple[int, int]] = []
    for dql in range(1, max_dql + 1):
        max_reqs = min(max_num_reqs, max_num_batched_tokens // dql)
        requests_by_grouping: dict[int, int] = {}
        for num_requests in range(1, max_reqs + 1):
            tiles_per_program = _decode_tiles_per_program(num_requests, columns)
            requests_by_grouping.setdefault(tiles_per_program, num_requests)
        profiles.extend(
            (dql, num_requests) for num_requests in requests_by_grouping.values()
        )
    return tuple(profiles)


def warmup_qsa_mqa_paged_decode(
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    max_decode_query_len: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
) -> tuple[tuple[int, int], ...]:
    """Compile every reachable decode specialization without launching it."""
    page_size = k_cache.shape[1]
    page_table_width = page_table.shape[1]
    columns = page_table_width * page_size
    profiles = _qsa_decode_warmup_profiles(
        max_decode_query_len,
        max_num_reqs,
        max_num_batched_tokens,
        columns,
    )
    if not profiles:
        return ()

    k_cache_ptr = TritonWarmupTensor(k_cache.dtype, shape=tuple(k_cache.shape))
    page_table_ptr = TritonWarmupTensor(
        page_table.dtype,
        shape=(max_num_reqs, page_table_width),
    )
    visible_blocks_ptr = TritonWarmupTensor(torch.int32)

    for decode_query_len, num_requests in profiles:
        num_rows = decode_query_len * num_requests
        q_ptr = TritonWarmupTensor(
            k_cache.dtype,
            shape=(num_rows, num_heads, head_dim),
        )
        logits_ptr = TritonWarmupTensor(
            torch.float32,
            shape=(num_rows, columns),
        )
        block_n, tiles_per_program, num_warps = _decode_profile(
            num_requests, columns, k_cache.dtype
        )
        _qsa_mqa_paged_uniform_kernel.warmup(
            q_ptr,
            k_cache_ptr,
            page_table_ptr,
            visible_blocks_ptr,
            logits_ptr,
            num_heads * head_dim,
            head_dim,
            k_cache.stride(0),
            k_cache.stride(1),
            page_table.stride(0),
            columns,
            PAGE_SIZE=page_size,
            PAGE_TABLE_WIDTH=page_table_width,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            DECODE_QUERY_LEN=decode_query_len,
            BLOCK_N=block_n,
            TILES_PER_PROG=tiles_per_program,
            STAGES=2,
            num_warps=num_warps,
            grid=(
                num_requests,
                triton.cdiv(columns, block_n * tiles_per_program),
            ),
        )
    return profiles


# Dense cuBLAS indexer scoring (SM75-tuned port of the 1cat SM70 path):
# fp16 inputs with fp32 accumulation (Tensor Core HMMA; the q/k inputs are
# already fp16 so no precision is lost vs an fp32 upcast), one fused
# ReLU+headsum+mask epilogue per tile (instead of three torch passes), and
# column tiling to bound transient memory on long contexts.
_INDEXER_CUBLAS_TILE_BYTES = (
    int(os.environ.get("VLLM_QWEN4_INDEXER_CUBLAS_TILE_MB", "64")) * 1024 * 1024
)
_INDEXER_CUBLAS_EPILOGUE_BN = int(
    os.environ.get("VLLM_QWEN4_INDEXER_CUBLAS_EPILOGUE_BN", "256")
)


@triton.jit
def _qsa_cublas_epilogue_kernel(
    score_ptr,
    visible_ptr,
    logits_ptr,
    stride_score_row,
    stride_score_col,
    stride_logits_row,
    stride_logits_col,
    width,
    col0,
    NUM_HEADS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused ReLU + head-sum + visible-mask for one cuBLAS score tile.

    scores is (rows*NUM_HEADS, tile_width) fp32; writes one fp32 logit per
    (row, absolute column). Dead columns get -inf (never selected by top-k;
    strictly safer than 0.0, which ties with genuine zero-score blocks).
    No score rescaling: the downstream top-k only consumes the ranking, and
    the divisor (sqrt(head_dim) > 0) is ranking-invariant.
    """
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    abs_cols = col0 + cols
    vis = tl.load(visible_ptr + row)
    live = (cols < width) & (abs_cols < vis)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for h in range(NUM_HEADS):
        v = tl.load(
            score_ptr
            + (row * NUM_HEADS + h) * stride_score_row
            + cols * stride_score_col,
            mask=cols < width,
            other=0.0,
        ).to(tl.float32)
        acc += tl.maximum(v, 0.0)
    tl.store(
        logits_ptr + row * stride_logits_row + abs_cols * stride_logits_col,
        tl.where(live, acc, -float("inf")),
        mask=cols < width,
    )


_PERSISTENT_LOGITS: dict = {}


_TOPK_WORKSPACE: dict = {}


def _topk_workspace(device):
    buf = _TOPK_WORKSPACE.get(device)
    if buf is None or buf.numel() < _TOPK_WORKSPACE_BYTES:
        buf = torch.empty((_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=device)
        _TOPK_WORKSPACE[device] = buf
    return buf



def _indexer_cublas_enabled() -> bool:
    return os.environ.get("VLLM_QWEN4_INDEXER_CUBLAS", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


@triton.jit
def _qsa_gather_map_kernel(
    page_table_ptr,
    blk_ptr,
    off_ptr,
    pages_needed,
    page_size,
    k_max,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < k_max
    pg = tl.minimum(i // page_size, pages_needed - 1)
    phys = tl.load(page_table_ptr + pg, mask=m, other=0).to(tl.int64)
    tl.store(blk_ptr + i, phys, mask=m)
    tl.store(off_ptr + i, (i % page_size).to(tl.int64), mask=m)


def _prefill_logits_cublas(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    visible_blocks: torch.Tensor,
    query_offset: int,
    num_queries: int,
    logits: torch.Tensor,
    logits_width: int,
) -> bool:
    """Fill `logits` via dense cuBLAS scoring; return False to fall back.

    Eligible only for the single-request fp16 prefill case (4 indexer heads,
    head_dim 128). Gathers the request's pooled keys densely once, scores
    with fp32 cuBLAS GEMM, applies ReLU + head-sum in torch, and masks
    columns beyond each row's visible count. Numerically equivalent to the
    Triton paged kernel up to GEMM accumulation order.
    """
    if q.dtype != torch.float16 or k_cache.dtype != torch.float16:
        return False
    if q.shape[1] != 4 or q.shape[2] != 128:
        return False
    if page_table.shape[0] != 1:
        return False
    if k_cache.ndim != 4 or k_cache.shape[2] != 1 or k_cache.shape[3] != 128:
        return False
    page_size = int(k_cache.shape[1])
    if page_size <= 0:
        return False
    # Only pay gather + GEMM launch when there is enough work; tiny chunks
    # (decode, short prefill tails) stay on the Triton path.
    _min_elems = int(
        __import__("os").environ.get("VLLM_QWEN4_INDEXER_CUBLAS_MIN_ELEMENTS", "1048576")
    )
    # Stage 1 (no D2H): the full width bounds k_max from above; if even that
    # is too small, return before paying the visible-max sync or the int64
    # slice copy below.
    if num_queries * logits_width < _min_elems:
        return False
    vis = visible_blocks[query_offset : query_offset + num_queries].to(torch.int64)
    # Stage 2: exact visible max costs one D2H sync; only reached when the
    # dense path is plausible.
    if int(vis.numel()) == 0:
        return True
    k_max = int(vis.max().item())
    if k_max <= 0:
        logits[:num_queries, :].zero_()
        return True
    k_max = min(k_max, logits_width)
    if num_queries * k_max < _min_elems:
        return False
    # Gather first k_max pooled keys: logical key i -> page i//page_size,
    # offset i%page_size, physical page_table[0, page].
    pages_needed = min(
        (k_max + page_size - 1) // page_size, int(page_table.shape[1])
    )
    if pages_needed * page_size < k_max:
        return False
    # Fused logical->physical map (single pass replacing the arange +
    # div + mod + clamp + int64-copy launch sequence). Single-request only
    # (gated above), so row 0 of the table is the only row.
    blk = torch.empty((k_max,), dtype=torch.int64, device=q.device)
    off = torch.empty((k_max,), dtype=torch.int64, device=q.device)
    _qsa_gather_map_kernel[(triton.cdiv(k_max, 1024),)](
        page_table, blk, off, pages_needed, page_size, k_max, BLOCK=1024
    )
    # fp16 in / fp32 out: cuBLAS HMMA with fp32 accumulation. The q/k
    # inputs are already fp16, so skipping the fp32 upcast loses nothing and
    # roughly doubles GEMM throughput on SM75 Tensor Cores.
    keys_dense = k_cache[blk, off][:, 0, :].contiguous()
    q2 = q[query_offset : query_offset + num_queries].reshape(-1, 128)
    rows4 = q2.shape[0]  # num_queries * 4 indexer heads
    bytes_per_col = max(1, rows4 * torch.float32.itemsize)
    tile_cols = max(256, min(k_max, _INDEXER_CUBLAS_TILE_BYTES // bytes_per_col))
    tile_cols = max(256, tile_cols // 256 * 256)
    bn = _INDEXER_CUBLAS_EPILOGUE_BN
    for col0 in range(0, k_max, tile_cols):
        col1 = min(col0 + tile_cols, k_max)
        scores = torch.mm(
            q2, keys_dense[col0:col1].t(), out_dtype=torch.float32
        )
        _qsa_cublas_epilogue_kernel[(num_queries, triton.cdiv(col1 - col0, bn))](
            scores,
            vis,
            logits,
            scores.stride(0),
            scores.stride(1),
            logits.stride(0),
            logits.stride(1),
            col1 - col0,
            col0,
            NUM_HEADS=4,
            BLOCK_N=bn,
        )
    if k_max < logits_width:
        logits[:num_queries, k_max:logits_width].zero_()
    return True


def _prefill_logits(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    max_query_len: int,
    logits_width: int,
    query_offset: int,
    num_queries: int,
) -> torch.Tensor:
    assert query_start_loc.shape == (page_table.shape[0] + 1,)
    assert visible_blocks.shape == (q.shape[0],)
    assert 0 <= query_offset <= query_offset + num_queries <= q.shape[0]
    assert 0 < logits_width <= page_table.shape[1] * k_cache.shape[1]

    if (
        os.environ.get("VLLM_QWEN4_INDEXER_PERSISTENT_LOGITS", "0").strip().lower()
        in ("1", "true", "yes", "on")
    ):
        # One persistent fp32 buffer per device, grown to the largest shape
        # seen (bounded by VLLM_SPARSE_INDEXER_MAX_LOGITS_MB): the per-call
        # torch.empty at a context-dependent width left the caching allocator
        # holding one block per distinct size (vllm-project/vllm#56457).
        need = num_queries * logits_width
        buf = _PERSISTENT_LOGITS.get(q.device)
        if buf is None or buf.numel() < need:
            cap = max(need, envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024 // 4)
            buf = torch.empty((cap,), dtype=torch.float32, device=q.device)
            _PERSISTENT_LOGITS[q.device] = buf
        logits = buf[:need].view(num_queries, logits_width)
    else:
        logits = torch.empty(
            (num_queries, logits_width), dtype=torch.float32, device=q.device
        )
    if _indexer_cublas_enabled() and _prefill_logits_cublas(
        q,
        k_cache,
        page_table,
        visible_blocks,
        query_offset,
        num_queries,
        logits,
        logits_width,
    ):
        return logits
    pre_ampere = not current_platform.has_device_capability(80)
    if pre_ampere:
        # Turing (sm75): 64 KiB smem/block. Measured on Quadro RTX 8000
        # (Triton 3.7): TILE_R=32 needs 80-96 KiB for any BLOCK_N (the fp32
        # [BLOCK_N, TILE_R*heads] score tile is staged through smem for the
        # head reduction); TILE_R=16/BLOCK_N=32 compiles at 48 KiB with 72
        # registers and no spills, the largest zero-spill profile that fits.
        TILE_R, BLOCK_N, STAGES, num_warps = 16, 32, 2, 8
    elif k_cache.dtype == torch.float8_e4m3fn:
        # tuned on GB300
        TILE_R, BLOCK_N, STAGES, num_warps = 32, 64, 2, 8
    else:
        # tuned on GB300
        TILE_R, BLOCK_N, STAGES, num_warps = 64, 64, 2, 4
    K_TILES = 16
    rows = min(num_queries, max_query_len)

    def _run(candidate: tuple[int, int, int]) -> None:
        # Idempotent: every candidate stores the same masked logits.
        tile_r, block_n, warps = candidate
        grid = (
            page_table.shape[0],
            triton.cdiv(rows, tile_r),
            triton.cdiv(logits_width, block_n * K_TILES),
        )
        _qsa_mqa_paged_prefill_kernel[grid](
            q,
            k_cache,
            page_table,
            query_start_loc,
            visible_blocks,
            logits,
            *q.stride()[:-1],
            *k_cache.stride()[:2],
            *page_table.stride()[:-1],
            *logits.stride()[:-1],
            num_queries,
            query_offset,
            page_table.shape[1],
            PAGE_SIZE=k_cache.shape[1],
            NUM_HEADS=q.shape[1],
            HEAD_DIM=q.shape[2],
            TILE_R=tile_r,
            BLOCK_N=block_n,
            K_TILES=K_TILES,
            STAGES=STAGES,
            num_warps=warps,
        )

    if pre_ampere and turing_autotune_enabled():
        fixed = (TILE_R, BLOCK_N, num_warps)
        key = (
            bucket_steps(rows, _TURING_PREFILL_ROW_BUCKETS),
            bucket_steps(logits_width, _TURING_PREFILL_WIDTH_BUCKETS),
            q.dtype,
            q.shape[1],
            q.shape[2],
            k_cache.shape[1],
        )
        candidates = _turing_prefill_candidates(
            q.shape[1], q.shape[2], q.element_size(), fixed
        )
        TILE_R, BLOCK_N, num_warps = _TURING_PREFILL_TUNER.choose(
            key, candidates, fixed, _run, _describe_prefill,
            outputs=lambda: (logits,),
        )
    _run((TILE_R, BLOCK_N, num_warps))
    return logits


def warmup_qsa_mqa_paged_prefill_turing_candidates(
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> tuple[tuple[int, int, int], ...]:
    """Compile the opt-in Turing prefill candidates without launching them.

    No-op unless VLLM_QWEN4_TURING_AUTOTUNE=1 on a pre-Ampere GPU. Runs at
    engine warmup so the first prefill of each shape bucket only benchmarks
    already-compiled kernels.
    """
    if not turing_autotune_enabled():
        return ()
    fixed = (16, 32, 8)  # the sm75 profile in _prefill_logits
    candidates = _turing_prefill_candidates(
        num_heads, head_dim, k_cache.dtype.itemsize, fixed
    )
    page_size = k_cache.shape[1]
    page_table_width = page_table.shape[1]
    columns = page_table_width * page_size
    num_rows = 64
    q_ptr = TritonWarmupTensor(k_cache.dtype, shape=(num_rows, num_heads, head_dim))
    k_cache_ptr = TritonWarmupTensor(k_cache.dtype, shape=tuple(k_cache.shape))
    page_table_ptr = TritonWarmupTensor(
        page_table.dtype, shape=(page_table.shape[0], page_table_width)
    )
    query_start_loc_ptr = TritonWarmupTensor(torch.int32)
    visible_blocks_ptr = TritonWarmupTensor(torch.int32)
    logits_ptr = TritonWarmupTensor(torch.float32, shape=(num_rows, columns))
    compiled = []
    for tile_r, block_n, num_warps in candidates:
        try:
            _qsa_mqa_paged_prefill_kernel.warmup(
                q_ptr,
                k_cache_ptr,
                page_table_ptr,
                query_start_loc_ptr,
                visible_blocks_ptr,
                logits_ptr,
                num_heads * head_dim,
                head_dim,
                k_cache.stride(0),
                k_cache.stride(1),
                page_table.stride(0),
                columns,
                num_rows,
                0,
                page_table_width,
                PAGE_SIZE=page_size,
                NUM_HEADS=num_heads,
                HEAD_DIM=head_dim,
                TILE_R=tile_r,
                BLOCK_N=block_n,
                K_TILES=16,
                STAGES=2,
                num_warps=num_warps,
                grid=(1, 1, 1),
            )
        except Exception as exc:
            logger.warning(
                "Turing autotune candidate %s skipped at warmup: %s",
                _describe_prefill((tile_r, block_n, num_warps)),
                str(exc).splitlines()[0][:120] if str(exc) else type(exc),
            )
            continue
        compiled.append((tile_r, block_n, num_warps))
    return tuple(compiled)


def expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    visible_blocks: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor,
) -> None:
    """Expand compressed blocks and compact the causal tail of the open group."""
    assert token_topk % compress_ratio == 0
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    assert block_indices.shape == (query_positions.numel(), block_topk)
    assert visible_blocks.shape == query_positions.shape
    # +1: the packed buffer's trailing column holds each row's valid-entry
    # count (never a token index); the index region below only writes
    # columns [0, output_width).
    assert out.shape == (block_indices.shape[0], output_width + 1)
    column_block = 256
    grid = (block_indices.shape[0], triton.cdiv(output_width, column_block))
    _expand_qsa_indices_kernel[grid](
        block_indices,
        query_positions,
        visible_blocks,
        out,
        *block_indices.stride(),
        *out.stride(),
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        COLUMN_BLOCK=column_block,
        num_warps=4,
    )


def _topk(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    block_indices: torch.Tensor,
    topk_workspace: torch.Tensor,
) -> None:
    # similar dispatch logic as DeepSeek indexer
    block_topk = token_topk // compress_ratio
    use_cooperative_topk = (
        logits.shape[0] <= 64
        and logits.stride(0) % 4 == 0
        and current_platform.has_device_capability(90)
        and not current_platform.is_device_capability_family(120)
    )
    topk_op = (
        torch.ops._C.cooperative_topk
        if use_cooperative_topk
        else torch.ops._C.persistent_topk
    )
    topk_op(
        logits,
        visible_blocks,
        block_indices,
        topk_workspace,
        block_topk,
        logits.shape[1],
    )


def qsa_select_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    decode_query_len: int,
    block_indices: torch.Tensor,
) -> None:
    """Score and select compressed blocks for a request-major decode batch.

    Args:
        q: Query tensor shaped ``[num_requests * decode_query_len, heads,
            head_dim]``.
        k_cache: Compressed key cache shaped ``[blocks, page_size, 1,
            head_dim]``.
        page_table: Request block table shaped ``[num_requests, max_pages]``.
        visible_blocks: Number of visible compressed blocks per query.
        token_topk: Number of logical tokens selected per query.
        compress_ratio: Number of logical tokens represented by a cache row.
        decode_query_len: Number of query tokens per request.
        block_indices: Compressed-index output buffer.

    """
    assert token_topk % compress_ratio == 0
    assert block_indices.shape == (q.shape[0], token_topk // compress_ratio)
    assert decode_query_len > 0 and q.shape[0] % decode_query_len == 0
    assert q.dtype == k_cache.dtype, "Q and the compressed K cache must match"
    num_requests = q.shape[0] // decode_query_len
    assert page_table.shape[0] == num_requests
    assert visible_blocks.shape == (q.shape[0],)

    columns = page_table.shape[1] * k_cache.shape[1]
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    block_n, tiles_per_program, num_warps = _decode_profile(
        num_requests, columns, k_cache.dtype
    )
    grid = (
        num_requests,
        triton.cdiv(columns, block_n * tiles_per_program),
    )
    _qsa_mqa_paged_uniform_kernel[grid](
        q,
        k_cache,
        page_table,
        visible_blocks,
        logits,
        *q.stride()[:-1],
        *k_cache.stride()[:2],
        *page_table.stride()[:-1],
        *logits.stride()[:-1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        DECODE_QUERY_LEN=decode_query_len,
        BLOCK_N=block_n,
        TILES_PER_PROG=tiles_per_program,
        STAGES=2,
        num_warps=num_warps,
    )
    _topk(
        logits,
        visible_blocks,
        token_topk,
        compress_ratio,
        block_indices,
        _topk_workspace(q.device),
    )


def qsa_select_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    max_query_len: int,
    block_indices: torch.Tensor,
    max_seq_len: int,
) -> None:
    """Score and select compressed prefill blocks in bounded chunks.

    Args:
        q: Packed prefill query tensor shaped ``[num_tokens, heads, head_dim]``.
        k_cache: Compressed key cache shaped ``[blocks, page_size, 1,
            head_dim]``.
        page_table: Block table shaped ``[num_requests, max_pages]``.
        query_start_loc: Packed prefill query offsets with a terminal offset.
            Offsets may share the base of a larger backing tensor.
        visible_blocks: Number of visible compressed blocks per query.
        token_topk: Number of logical tokens selected per query.
        compress_ratio: Number of logical tokens represented by a cache row.
        max_query_len: Maximum number of query tokens in one request.
        block_indices: Compressed-index output buffer.
        max_seq_len: Longest context length in the batch this step.

    """
    assert token_topk % compress_ratio == 0
    assert block_indices.shape == (q.shape[0], token_topk // compress_ratio)
    assert q.dtype == k_cache.dtype, "Q and the compressed K cache must match"
    rows = q.shape[0]
    # No row scores beyond cdiv(max_seq_len, compress_ratio) compressed
    # columns. Round up to 64 to keep the logits row stride
    # cooperative_topk-compatible.
    logits_width = triton.cdiv(triton.cdiv(max_seq_len, compress_ratio), 64) * 64
    logits_width = min(max(64, logits_width), page_table.shape[1] * k_cache.shape[1])

    # chunk the inputs to keep temp logits below VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
    max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    rows_per_chunk = max(1, max_logits_bytes // (logits_width * 4))
    topk_workspace = _topk_workspace(q.device)

    for query_start in range(0, rows, rows_per_chunk):
        query_end = min(query_start + rows_per_chunk, rows)
        query_slice = slice(query_start, query_end)
        logits = _prefill_logits(
            q,
            k_cache,
            page_table,
            query_start_loc,
            visible_blocks,
            max_query_len,
            logits_width,
            query_offset=query_start,
            num_queries=query_end - query_start,
        )
        _topk(
            logits,
            visible_blocks[query_slice],
            token_topk,
            compress_ratio,
            block_indices[query_slice],
            topk_workspace,
        )


__all__ = [
    "expand_qsa_block_indices",
    "qsa_select_paged_decode",
    "qsa_select_paged_prefill",
    "warmup_qsa_mqa_paged_decode",
    "warmup_qsa_mqa_paged_prefill_turing_candidates",
]
