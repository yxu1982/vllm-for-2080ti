# vLLM SM75 Fork — Qwen3.8-Flash-Next on 8× RTX 2080 Ti 22G

本分支让 Qwen3.8-Flash-Next（AWQ / NVFP4 / FP8）在 **8× RTX 2080 Ti 22G（Turing sm75，无原生 FP8/BF16，无 NVLink）**
上跑起来，并针对 prefill/decode 做了 SM75 专项优化与稳定性修复。

- 基线：上游 `main @7f1a5398e`（2026-09-24，约 `v0.30.1rc0+60`），分支 `sm75-flashnext-awq`。
- 目标拓扑：**TP2×PP4 + EP**（同 PCIe 岛内 TP，跨岛 PP）。
- License：Apache-2.0（同上游）。

## 主要改动

| 改动 | 文件 / 开关 | 说明 |
|---|---|---|
| QSA prefill `BLOCK_N=16` | `vllm/models/qwen4_exp/nvidia/ops/qsa.py`，`VLLM_QWEN4_QSA_PREFILL_BN` | 32 KiB smem、每 SM 双 block；实测 25K prefill −9%、110K −12%，PPL 中性 |
| PLE mmap overlay | `vllm/vllm_ple_mmap.py`，`VLLM_PLE_MMAP_MODE=mem` + `VLLM_PLE_MEM_LAZY=1` + `VLLM_PLE_HOST_GATHER=1` | 95.4 GiB ngram 表驻 `/dev/shm`，后台填充、未填片回磁盘；gather 移出 forward，decode 可用 FULL 图 |
| QSA indexer / Turing autotune | `VLLM_QWEN4_TURING_AUTOTUNE=1`，`VLLM_QWEN4_INDEXER_CUBLAS` | cuBLAS 大 prefill 打分（密集形状约 2.6x；已做数值对照：与 Triton 最大差 4.6e-05，top-k 零翻转，默认 `0` 最稳） |
| Turing GEMV（实验性） | `vllm/turing_gemv.py` + `linear.py`，`VLLM_TURING_GEMV=1` | M=1 fp16 dense 的 Triton streaming GEMV；**未充分验证，稳定性测试建议先关** |
| **AR 256 KiB 安全门（稳定性关键）** | `custom_all_reduce.py`（`VLLM_SM75_CUSTOM_AR_MAX_BYTES`，默认 262144）+ `flashinfer_pcie_ipc_all_reduce.py`（`VLLM_SM75_FI_PCIE_IPC_MAX_BYTES`） | custom AR / FlashInfer PCIe-IPC 单发超 256 KiB 会挂 GSP（Xid 119）；超限自动回退 NCCL，小包（decode）继续走高速路径 |
| FlashInfer PCIe-IPC 回填 | 需配合 venv 中 flashinfer 0.7.0 回填；`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` | TP4 上 decode +17%、prefill 中性（TP2 走 custom AR/NCCL，此项无收益） |

## 快速启动（AWQ，TP2PP4）

```bash
export PATH="/home/yxu/llm/venvs/vllm/bin:/usr/local/cuda/bin:$PATH"
export VLLM_PLE_MMAP_MODE=mem VLLM_PLE_MEM_LAZY=1
export VLLM_DEFER_ENGRAM_LOAD=0 VLLM_DROP_CHECKPOINT_CACHE=0
export VLLM_PLE_HOST_GATHER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_QWEN4_TURING_AUTOTUNE=1 TRITON_PRINT_AUTOTUNING=0
export VLLM_QWEN4_QSA_PREFILL_BN=16
export VLLM_QWEN4_INDEXER_CUBLAS=0        # 1 更快，0 最稳
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64
# export VLLM_TURING_GEMV=1               # 实验性，稳定性测试先别开
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# export VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1  # TP4 才有收益
rm -f /dev/shm/ple_mmap_* /dev/shm/psm_*  # 起服前清残留（host 125G，单表 95G）

vllm serve <model> \
  --tensor-parallel-size 2 --pipeline-parallel-size 4 --enable-expert-parallel \
  --dtype float16 --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 4096 \
  --gdn-prefill-backend flashqla_sm75 \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --trust-remote-code --host 0.0.0.0 --port 8000
```

要点：
- 起服前确认 8 卡显存为 0（残留 `VLLM::Worker` 会占卡，用 `pkill -9 -f 'VLLM::Worker'` 清）。
- FP8 模型另需 `--max-num-seqs 64`（默认 256 会超 Mamba cache）与 `--language-model-only`。
- 6 并发以上建议保持 `--disable-custom-all-reduce`（NCCL）直到 AR 路径 soak 通过；本分支默认已把 custom AR 钳在 256 KiB，日志出现
  `SM75: capping custom allreduce max_size 8388608 -> 262144 bytes` 即生效。

## 稳定性备忘（2080Ti 魔改卡 + open 内核）

- 本机必须用 **NVIDIA open 内核模块**（GeForce 开 PCIe P2P 需要它），代价是 GSP 关不掉。
- custom AR / FlashInfer PCIe-IPC / PLX two-stage 的**大包（>256 KiB）P2P 猛冲会挂 GSP（Xid 119，游走）**；NCCL 的 P2P 正常。本分支用 256 KiB 门限隔离。
- 另见过 **Xid 32（MMU 非法访存）→ UVM fatal → 整机需 reboot**；与 custom AR 开启强相关，生产环境先用 NCCL。
- 平台不支持 PCIe AER；`watch nvidia-smi` 这类高频全量 NVML 轮询会加剧 GSP 压力，服务时别开（要用就用 `--loop-ms` 单进程低频）。
- 功耗墙已压到 188W；温度正常（40~50°C）。

## 性能基线（AWQ，TP2PP4，8×2080Ti 22G）

- 25K prefill ≈ 4.7s，110K ≈ 18s；decode ≈ 51 tok/s；PPL ≈ 10.74。
- QSA BN16：prefill −9%（25K）/ −12%（110K），精度中性。
- 精度：自建栈 + Flash-Next AWQ 与 ModelScope 官方 API 跑分基本持平（549 vs 549，6 项工具基准：ToolCall/InstructFollow/PromptAuthority/ReasonMath/BugFind/StructOutput）。
