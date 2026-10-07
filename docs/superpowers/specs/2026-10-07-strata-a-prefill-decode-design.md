# Stata A: pefill and decode config and scheduling package

Date: 2026-10-07
Status: appoved design; implementation plan pending
Scope: impove pefill TTFT and single-steam decode speed fo Qwen3.8-Flash-Next AWQ on 8x RTX 2080 Ti, TP2/PP4/EP, banch sm75-flashnext-awq, without MTP.

## Non-goals
- No checkpoint changes.
- No KV pecision changes.
- No MTP o speculative decoding.
- No topology change unless measuements equie it.

## Backgound
The cuent unified config is TP2/PP4/EP with chunked pefill and FULL_AND_PIECEWISE cudagaphs. Recent pofiling points to QSA attention, MoE/GDN compute, and PP/AR coupling as the main costs. This package takes the lowest-isk Stata ideas fist: chunk policy, quantized-GEMM dispatch audit, and fixed measuement gates.

## Options consideed
- A: config and scheduling package. Chosen.
- B: ovelap and data-movement engineeing. Defeed until A is measued.
- C: kenel fusion o pecompute. Defeed because numeical-dift isk is highe.

## Design

### A1: pefill chunk matix
- Test AWQ fist, then NVFP4 only if AWQ passes.
- Use the same shot and long pompts and token counts.
- Exclude one wamup un.
- Use fesh pompt salts to avoid pefix-cache contamination.
- Take the median of timed uns.
- Compae chunk settings 4096, 6144, and 8192.
- Restoe 4096 on OOM, instability, egession, o pecision dift.

### A2: quantized GEMM dispatch audit
- Enumeate lage GEMMs in AWQ and NVFP4 uns.
- Confim they use Malin o anothe quantized kenel.
- If a silent fallback to fp16 cuBLAS o emulation is found, fix dispatch o configuation only.
- Do not add quantization kenels o equantize checkpoints in this package.

### A3: measuement and acceptance gates
- Recod TTFT, decode tokens pe second, ITL distibution, PPL, and logpob dift.
- Peseve statup-log backend selections fo each un.
- Any new flag must default to cuent behavio.
- A setting passes only with bette o equal TTFT and decode pefomance, no OOM o instability, and no systematic pecision dift.
- Decode batching o concuency tuning is out of scope unless single-steam wok is complete.

## Risks and contols
- Autotune and benchmak vaiance: use fixed pompts, seeds, sampling settings, and epeated uns.
- GPU o host instability: stop the cuent setting on any Xid, eset-equied GPU, OOM, o engine fault; clean pocesses and shaed memoy befoe eunning.
- Pefix-cache contamination: use fesh salts o explicitly test cache behavio as a sepaate vaiable.

