# Colibri-G10 Architecture

## What This Fork Is

A disciplined fork of [JustVugg/colibri](https://github.com/JustVugg/colibri) targeting the **NVIDIA GB10 GPU (122 GB VRAM)** — the DGX Spark / Lenovo ThinkStation PGX.

The upstream colibri is tuned for consumer laptops with 16-32 GB RAM and NVMe storage. The G10 has 122 GB of unified GPU memory, changing the optimization target from "stream from disk" to "keep as much model on GPU as possible."

## Target Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GB10 (CUDA 13.0, compute 12.1) |
| VRAM | 122 GB (~112 GB free after OS) |
| System RAM | 119 GB |
| NVMe | 916 GB (233 GB free) |
| CPU | Grace ARM CPU, 20 cores |
| Per-token compute | ~40B active params (GLM-5.2 MoE, 8/256 experts per token) |

## Upstream Architecture (for Reference)

```
                    ┌─────────────────────┐
                    │   Model on Disk      │
                    │  (370 GB, int4)      │
                    └──────┬──────────────┘
                           │ pread + fadvise(DONTNEED)
                           ▼
               ┌───────────────────────┐
               │  OS Page Cache (L2)   │  (passive — kernel-managed)
               └───────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  CPU RAM    │
                    │  ┌────────┐ │
                    │  │ Dense  │ │  ~10 GB (attention, embed, norms)
                    │  │ Core   │ │
                    │  └────────┘ │
                    │  ┌────────┐ │
                    │  │ Expert  │ │  LRU cache, ~16 slots/layer = ~1,200 experts
                    │  │ LRU     │ │  ~22 GB at int4
                    │  │ Cache   │ │
                    │  └────────┘ │
                    │  ┌────────┐ │
                    │  │ Pinned │ │  Hot-store: user's favorite experts
                    │  │ Hot    │ │  Learned from usage logs
                    │  │ Store  │ │
                    │  └────────┘ │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  CPU Matmul │  OpenMP-threaded, NEON/AVX2 kernels
                    │  (no BLAS)  │
                    └─────────────┘
```

## G10 Architecture (Target)

```
                    ┌─────────────────────┐
                    │   Model on Disk      │
                    │  (370 GB, int4)      │
                    └──────┬──────────────┘
                           │ pread + fadvise(DONTNEED)
                           ▼
               ┌───────────────────────┐
               │  OS Page Cache (L2)   │
               └───────────────────────┘
                           │
                    ┌──────▼──────────────┐
                    │  CPU RAM            │
                    │  ┌───────────────┐  │
                    │  │ Dense Core    │  │  ~10 GB (unchanged from upstream)
                    │  │ (attention,   │  │
                    │  │ embed, norms) │  │
                    │  └───────────────┘  │
                    │  ┌───────────────┐  │
                    │  │ Fallback LRU  │  │  Small CPU cache for infrequent
                    │  │ (reduced)     │  │  experts that spill from VRAM
                    │  └───────────────┘  │
                    └──────┬──────────────┘
                           │
              ┌────────────▼─────────────┐
              │  CUDA  GPU  (GB10 VRAM)  │  122 GB total
              │                          │
              │  ┌────────────────────┐  │
              │  │ VRAM Dense Core    │  │  ~10 GB (mirror for GPU compute)
              │  │ (CUDA resident)    │  │
              │  └────────────────────┘  │
              │  ┌────────────────────┐  │
              │  │ VRAM Expert Tier   │  │  ~100 GB (~5,000+ experts in int4)
              │  │ (managed LRU on    │  │  Top 20-25% of expert pool
              │  │  GPU, pinned +     │  │  + dynamic promotion
              │  │  preloaded from    │  │
              │  │  usage data)       │  │
              │  └────────────────────┘  │
              │  ┌────────────────────┐  │
              │  │ Tensor Core        │  │  WMMA int4 matmul for grouped experts
              │  │ Matmul Engine      │  │  (CUDA tensor cores, SM 12.1+)
              │  └────────────────────┘  │
              └──────────────────────────┘
```

## Memory Budget (G10, 122 GB VRAM)

| Component | Size | How Managed |
|-----------|------|-------------|
| GPU driver + OS | ~10 GB | System reserved |
| Dense core (int4) | ~10 GB | Always resident |
| Expert VRAM cache | ~100 GB | Managed LRU, 5,000+ experts |
| KV-cache (32K tokens) | ~1 GB | Grows with context |
| Scratch buffers | ~1 GB | Temporary |
| **Total** | **~122 GB** | |

## Key Design Decisions (ADRs in `docs/decisions/`)

| # | Decision | Status |
|---|----------|--------|
| 001 | VRAM as first-class expert cache tier, not just pinned tensor store | Draft |
| 002 | Single-query decode uses tensor cores regardless of batch size | Draft |
| 003 | Expert heat-map profiling on G10 at startup from static routing data | Draft |
| 004 | Waterfall test-as-we-implement: each tier validated before next | Draft |

## Development Workflow

```
┌────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  PHASE 1   │───▶│   PHASE 2    │───▶│   PHASE 3    │───▶│  PHASE 4 │
│  Baseline  │    │  VRAM Cache  │    │  Tensor Core │    │  Profiling│
│  (vanilla  │    │  Tier        │    │  Optimize    │    │  Auto-    │
│  colibri   │    │              │    │              │    │  Tuning   │
│  on G10)   │    │              │    │              │    │           │
└────────────┘    └──────────────┘    └──────────────┘    └──────────┘
     │                  │                   │                   │
     ▼                  ▼                   ▼                   ▼
  Measure            Measure             Measure             Self-hosted
  0.1-1 tok/s        2-3 tok/s           3-5 tok/s           5+ tok/s
  (cold SSD)         (VRAM cache)        (tensor core        (evolutionary
                                           decode)            optimization)
```

Each phase ends with:
1. All tests passing (validation against upstream oracle)
2. Published benchmark number from actual G10 hardware
3. ADR capture of what was learned

## Project Layout

```
colibri-g10/
├── c/                          # Upstream C engine (untouched core)
│   ├── glm.c                   # Main GLM-5.2 inference engine
│   ├── backend_cuda.cu         # CUDA backend (WE MODIFY THIS)
│   ├── backend_cuda.h          # CUDA header
│   ├── backend_metal.mm        # Apple GPU (untouched)
│   ├── st.h, tok.h, ...        # Upstream headers
│   └── Makefile                # Build system
├── g10/                        # G10-specific additions
│   ├── vram_cache.c            # VRAM expert LRU cache implementation
│   ├── vram_cache.h            # VRAM cache public API
│   ├── expert_profile.c        # Expert usage profiler / preloader
│   ├── expert_profile.h
│   ├── g10_matmul.cu           # G10-optimized tensor core matmul kernels
│   ├── g10_matmul.h
│   └── BUILD.md                # How to build for G10 specifically
├── docs/
│   ├── architecture/
│   │   └── README.md           # This file
│   ├── decisions/              # Architecture Decision Records
│   │   ├── ADR-001-vram-expert-cache.md
│   │   ├── ADR-002-single-row-tensor-core.md
│   │   └── ...
│   └── tests/
│       ├── test-strategy.md    # How we validate each phase
│       └── benchmarks/         # Benchmark results as we go
├── ci/
│   ├── test-phase-1.sh         # Baseline validation script
│   ├── test-phase-2.sh         # VRAM cache validation
│   └── ...
├── scripts/                    # Developer tooling
│   └── build-g10.sh            # One-command G10 build
├── tests/                      # Upstream tests (preserved)
│   ├── test_json.c
│   ├── test_st.c
│   └── ...
└── README.md                   # Fork-specific readme
```
