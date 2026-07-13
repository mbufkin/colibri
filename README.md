# Colibri-G10 🐦 → 🚀

**Fork of [JustVugg/colibri](https://github.com/JustVugg/colibri) targeting the NVIDIA GB10 GPU (122 GB VRAM).**

Upstream colibri is a pure-C MoE streaming engine that runs 744B-parameter GLM-5.2 on consumer laptops by streaming experts from disk. This fork adds a **VRAM expert cache tier** to take advantage of the G10's 122 GB of unified GPU memory — targeting **2–5+ tokens/second** on the same model.

## What's Different

| | Upstream | Colibri-G10 |
|---|---|---|
| **Target** | Consumer laptop, 16-32 GB RAM | NVIDIA GB10, 122 GB VRAM |
| **Expert storage** | SSD + OS page cache + small CPU LRU | SSD + CPU LRU + **VRAM LRU (~5,500 experts)** |
| **Matmul** | CPU (NEON/AVX2/VSX) or optional CUDA | **GPU tensor core (WMMA int4) for all paths** |
| **Loading speed** | 0.1-1 tok/s | **Target: 2-5+ tok/s** |
| **Design** | Zero-dependency C, minimal changes | Same base + targeted `g10/` module additions |

## Project Structure

```
colibri-g10/
├── c/                    # Upstream C engine (preserved, minimal changes)
├── g10/                  # G10-specific additions (VRAM cache, tensor core opt, profiling)
├── docs/
│   ├── architecture/     # Architecture overview + G10 implementation plan
│   ├── decisions/        # Architecture Decision Records (ADRs)
│   └── tests/            # Test strategy and benchmarks
├── ci/                   # Phase-gated CI scripts
├── scripts/              # Build and benchmark tooling
└── docs/architecture/    # This fork's documentation
```

## Development Phases (Waterfall + Phase-Gated)

| Phase | Goal | Expected Speed | Status |
|-------|------|----------------|--------|
| **0** | Vanilla colibri build + test on G10 | 0.05–0.1 tok/s | ❌ Not started |
| **1** | VRAM expert LRU cache | 2–3 tok/s | ❌ Planned |
| **2** | Tensor core single-row decode | 3–5 tok/s | ❌ Planned |
| **3** | Expert profiling + auto-tuning preload | 5+ tok/s (evolving) | ❌ Planned |

Each phase is validated independently; tests must pass before the next phase starts.

## Quick Start (Phase 0)

```bash
# Build vanilla colibri on G10
cd c
make -j20

# Smoke test
SNAP=./glm_tiny ./glm 16 8 2 ref_glm.json

# Once model is on disk:
COLI_MODEL=/path/to/model ./coli chat
```

## Branch Strategy

- `main` — stable, tested, at the latest completed phase
- `phase-0-baseline` — vanilla colibri running on G10
- `phase-1-vram-cache` — VRAM cache work
- `phase-2-tensor-core` — tensor core decode optimization
- `phase-3-auto-tuning` — profiling and auto-tuning

## License

Same as upstream: [LICENSE](LICENSE) (Apache 2.0 or MIT, TBD — verify upstream).
