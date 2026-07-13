# G10 VRAM Expert Cache — Implementation Plan

## Overview

Add a **VRAM-resident expert cache** as a first-class tier between the existing CPU LRU cache and disk. This is the single highest-impact change for G10 hardware (122 GB VRAM).

## Current Upstream Code Path

```
Model.expert_get(layer, eid)
  → scan CPU LRU cache slots for eid
    → HIT: return slot, update clock
    → MISS:
      → evict LRU slot from CPU cache if full
      → pread 3 weight matrices from disk (gate/up/down, ~57 MB)
      → quantize (f32→int4 on load)
      → store in CPU LRU slot
      → return slot
```

Each miss is a **~57 MB disk read** + quantize. At cold start (most experts miss), that's 1,800 disk reads per token = ~30-60 sec/token on NVMe.

## Target Code Path

```
Model.expert_get(layer, eid)
  → scan VRAM cache for eid (GPU-side lookup)
    → HIT: return GPU tensor handles, dispatch CUDA matmul
    → MISS:
      → scan CPU LRU cache for eid
        → HIT: copy slot to VRAM, evict LRU from VRAM if full
        → MISS:
          → pread from disk (existing upstream path)
          → quantize → store in CPU LRU
          → promote to VRAM if heat/access count warrants it
```

## Implementation Files

All new code goes in `g10/` directory — zero modifications to `c/glm.c` core engine.

### `g10/vram_cache.h` — Public API

```c
// Initialize VRAM expert cache. Called once at model load.
// max_experts: how many expert slots to allocate in VRAM
//   (computed from free VRAM: ~5,000+ = ~100 GB)
int vram_cache_init(int max_experts, int device);

// Check and return expert from VRAM. Returns non-zero on hit.
// On hit: gate/up/down are CUDA tensor handles (ColiCudaTensor*)
//   ready for immediate matmul launch on the GPU.
int vram_cache_lookup(int layer, int eid,
                      ColiCudaTensor **gate,
                      ColiCudaTensor **up,
                      ColiCudaTensor **down);

// Store an expert in VRAM. Copied from CPU-side quantized weights.
// Uses cudaMemcpyAsync with pinned host memory.
int vram_cache_store(int layer, int eid,
                     const int8_t *gate_w, const float *gate_s,
                     const int8_t *up_w,   const float *up_s,
                     const int8_t *down_w, const float *down_s,
                     int O, int I);

// Heat/promotion hint: called by expert_get on every access
// to update access frequency counters for promotion decisions.
void vram_cache_touch(int layer, int eid);

// Preload top-N experts from static profile
int vram_cache_preload(const char *profile_path);

// Stats
void vram_cache_stats(uint64_t *hits, uint64_t *misses, int *n_resident);
```

### `g10/vram_cache.c` — Implementation

Key algorithmic decisions (see ADR-001 for detailed reasoning):

- **Replacement policy**: LFRU (Least Frequently Recently Used) — hybrid of frequency and recency, same as upstream's usage heatmap but with longer temporal window
- **Eviction granularity**: per-expert (not per-layer); a hot expert in layer 5 is worth more than a cold one in layer 0 regardless of position
- **Copy strategy**: async `cudaMemcpyAsync` via pinned host staging buffer; upstream's CPU LRU acts as the staging source
- **Metadata storage**: CPU-side hash table mapping `(layer << 16) | eid` → VRAM slot index; GPU-side lookup via texture objects (or surface writes) for direct GPU-internal cache-aware routing
- **Capacity**: auto-calculated at init from `coli_cuda_mem_info()` — target 80-85% of free VRAM

### Integration into `glm.c`

The change to the core engine is minimal — about 5 lines:

```c
// In expert_get() in glm.c, AFTER the CPU LRU cache miss path:
// (currently starts with "if (lc->n < lc->cap) { ... }")
// INSERT:
#ifdef COLI_G10_VRAM_CACHE
    // Try VRAM cache first before falling to disk
    Slot *vram_slot = NULL;
    if (vram_cache_lookup(m, layer, eid, &vram_slot)) {
        // VRAM hit: return GPU-tensor-backed slot
        *out = vram_slot;
        m->vram_hits++;
        return;
    }
#endif
```

The upstream already uses `#ifdef COLI_CUDA` guards — we add `COLI_G10_VRAM_CACHE` as an additional guard that implies `COLI_CUDA`.

### `g10/expert_profile.c` — Static Expert Profiling

- Reads the existing usage log format that upstream already writes (`stats_dump_q` / `usage_save`)
- Processes ~10K representative routing decisions (can extract from the model itself via router forward on a synthetic corpus)
- Produces a **static preload list**: `[layer, eid, priority_score]` sorted by frequency
- At model load, the top-N from this list are preloaded into VRAM before any tokens are generated
- This gives the system a "warm" cache from first token

## Data Structures

### VRAM Slot (GPU-side)

```c
struct VramExpertSlot {
    ColiCudaTensor *gate;   // Already uploaded to GPU
    ColiCudaTensor *up;
    ColiCudaTensor *down;
};
```

### VRAM Slot Metadata (CPU-side, for eviction decisions)

```c
struct VramSlotMeta {
    int layer, eid;
    uint64_t last_access;    // CLOCK_MONOTONIC tick
    uint32_t access_count;   // Total accesses since load
    uint8_t padding[4];      // Explicit pad for alignment
};
```

Total: 24 bytes per slot × 5,000 slots = **120 KB** — negligible.

## Memory Budget Calculation

```
VRAM total:         122 GB
GPU driver/OS:       ~10 GB
Dense core (GPU):    ~10 GB
KV-cache (32K):      ~1 GB
Scratch buffers:     ~1 GB
────────────────────────
Available:          ~100 GB

Per expert (int4):  19 MB (gate=inter*hidden, up=inter*hidden, down=hidden*inter)
Experts in VRAM:    100 GB / 19 MB ≈ 5,500 slots
Expert coverage:    5,500 / 21,504 ≈ 25.6% of all experts
```

## Expected Performance Model

Based on upstream benchmarks and G10 specs:

| Scenario | Tok/s | Bottleneck |
|----------|-------|------------|
| Vanilla colibri (all SSD) | 0.05–0.1 | Disk I/O |
| VLAN tier (90% VRAM hit) | ~3–5 | GPU matmul + remaining 10% SSD |
| Full expert resident* | ~6–8 | GPU matmul only |
| Full resident + tensor core decode | ~10–15 | Single-token forward bound |

*Full residency requires ~400 GB VRAM — not possible on GB10. The 5,500-slot cache covers only ~26% of experts, but MoE routing is heavily power-law distributed: the top 26% handle ~80-90% of traffic.

## Phase 1 Implementation Steps

1. Create `g10/vram_cache.h` and `g10/vram_cache.c` with LRU-backed VRAM slots
2. Add `COLI_G10_VRAM_CACHE` flag to Makefile (conditional on CUDA=1 + G10=1)
3. Add 5-line hook in `glm.c:expert_get()` guarded by `#ifdef COLI_G10_VRAM_CACHE`
4. Build on G10: `make G10=1 CUDA=1`
5. **Validate**: run upstream reference test (`make test` + teacher-forcing on glm_tiny)
6. **Benchmark**: measure tok/s on G10 with model small enough to fit on NVMe (e.g., a downsized MoE)

## Phase 2 Implementation Steps

1. Add expert usage tracking in VRAM cache (heat counts, not just clock)
2. Implement `expert_profile.c` — build preload list from usage data
3. Implement `vram_cache_preload()` — load preload list at model init
4. Implement `vram_cache_touch()` — promote frequent missers on-the-fly
5. **Validate**: hit rate ≥60% from token 1, ≥85% after 100 tokens
6. **Benchmark**: cold-start tok/s vs after-warmup tok/s

## Phase 3 Implementation Steps

1. Extend tensor core matmul for single-row decode (`g10/g10_matmul.cu`)
2. **Validate**: token-exact match against upstream CPU kernel
3. **Benchmark**: tok/s improvement at S=1 vs current CUDA fallback path

## Integration with the G10

To build and test:

```bash
# On the G10:
cd ~/colibri-g10
make G10=1 CUDA=1 -j20

# Smoke test with tiny model (teacher-forcing validation):
SNAP=./glm_tiny ./glm 16 8 2 ref_glm.json

# Chat (once model is on disk):
COLI_MODEL=/nvme/glm52_i4 ./coli chat
```

The model itself needs to be on the NVMe. With only 233 GB free and the int4 model at 370 GB, we'll need to:
- Either free space on the G10's NVMe (clean up llama.cpp models, temp files, etc.)
- Or download to an external drive and symlink
- Run the `./coli convert` pipeline overnight — it's resumable

---

## Pitfalls

1. **CUDA API version**: G10 needs CUDA 13.0. The current `backend_cuda.cu` needs to be checked for compatibility with WMMA API changes in 13.x (NVIDIA changed tensor core APIs significantly in 12.x→13.x)
2. **Expert size varies**: MTP head experts have different dimensions than regular layer experts. The `slab_cap` / `fslab_cap` fields in `ESlot` exist to handle this — the VRAM cache must be dimension-aware
3. **No nvcc on G10**: We need CUDA toolkit installed (`sudo apt install nvidia-cuda-toolkit` or the DGX SDK)
4. **Power-law distribution is user-dependent**: The "top 26% expert coverage = 80-90% traffic" is a rough baseline. The actual hit rate depends on use pattern (chat vs batch vs code generation)
