# ADR-001: VRAM Expert Cache as First-Class Memory Tier

**Status:** Accepted (Draft)

**Date:** 2026-07-13

**Author:** Michael Bufkin / Hermes Agent

## Context

The G10 has 122 GB of unified VRAM. The upstream colibri treats all expert storage as "on disk with CPU RAM LRU cache." With 100+ GB free on GPU, we can cache thousands of experts in VRAM for near-instant access.

## Decision

Add a **G10-specific VRAM expert tier** as a new `g10/` module, integrated via a 5-line hook in `glm.c:expert_get()` guarded by `#ifdef COLI_G10_VRAM_CACHE`. This tier sits between the CPU LRU and disk:

```
check VRAM → HIT: GPU matmul
           → MISS: check CPU LRU (existing path)
                  → HIT: promote to VRAM
                  → MISS: load from disk → CPU LRU → (conditional promote to VRAM)
```

## Rationale

- **Minimal core changes**: The upstream engine is well-tested and validated. Inserting a hook is safer than rewriting the memory hierarchy.
- **Clear ownership**: All G10-specific code lives in `g10/`, not spread through `c/glm.c`.
- **Progressive enablement**: Existing tests still pass with `COLI_G10_VRAM_CACHE=0`. We can validate each tier independently.

## Consequences

- The VRAM cache must handle experts of varying sizes (MTP vs regular layers)
- Eviction policy (LFRU) must balance frequency vs recency — wrong policy wastes VRAM
- Heat map persistence across sessions is necessary for startup preload to work well

## Status

Phase 1 — implementation planned. Not yet coded.

---

# ADR-002: Single-Row Tensor Core Decode Path

**Status:** Accepted (Draft)

**Date:** 2026-07-13

## Context

Upstream's CUDA expert group matmul only uses tensor cores (WMMA) when batch size ≥ 8 rows. For decode (S=1), it falls back to scalar per-element matmul. The G10 has tensor cores — we should use them even for single-query decode.

## Decision

Implement an **S=1 fast path** in `g10/g10_matmul.cu` that uses WMMA int4 tensor core matmul for single-row expert operations:

- `matmul_i4_single_gpu()` — gate_proj and up_proj: 1×hidden @ hidden×inter → 1×inter
- Since the result is 1 row × 4-8K columns, the WMMA tile overhead (loading 8×32 fragments) is worth it when the alternative is a CPU-bound scalar loop

## Rationale

- Decode is 100% S=1 — it's the dominant latency mode in chat
- Tensor core WMMA int4 throughput is ~10× scalar CUDA kernel on GB10
- The weight matrices are already in VRAM (from ADR-001)

## Consequences

- The WMMA API on CUDA 13.0 (compute 12.1) may differ from earlier versions — needs verification
- For very narrow matrices (hidden < 256), WMMA overhead may not pay off — add a dimension threshold

---

# ADR-003: Static Expert Preload from Usage Profile

**Status:** Accepted (Draft)

**Date:** 2026-07-13

## Context

The VRAM cache is cold at startup. Use upstream's already-existing usage logging infrastructure to identify which experts to preload before generating the first token.

## Decision

Build `g10/expert_profile.c` that:

1. Reads the existing `stats_dump_q` / `usage_save` data format
2. Processes a representative corpus through the router (offline, one-time) to build a frequency-sorted expert list
3. Writes a compact binary profile: `[layer:eid:score]`
4. At model init, reads profile → uploads top-N experts to VRAM via `vram_cache_preload()`

## Rationale

- Upstream already logs access patterns — no new telemetry needed
- Gives 60-80% VRAM hit rate from the very first token
- Preload happens during model load (~30 seconds), not during inference

## Consequences

- Profile is user-pattern-dependent — a code-usage profile won't help with conversation and vice versa
- Consider rotating profiles or maintaining multiple (chat, code, extraction, etc.)

---

# ADR-004: Waterfall with Phase-Gated Validation

**Status:** Accepted (Draft)

**Date:** 2026-07-13

## Context

This fork uses a waterfall methodology — each phase is implemented, tested, and validated before the next begins. Phase boundaries are gated on published benchmark numbers and test suite passes.

## Decision

**Phase gates:**

| Phase | Gate Criterion |
|-------|----------------|
| 0 — Baseline | Vanilla colibri builds on G10, `make test` passes, teacher-forcing token-exact match validated |
| 1 — VRAM Cache | VRAM cache implemented + validated. Backend tests for cache logic pass. Benchmark >= 2× speedup on cold start |
| 2 — Tensor Core | Single-row decode uses tensor cores. Token-exact match against CPU kernel. Benchmark >= 50% speedup on S=1 |
| 3 — Auto-tuning | Profile generation works. Preload gives ≥60% hit rate from token 1. Full benchmark suite runs automatically |

Each phase:
1. Must pass ALL upstream tests unchanged
2. Must have a new test in `ci/test-phase-N.sh` that exercises the new path
3. Must include an ADR retrospective note on what was learned
4. Must publish a benchmark number from actual G10 hardware

## Consequences

- Slower feature velocity but higher confidence at each phase
- G10 hardware is physically separate — each phase requires a deployment and measurement cycle
- If a phase reveals a design flaw, we roll back to the previous phase boundary (git revert + ADR-errata)
