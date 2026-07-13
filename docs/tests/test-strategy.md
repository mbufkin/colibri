# Test Strategy — Colibri-G10

## Philosophy

**Waterfall with phase-gated validation.** Every phase produces a working, tested artifact before the next phase starts. No building on top of untested code.

## Testing Tiers

### Tier 1: Upstream Validation (Never Break)

These tests must pass unchanged on every commit, regardless of G10 features:

| Test | What It Validates | How To Run |
|------|-------------------|------------|
| JSON parser | `json.h` correctness | `make test-c` → `tests/test_json` |
| Shard reader | `st.h` safetensors I/O | `make test-c` → `tests/test_st` |
| Tokenizer | BPE tokenization | `make test-c` → `tests/test_tier` |
| Grammar | GBNF parsing | `make test-c` → `tests/test_grammar` |
| Batch decode | `decode_batch.h` | `make test-c` → `tests/test_decode_batch` |
| Integer dot | IDOT kernel correctness | `make test-c` → `tests/test_idot` |

### Tier 2: Phase-Specific Validation

Each phase introduces its own test script in `ci/test-phase-N.sh`.

#### Phase 0 — Baseline

```bash
# 0. Build — default config (no G10 flags)
make check
# 1. Build with GPU flags (no G10)
make clean && make CUDA=1
# 2. Teacher-forcing validation (upstream oracle check)
#    (requires glm_tiny test model)
SNAP=./glm_tiny ./glm 16 8 2 ref_glm.json
# Expected: "Matching tokens: 20/20"
```

CI script: `tests/test-phase-0.sh`

#### Phase 1 — VRAM Cache

```bash
# 1. Build with G10 VRAM cache
make clean && make G10=1 CUDA=1
# 2. Upstream tests still pass
make test-c
# 3. Teacher-forcing still matches (VRAM cache must be bit-identical)
SNAP=./glm_tiny ./glm 16 8 2 ref_glm.json
# 4. VRAM cache unit test
#    - Allocate VRAM slots
#    - Store 10 fake experts
#    - Lookup — 100% hit
#    - Store beyond capacity — confirm eviction works
#    - Lookup evicted — confirm miss
g10/test_vram_cache
# 5. Integration: confirm VRAM cache stats are non-zero and sensible
#    (run with environmental flag to print stats)
COLI_G10_DEBUG=1 ./glm ... 2>&1 | grep "vram_cache"
```

CI script: `tests/test-phase-1.sh`

#### Phase 2 — Tensor Core Single-Row

```bash
# 1. Build
make clean && make G10=1 CUDA=1
# 2. All prior tests pass
make test-c
g10/test_vram_cache
SNAP=./glm_tiny ./glm 16 8 2 ref_glm.json
# 3. Tensor core matmul bit-exactness
#    Run same matmul via CPU scalar and GPU tensor core
#    Compare output — must match within 1e-5
g10/test_tensor_core
# 4. Benchmark single-row decode
#    Measure 1000 decode steps — print avg tok/s
COLI_BENCH=1 ./glm ... 2>&1 | grep "decode_tok_s"
```

CI script: `tests/test-phase-2.sh`

#### Phase 3 — Profile / Auto-Tuning

```bash
# 1. All prior tests pass
# 2. Profile generation
#    Generate profile from synthetic corpus
g10/test_profile_gen
# 3. Preload: start with profile, measure hit rate from token 1
#    Must be ≥60%
COLI_PROFILE=./profile.bin ./glm ... 2>&1 | grep "vram_hit_rate"
# 4. Full benchmark suite
./scripts/bench-g10.sh
```

CI script: `tests/test-phase-3.sh`

## Benchmarking Protocol

All benchmarks:

1. Run on the actual G10 hardware (GB10 GPU)
2. 3 runs, report mean ± stddev
3. Warm up: 10 tokens before measurement
4. Measure: next 100 tokens of decode
5. Report: `tok/s`, `VRAM hit rate`, `peak VRAM usage`
6. Publish in `docs/tests/benchmarks/`

### Metrics That Matter

| Metric | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|---------|
| Tokens/sec (decode) | Baseline | ≥2× baseline | ≥3× baseline | ≥4× baseline |
| VRAM hit rate | N/A | ≥80% | ≥85% | ≥90% |
| Peak VRAM usage | ~10 GB | ~110 GB | ~110 GB | ~115 GB |
| Cold start latency (first token) | ~40s | <10s | <8s | <5s |

## Validation Philosophy

- **Token-exact**: The G10 path MUST produce the exact same output tokens as the CPU path for the same input. Our int4 quantization is deterministic (per-row symmetric, the same upstream code) so any token change means a bug.
- **The exception**: The upstream itself documents that batched forward (S>1) and single-token forward diverge on token boundaries due to quantized kernel rounding differences (see README #100). We lock to `IDOT=0` for validation runs to force f32 path and get byte-exact reproducibility.
- **G10 doesn't change this**: Our VRAM cache stores the *same* int4 data as the CPU LRU — we're just changing *where* the multiply happens, not the weights. Token-exact match against CPU is the correctness test.

## When Things Go Wrong

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Tokens diverge from CPU | GPU kernel rounding difference | Enable IDOT=0 for f32 path; check datatype handling |
| VRAM cache miss shows 0 experts | CUDA memory allocation failure | Check `vram_cache_init()` return; increase host buffer |
| Engine crashes on GPU launch | Tensor core tile size mismatch | Check `D % 32 == 0 && I % 32 == 0` guard |
| Profile not loading | File format mismatch | Add `--print` to profile generator to dump first 10 entries |
| Build fails: `nvcc not found` | CUDA toolkit missing | Install: `sudo apt install nvidia-cuda-toolkit` |
