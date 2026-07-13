/* g10_vram_cache.h — VRAM expert cache for NVIDIA GB10 GPU.
 * Holds hot expert weights permanently in VRAM, avoiding reload from CPU RAM.
 * Integrated via a single check in glm.c's expert resolution loop.
 *
 * Design: fixed-size LRU cache of expert slots per device.
 * Each slot holds the three ColiCudaTensors (gate, up, down) for one expert.
 *
 * Thread-safety: the inference path is single-threaded for expert
 * resolution per block, so we use relaxed atomics for the clock. */

#ifndef G10_VRAM_CACHE_H
#define G10_VRAM_CACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to one cached expert's GPU tensors. */
typedef struct G10VramExpert G10VramExpert;

/* Initialize cache with up to `capacity` expert slots per device.
 * capacity=0 auto-selects based on VRAM size and expert footprint.
 * Returns 0 on success, -1 on failure. */
int g10_vram_cache_init(int capacity, int device);

/* Shut down and free all GPU tensors. */
void g10_vram_cache_shutdown(void);

/* Find expert `eid` in the VRAM cache.
 * Returns handle if resident, NULL on miss. */
G10VramExpert *g10_vram_cache_lookup(int layer, int eid);

/* Insert a newly-loaded expert (from CPU RAM) into the VRAM cache.
 * The weights are in `slab` with `scales` in `fslab`, format `fmt`.
 * If the cache is full, evicts the coldest entry.
 * Returns handle on success, NULL if no eviction candidate or OOM.
 * Caller MUST NOT free `slab`/`fslab` until the handle is evicted. */
G10VramExpert *g10_vram_cache_insert(int layer, int eid,
                                      const void *gate_w, const void *up_w, const void *down_w,
                                      const float *gate_s, const float *up_s, const float *down_s,
                                      int fmt, int I, int D, int device);

/* Mark `eid` as recently used (call on every hit). */
void g10_vram_cache_touch(int layer, int eid);

/* Get the ColiCudaTensor handles from a cache entry. */
void g10_vram_cache_get_tensors(G10VramExpert *entry,
                                 void **gate, void **up, void **down);

/* Statistics. */
int g10_vram_cache_count(void);              /* currently resident */
int g10_vram_cache_capacity(void);           /* max slots */
uint64_t g10_vram_cache_hits(void);
uint64_t g10_vram_cache_misses(void);
uint64_t g10_vram_cache_evictions(void);
size_t g10_vram_cache_bytes(void);           /* total VRAM used */

#ifdef __cplusplus
}
#endif

#endif /* G10_VRAM_CACHE_H */
