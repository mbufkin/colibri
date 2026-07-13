/* g10_vram_cache.c — VRAM expert cache implementation.
 *
 * Maintains an LRU cache of expert weights in GPU VRAM.
 * Uses the existing ColiCudaTensor API for upload and management.
 *
 * Eviction policy: LFRU (Least Frequently + Recently Used) matching
 * the upstream tier.h tier_lfru_score strategy, so hot experts stay
 * and dead ones get evicted.
 *
 * Expert size: each expert = 3 matrices × (I×D×b/8 + O×4) bytes
 * For GLM-5.2 int4: gate=6144×2048/2 + 2048×4 = 6.3 MB, up=6.3 MB,
 * down=2048×6144/2 + 6144×4 = 6.3 MB → ~19 MB per expert in VRAM.
 * With 128 GB VRAM, we can cache ~6,000-7,000 experts (out of 19,456).
 */

#include "g10_vram_cache.h"
#include "../c/backend_cuda.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdatomic.h>
#include <pthread.h>

/* ─── configuration ─── */

/* Default: auto-select. g10_vram_cache_init(0, dev) picks a fraction. */
#define G10_VRAM_CACHE_DEFAULT_CAPACITY 0

/* Fraction of VRAM to reserve for the expert cache.
 * Reserve 70 GB for dense compute (attention, embeddings, shared expert),
 * the rest for cached experts. */
#define G10_VRAM_CACHE_VRAM_FRACTION 0.55f    /* use 55% of total VRAM */

/* Per-expert VRAM cost for a worst-case int4 expert (hidden=6144, moe_inter=2048). */
#define G10_EXPERT_BYTES_6144_2048 ((size_t)(3 * (6144 * 2048 / 2 + 2048 * 4)))

/* ─── per-slot structure ─── */

typedef struct {
    int layer;                          /* layer index, -1 = free */
    int eid;                            /* expert id, -1 = free */
    ColiCudaTensor *gate, *up, *down;  /* GPU tensors (NULL = not uploaded) */
    uint32_t heat;                      /* frequency counter */
    uint32_t last;                      /* recency clock value at last access */
    int device;                         /* GPU device index */
} VramSlot;

/* ─── cache state ─── */

static struct {
    VramSlot *slots;
    int capacity;                       /* total slot count */
    int count;                          /* currently occupied */
    int device;                         /* default device */
    uint32_t clock;                     /* global recency counter */
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    int initialized;
    pthread_mutex_t mtx;               /* protects slot array during eviction */
} g_vc = {0};

/* ─── internal helpers ─── */

static inline uint64_t vram_lfru_score(uint32_t heat, uint32_t last, uint32_t clock) {
    uint32_t age = clock - last;
    uint32_t recent = age < 255 ? 255 - age : 0;
    return ((uint64_t)heat << 8) | recent;
}

/* Find the slot with the lowest LFRU score among occupied slots.
 * Returns -1 if no occupied slots. */
static int vram_find_evict_candidate(void) {
    int cold = -1;
    uint64_t cs = UINT64_MAX;
    for (int i = 0; i < g_vc.capacity; i++) {
        if (g_vc.slots[i].eid < 0) continue;  /* free slot */
        uint64_t s = vram_lfru_score(g_vc.slots[i].heat,
                                      g_vc.slots[i].last,
                                      g_vc.clock);
        /* Apply the same 25%+4 hysteresis as upstream tier.h */
        if (cold < 0 || s < cs - (cs >> 2) - (4u << 8)) {
            cold = i;
            cs = s;
        }
    }
    return cold;
}

/* Free one slot's GPU tensors and reset. */
static void vram_free_slot(VramSlot *slot) {
    if (slot->gate) { coli_cuda_tensor_free(slot->gate); slot->gate = NULL; }
    if (slot->up)   { coli_cuda_tensor_free(slot->up);   slot->up   = NULL; }
    if (slot->down) { coli_cuda_tensor_free(slot->down); slot->down = NULL; }
    slot->layer = -1;
    slot->eid   = -1;
    slot->heat  = 0;
    slot->last  = 0;
}

/* ─── public API ─── */

int g10_vram_cache_init(int capacity, int device) {
    if (g_vc.initialized) return 0;

    if (capacity == G10_VRAM_CACHE_DEFAULT_CAPACITY) {
        /* Auto-size based on free VRAM. */
        size_t free_bytes = 0, total_bytes = 0;
        if (!coli_cuda_mem_info(device, &free_bytes, &total_bytes)) {
            fprintf(stderr, "[G10-VC] coli_cuda_mem_info failed\n");
            return -1;
        }
        size_t budget = (size_t)(total_bytes * G10_VRAM_CACHE_VRAM_FRACTION);
        /* Leave headroom: don't use more than 80% of free VRAM */
        size_t free_headroom = (size_t)(free_bytes * 0.8f);
        if (budget > free_headroom) budget = free_headroom;
        capacity = (int)(budget / G10_EXPERT_BYTES_6144_2048);
        if (capacity < 8) capacity = 8;  /* minimum sensible */
        fprintf(stderr, "[G10-VC] auto-capacity: VRAM %.2f GB free %.2f GB → budget %.2f GB → %d slots\n",
                total_bytes / 1e9, free_bytes / 1e9, budget / 1e9, capacity);
    }

    g_vc.slots = calloc((size_t)capacity, sizeof(VramSlot));
    if (!g_vc.slots) return -1;

    for (int i = 0; i < capacity; i++) {
        g_vc.slots[i].layer = -1;
        g_vc.slots[i].eid   = -1;
    }

    g_vc.capacity = capacity;
    g_vc.device   = device;
    g_vc.clock    = 0;
    g_vc.count    = 0;
    g_vc.hits     = 0;
    g_vc.misses   = 0;
    g_vc.evictions = 0;
    g_vc.initialized = 1;
    pthread_mutex_init(&g_vc.mtx, NULL);

    fprintf(stderr, "[G10-VC] initialized: %d slots on device %d\n", capacity, device);
    return 0;
}

void g10_vram_cache_shutdown(void) {
    if (!g_vc.initialized) return;
    for (int i = 0; i < g_vc.capacity; i++) {
        vram_free_slot(&g_vc.slots[i]);
    }
    free(g_vc.slots);
    g_vc.slots = NULL;
    g_vc.capacity = 0;
    g_vc.count = 0;
    g_vc.initialized = 0;
    pthread_mutex_destroy(&g_vc.mtx);
}

G10VramExpert *g10_vram_cache_lookup(int layer, int eid) {
    if (!g_vc.initialized) return NULL;

    for (int i = 0; i < g_vc.capacity; i++) {
        if (g_vc.slots[i].layer == layer && g_vc.slots[i].eid == eid) {
            g_vc.hits++;
            g_vc.slots[i].last = ++g_vc.clock;
            if (g_vc.slots[i].heat < UINT32_MAX) g_vc.slots[i].heat++;
            return (G10VramExpert *)&g_vc.slots[i];
        }
    }
    g_vc.misses++;
    return NULL;
}

G10VramExpert *g10_vram_cache_insert(int layer, int eid,
                                      const void *gate_w, const void *up_w, const void *down_w,
                                      const float *gate_s, const float *up_s, const float *down_s,
                                      int fmt, int I, int D, int device)
{
    if (!g_vc.initialized) return NULL;

    pthread_mutex_lock(&g_vc.mtx);

    /* Find a free slot, or evict one. */
    int idx = -1;
    for (int i = 0; i < g_vc.capacity; i++) {
        if (g_vc.slots[i].eid < 0) { idx = i; break; }
    }
    if (idx < 0) {
        /* Cache full — evict coldest. */
        idx = vram_find_evict_candidate();
        if (idx < 0) { pthread_mutex_unlock(&g_vc.mtx); return NULL; }
        VramSlot *old = &g_vc.slots[idx];
        fprintf(stderr, "[G10-VC] evict L%d/E%d (heat=%u) → make room for L%d/E%d\n",
                old->layer, old->eid, old->heat, layer, eid);
        vram_free_slot(old);
        g_vc.evictions++;
    }

    VramSlot *slot = &g_vc.slots[idx];

    /* Upload the three expert matrices to GPU.
     * Non-contiguous in slab — upload each separately. */
    if (!coli_cuda_tensor_upload(&slot->gate, gate_w, gate_s, fmt, I, D, device)) {
        fprintf(stderr, "[G10-VC] upload gate failed for L%d/E%d\n", layer, eid);
        slot->gate = NULL;
        pthread_mutex_unlock(&g_vc.mtx);
        return NULL;
    }
    if (!coli_cuda_tensor_upload(&slot->up, up_w, up_s, fmt, I, D, device)) {
        fprintf(stderr, "[G10-VC] upload up failed for L%d/E%d\n", layer, eid);
        vram_free_slot(slot);
        pthread_mutex_unlock(&g_vc.mtx);
        return NULL;
    }
    if (!coli_cuda_tensor_upload(&slot->down, down_w, down_s, fmt, D, I, device)) {
        fprintf(stderr, "[G10-VC] upload down failed for L%d/E%d\n", layer, eid);
        vram_free_slot(slot);
        pthread_mutex_unlock(&g_vc.mtx);
        return NULL;
    }

    slot->layer = layer;
    slot->eid   = eid;
    slot->heat  = 1;
    slot->last  = ++g_vc.clock;
    slot->device = device;
    g_vc.count++;

    pthread_mutex_unlock(&g_vc.mtx);
    return (G10VramExpert *)slot;
}

void g10_vram_cache_touch(int layer, int eid) {
    if (!g_vc.initialized) return;
    for (int i = 0; i < g_vc.capacity; i++) {
        if (g_vc.slots[i].layer == layer && g_vc.slots[i].eid == eid) {
            g_vc.slots[i].last = ++g_vc.clock;
            if (g_vc.slots[i].heat < UINT32_MAX) g_vc.slots[i].heat++;
            return;
        }
    }
}

void g10_vram_cache_get_tensors(G10VramExpert *entry,
                                 void **gate, void **up, void **down)
{
    VramSlot *slot = (VramSlot *)entry;
    if (gate) *gate = slot->gate;
    if (up)   *up   = slot->up;
    if (down) *down = slot->down;
}

int g10_vram_cache_count(void) { return g_vc.count; }
int g10_vram_cache_capacity(void) { return g_vc.capacity; }
uint64_t g10_vram_cache_hits(void) { return g_vc.hits; }
uint64_t g10_vram_cache_misses(void) { return g_vc.misses; }
uint64_t g10_vram_cache_evictions(void) { return g_vc.evictions; }

size_t g10_vram_cache_bytes(void) {
    size_t total = 0;
    if (!g_vc.initialized) return 0;
    for (int i = 0; i < g_vc.capacity; i++) {
        if (g_vc.slots[i].eid < 0) continue;
        total += coli_cuda_tensor_bytes(g_vc.slots[i].gate) * 3;  /* approx: gate+up+down */
    }
    return total;
}
