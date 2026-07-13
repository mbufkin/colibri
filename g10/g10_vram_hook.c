/* g10_vram_hook.c — VRAM cache promotion for experts loaded from disk.
 *
 * This file is #included into glm.c when COLI_G10_VRAM_CACHE is defined.
 * It provides the promotion logic that:
 *   1. Checks if an expert should be uploaded to VRAM
 *   2. Uploads the three expert matrices via the VRAM cache
 *   3. Sets the ESlot's CUDA pointers so the existing GPU path takes over
 *
 * All G10-specific code stays in g10/ — this is the only bridge into the engine.
 */

#ifdef COLI_G10_VRAM_CACHE

#include "../g10/g10_vram_cache.h"

/* Forward declarations needed because we're included mid-file in glm.c */
#ifndef G10_VRAM_CACHE_ACTIVE
#define G10_VRAM_CACHE_ACTIVE

static int g_g10_vram_enabled = 0;   /* set by env var COLI_G10_VRAM_CACHE=n */
static int g_g10_vram_device = 0;    /* GPU device */
static int g_g10_expert_uploaded = 0; /* stats: count of uploads */

/* Called once during model init to initialize the VRAM cache.
 * Pass the CUDA device index and capacity (0 = auto). */
static void g10_vram_init(int capacity) {
    if (!g_cuda_enabled) return;
    if (g10_vram_cache_init(capacity, g_cuda_devices[0]) == 0) {
        g_g10_vram_enabled = 1;
        g_g10_vram_device = g_cuda_devices[0];
    }
}

/* Called after an expert is loaded from disk into ESlot `s`.
 * Uploads the expert weights to VRAM and sets the CUDA tensor pointers
 * so the existing group matmul path can dispatch to GPU.
 *
 * Safe to call on any slot — no-op if VRAM cache disabled or upload fails. */
static void g10_vram_promote_expert(int layer, ESlot *s) {
    if (!g_g10_vram_enabled) return;
    /* Already uploaded? Check cuda pointer. */
    if (s->g.cuda) return;

    G10VramExpert *entry = g10_vram_cache_lookup(layer, s->eid);
    if (entry) {
        /* Already in VRAM cache — just wire the pointers. */
        void *gate = NULL, *up = NULL, *down = NULL;
        g10_vram_cache_get_tensors(entry, &gate, &up, &down);
        s->g.cuda = (ColiCudaTensor *)gate;
        s->u.cuda = (ColiCudaTensor *)up;
        s->d.cuda = (ColiCudaTensor *)down;
        s->g.cuda_eligible = 1;
        s->u.cuda_eligible = 1;
        s->d.cuda_eligible = 1;
        return;
    }

    /* Upload from slab. Use the raw weight/scale pointers. */
    const void *gate_w = s->g.fmt == 0 ? (const void *)s->g.qf
                      : s->g.fmt == 1 ? (const void *)s->g.q8
                      : (const void *)s->g.q4;
    const void *up_w   = s->u.fmt == 0 ? (const void *)s->u.qf
                      : s->u.fmt == 1 ? (const void *)s->u.q8
                      : (const void *)s->u.q4;
    const void *down_w = s->d.fmt == 0 ? (const void *)s->d.qf
                      : s->d.fmt == 1 ? (const void *)s->d.q8
                      : (const void *)s->d.q4;

    entry = g10_vram_cache_insert(layer, s->eid,
                                   gate_w, up_w, down_w,
                                   s->g.s, s->u.s, s->d.s,
                                   s->g.fmt, s->g.I, s->g.O,
                                   g_g10_vram_device);
    if (entry) {
        void *gate = NULL, *up = NULL, *down = NULL;
        g10_vram_cache_get_tensors(entry, &gate, &up, &down);
        s->g.cuda = (ColiCudaTensor *)gate;
        s->u.cuda = (ColiCudaTensor *)up;
        s->d.cuda = (ColiCudaTensor *)down;
        s->g.cuda_eligible = 1;
        s->u.cuda_eligible = 1;
        s->d.cuda_eligible = 1;
        g_g10_expert_uploaded++;
    }
}

/* Called at shutdown to free VRAM cache. */
static void g10_vram_cleanup(void) {
    if (g_g10_vram_enabled) {
        fprintf(stderr, "[G10-VC] shutdown: %d uploads, ~%.2f GB VRAM used\n",
                g_g10_expert_uploaded,
                g10_vram_cache_bytes() / 1e9);
        g10_vram_cache_shutdown();
    }
}

#endif /* G10_VRAM_CACHE_ACTIVE */
#endif /* COLI_G10_VRAM_CACHE */
