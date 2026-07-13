/* g10_vram_hook.h — minimal integration header for inserting VRAM cache into glm.c
 *
 * Usage:
 *   1. In glm.c:
 *      #ifdef COLI_G10_VRAM_CACHE
 *      #include "../g10/g10_vram_cache.h"
 *      static void g10_vram_load_expert(Model *m, int layer, ESlot *s);
 *      #endif
 *
 *   2. Call g10_vram_load_expert(m, layer, &m->ws[q]) after expert_load()
 *      or after pipe_wait().
 *
 *   3. Add -DCOLI_G10_VRAM_CACHE -I../g10 and link g10_vram_cache.o
 */

#ifndef G10_VRAM_HOOK_H
#define G10_VRAM_HOOK_H

#include "glm_forward.h"  /* forward declares Model, ESlot, QT */
#include "../c/backend_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/* After expert weights are loaded into CPU slot s, uploading them to
 * the VRAM cache and setting up CUDA pointers so subsequent accesses
 * use the GPU path automatically (via the existing cuda_eligible check).
 *
 * Call right after expert_load() or pipe_wait() completes for a miss.
 * Safe to call on any slot — if the expert is already GPU-resident or
 * VRAM cache is full, this is a no-op. */
void g10_vram_promote_expert(int layer, ESlot *s, int device);

#ifdef __cplusplus
}
#endif

#endif /* G10_VRAM_HOOK_H */
