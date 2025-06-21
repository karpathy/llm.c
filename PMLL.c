/********************************************************************
  PMLL.c  —  Persistent-Memory Logic Loop (CPU reference)
  • ring-buffer KV store per attention head
  • zero external dependencies
  • thread-safe if you call it from a single OpenMP thread per batch
  • MIT licence (same as llm.c)
*********************************************************************/

#include "pmll.h"           /* public ABI */
#include <stdlib.h>         /* malloc / free */
#include <string.h>         /* memcpy / memset */

/*---------------------------------------------------------------
  Life-cycle helpers
----------------------------------------------------------------*/

/* Allocate and zero-initialise the ring buffers.
   Return 0 on success, −1 on out-of-memory. */
int pmll_init(pmll_state *S, int NH, int hs)
{
    S->T  = 0;
    S->hs = hs;

    const size_t bytes = (size_t)NH * MAX_MEM_T * hs * sizeof(float);

    S->k = (float*)calloc(1, bytes);
    S->v = (float*)calloc(1, bytes);

    return (S->k && S->v) ? 0 : -1;
}

/* Discard history but keep the allocated buffers. */
void pmll_reset(pmll_state *S) { S->T = 0; }

/* Free device memory. */
void pmll_free(pmll_state *S)
{
    free(S->k);
    free(S->v);
    S->k = S->v = NULL;
    S->T = S->hs = 0;
}

/*---------------------------------------------------------------
  Data flow
----------------------------------------------------------------*/

/* Read historic KV for head *h* to caller-provided scratch.
   out_k/out_v length = (Tmem + Tctx) * hs floats. */
void pmll_read(float *out_k, float *out_v,
               const pmll_state *S, int h, int Tctx)
{
    const int hs   = S->hs;
    const int Tmem = S->T;

    const float *src_k = S->k + (size_t)h * MAX_MEM_T * hs;
    const float *src_v = S->v + (size_t)h * MAX_MEM_T * hs;

    /* 1. copy history */
    const size_t hist_bytes = (size_t)Tmem * hs * sizeof(float);
    memcpy(out_k, src_k, hist_bytes);
    memcpy(out_v, src_v, hist_bytes);

    /* 2. zero-pad the fresh region so the attention kernel
          sees “no data” there until we overwrite it.              */
    const size_t pad_bytes  = (size_t)Tctx * hs * sizeof(float);
    memset(out_k + Tmem*hs, 0, pad_bytes);
    memset(out_v + Tmem*hs, 0, pad_bytes);
}

/* Blend-write new KV into the ring buffer.
   gate[t] ∈ [0,1] decides how much of the new vector to keep.
   Pass gate = NULL to “remember everything” (g == 1).            */
void pmll_write(pmll_state *S, int h,
                const float *new_k, const float *new_v,
                int Tctx, const float *gate)
{
    const int hs = S->hs;

    float *dst_k = S->k + (size_t)h * MAX_MEM_T * hs;
    float *dst_v = S->v + (size_t)h * MAX_MEM_T * hs;

    for (int t = 0; t < Tctx; ++t)
    {
        const float g = gate ? gate[t] : 1.0f;          /* default keep-all */
        const int   idx = (S->T + t) % MAX_MEM_T;       /* ring offset      */

        const float *nk = new_k + t*hs;
        const float *nv = new_v + t*hs;
              float *dk = dst_k + idx*hs;
              float *dv = dst_v + idx*hs;

        for (int i = 0; i < hs; ++i) {
            dk[i] = g * nk[i] + (1.f - g) * dk[i];
            dv[i] = g * nv[i] + (1.f - g) * dv[i];
        }
    }

    /* advance ring pointer */
    S->T += Tctx;
    if (S->T > MAX_MEM_T) S->T = MAX_MEM_T;
}
