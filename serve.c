/*
llm-serve: LLaMA inference server with HTTP API.
Loads GGUF model files, serves OpenAI-compatible /v1/completions endpoint.

Usage:
  ./llm-serve -m model.gguf [-p port] [-c context_size] [-t threads]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "llmc/gguf.h"
#include "llmc/gguf_dequant.h"
#include "llmc/bpe_tokenizer.h"
#include "llmc/sampler.h"

// ============================================================================
// LLaMA Model Config

typedef struct {
    int vocab_size;
    int embed_dim;      // embedding_length / n_embd
    int n_layers;
    int n_heads;        // Q heads
    int n_kv_heads;     // KV heads (GQA)
    int ffn_dim;        // feed_forward_length
    int context_len;
    float rope_theta;   // rope.freq_base
    float rms_eps;      // attention.layer_norm_rms_epsilon
    int head_dim;       // embed_dim / n_heads
} LlamaConfig;

// ============================================================================
// LLaMA Model Weights (pointers into GGUF data, dequantized lazily)

typedef struct {
    // token embedding table [vocab_size, embed_dim]
    GGUFTensor *token_embd;

    // per-layer
    GGUFTensor **attn_norm;    // [n_layers] RMSNorm weight [embed_dim]
    GGUFTensor **attn_q;       // [n_layers] [n_heads * head_dim, embed_dim]
    GGUFTensor **attn_k;       // [n_layers] [n_kv_heads * head_dim, embed_dim]
    GGUFTensor **attn_v;       // [n_layers] [n_kv_heads * head_dim, embed_dim]
    GGUFTensor **attn_out;     // [n_layers] [embed_dim, n_heads * head_dim]
    GGUFTensor **ffn_norm;     // [n_layers] RMSNorm weight [embed_dim]
    GGUFTensor **ffn_gate;     // [n_layers] [ffn_dim, embed_dim]
    GGUFTensor **ffn_up;       // [n_layers] [ffn_dim, embed_dim]
    GGUFTensor **ffn_down;     // [n_layers] [embed_dim, ffn_dim]

    // output
    GGUFTensor *output_norm;   // [embed_dim]
    GGUFTensor *output;        // [vocab_size, embed_dim] (may be NULL if tied)
} LlamaWeights;

// ============================================================================
// KV Cache

typedef struct {
    float *k;   // [n_layers, context_len, n_kv_heads, head_dim]
    float *v;   // [n_layers, context_len, n_kv_heads, head_dim]
    int n_layers, context_len, n_kv_heads, head_dim;
} KVCache;

static KVCache kvcache_alloc(int n_layers, int ctx_len, int n_kv_heads, int head_dim) {
    KVCache c;
    c.n_layers   = n_layers;
    c.context_len = ctx_len;
    c.n_kv_heads = n_kv_heads;
    c.head_dim   = head_dim;
    size_t sz = (size_t)n_layers * ctx_len * n_kv_heads * head_dim;
    c.k = (float *)calloc(sz, sizeof(float));
    c.v = (float *)calloc(sz, sizeof(float));
    if (!c.k || !c.v) { fprintf(stderr, "OOM for KV cache\n"); exit(EXIT_FAILURE); }
    return c;
}

static float *kvcache_k(KVCache *c, int layer, int pos) {
    return c->k + ((size_t)layer * c->context_len + pos) * c->n_kv_heads * c->head_dim;
}

static float *kvcache_v(KVCache *c, int layer, int pos) {
    return c->v + ((size_t)layer * c->context_len + pos) * c->n_kv_heads * c->head_dim;
}

// ============================================================================
// Run state (activations)

typedef struct {
    float *x;        // [embed_dim] current residual
    float *xb;       // [embed_dim] scratch
    float *xb2;      // [embed_dim] scratch
    float *hb;       // [ffn_dim]
    float *hb2;      // [ffn_dim]
    float *q;        // [n_heads * head_dim]
    float *k;        // [n_kv_heads * head_dim]
    float *v;        // [n_kv_heads * head_dim]
    float *att;      // [n_heads * context_len]
    float *logits;   // [vocab_size]
} RunState;

static RunState runstate_alloc(LlamaConfig *cfg) {
    RunState s;
    int d = cfg->embed_dim;
    int ff = cfg->ffn_dim;
    int qd = cfg->n_heads * cfg->head_dim;
    int kvd = cfg->n_kv_heads * cfg->head_dim;
    s.x      = (float *)calloc(d,  sizeof(float));
    s.xb     = (float *)calloc(d,  sizeof(float));
    s.xb2    = (float *)calloc(d,  sizeof(float));
    s.hb     = (float *)calloc(ff, sizeof(float));
    s.hb2    = (float *)calloc(ff, sizeof(float));
    s.q      = (float *)calloc(qd, sizeof(float));
    s.k      = (float *)calloc(kvd, sizeof(float));
    s.v      = (float *)calloc(kvd, sizeof(float));
    s.att    = (float *)calloc(cfg->n_heads * cfg->context_len, sizeof(float));
    s.logits = (float *)calloc(cfg->vocab_size, sizeof(float));
    if (!s.x || !s.xb || !s.xb2 || !s.hb || !s.hb2 ||
        !s.q || !s.k || !s.v || !s.att || !s.logits) {
        fprintf(stderr, "OOM for run state\n"); exit(EXIT_FAILURE);
    }
    return s;
}

// ============================================================================
// Math primitives

static void rmsnorm(float *out, const float *x, const float *weight, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = weight[i] * (x[i] * ss);
}

static void softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

// Matrix-vector multiply: out[m] = W[m,n] @ in[n]
// W is stored as a GGUFTensor (rows=m, cols=n), dequantized row by row.
static void matvec(float *out, GGUFTensor *W, const float *in, int m, int n) {
    float *row = (float *)malloc(n * sizeof(float));
    if (!row) { fprintf(stderr, "OOM matvec\n"); exit(EXIT_FAILURE); }
    #pragma omp parallel for schedule(dynamic,16)
    for (int i = 0; i < m; i++) {
        // dequantize row i
        float *row_local = (float *)malloc(n * sizeof(float));
        gguf_dequant_row(W, i, row_local);
        float dot = 0.0f;
        for (int j = 0; j < n; j++) dot += row_local[j] * in[j];
        out[i] = dot;
        free(row_local);
    }
    free(row);
}

// ============================================================================
// RoPE

static void rope_apply(float *q, float *k, int pos, LlamaConfig *cfg) {
    int hd = cfg->head_dim;
    float theta = cfg->rope_theta;
    // Q
    for (int h = 0; h < cfg->n_heads; h++) {
        float *qh = q + h * hd;
        for (int i = 0; i < hd / 2; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / (float)hd);
            float val = (float)pos * freq;
            float cos_val = cosf(val), sin_val = sinf(val);
            float q0 = qh[2*i], q1 = qh[2*i+1];
            qh[2*i]   = q0 * cos_val - q1 * sin_val;
            qh[2*i+1] = q0 * sin_val + q1 * cos_val;
        }
    }
    // K
    for (int h = 0; h < cfg->n_kv_heads; h++) {
        float *kh = k + h * hd;
        for (int i = 0; i < hd / 2; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / (float)hd);
            float val = (float)pos * freq;
            float cos_val = cosf(val), sin_val = sinf(val);
            float k0 = kh[2*i], k1 = kh[2*i+1];
            kh[2*i]   = k0 * cos_val - k1 * sin_val;
            kh[2*i+1] = k0 * sin_val + k1 * cos_val;
        }
    }
}

// ============================================================================
// Forward pass (single token at position `pos`)

static void llama_forward(LlamaConfig *cfg, LlamaWeights *w, RunState *s,
                           KVCache *kv, int token_id, int pos) {
    int d    = cfg->embed_dim;
    int hd   = cfg->head_dim;
    int nh   = cfg->n_heads;
    int nkv  = cfg->n_kv_heads;
    int kv_mul = nh / nkv; // GQA expansion factor
    int ff   = cfg->ffn_dim;
    float eps = cfg->rms_eps;

    // --- Token embedding ---
    // token_embd: [vocab_size, embed_dim] stored as (vocab_size rows, embed_dim cols)
    gguf_dequant_row(w->token_embd, token_id, s->x);

    // --- Transformer layers ---
    for (int l = 0; l < cfg->n_layers; l++) {
        // RMSNorm before attention
        float *norm_w = (float *)malloc(d * sizeof(float));
        gguf_dequant_row(w->attn_norm[l], 0, norm_w);
        rmsnorm(s->xb, s->x, norm_w, d, eps);
        free(norm_w);

        // Q, K, V projections
        matvec(s->q, w->attn_q[l], s->xb, nh * hd, d);
        matvec(s->k, w->attn_k[l], s->xb, nkv * hd, d);
        matvec(s->v, w->attn_v[l], s->xb, nkv * hd, d);

        // RoPE
        rope_apply(s->q, s->k, pos, cfg);

        // Store K, V in cache
        memcpy(kvcache_k(kv, l, pos), s->k, nkv * hd * sizeof(float));
        memcpy(kvcache_v(kv, l, pos), s->v, nkv * hd * sizeof(float));

        // Attention: for each Q head
        memset(s->xb, 0, d * sizeof(float));
        float scale = 1.0f / sqrtf((float)hd);

        for (int h = 0; h < nh; h++) {
            float *qh  = s->q + h * hd;
            int    kvh = h / kv_mul;  // which KV head to use (GQA)
            float *att = s->att + h * cfg->context_len;

            // Dot product Q with all K in cache [0..pos]
            for (int t = 0; t <= pos; t++) {
                float *kh = kvcache_k(kv, l, t) + kvh * hd;
                float dot = 0.0f;
                for (int i = 0; i < hd; i++) dot += qh[i] * kh[i];
                att[t] = dot * scale;
            }
            softmax(att, pos + 1);

            // Weighted sum of V
            float *xbh = s->xb + h * hd;
            memset(xbh, 0, hd * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *vh = kvcache_v(kv, l, t) + kvh * hd;
                float a   = att[t];
                for (int i = 0; i < hd; i++) xbh[i] += a * vh[i];
            }
        }

        // Output projection + residual
        matvec(s->xb2, w->attn_out[l], s->xb, d, nh * hd);
        for (int i = 0; i < d; i++) s->x[i] += s->xb2[i];

        // RMSNorm before FFN
        float *fnorm_w = (float *)malloc(d * sizeof(float));
        gguf_dequant_row(w->ffn_norm[l], 0, fnorm_w);
        rmsnorm(s->xb, s->x, fnorm_w, d, eps);
        free(fnorm_w);

        // SwiGLU FFN
        matvec(s->hb,  w->ffn_gate[l], s->xb, ff, d);
        matvec(s->hb2, w->ffn_up[l],   s->xb, ff, d);
        for (int i = 0; i < ff; i++) s->hb[i] = silu(s->hb[i]) * s->hb2[i];
        matvec(s->xb,  w->ffn_down[l], s->hb,  d, ff);

        // Residual
        for (int i = 0; i < d; i++) s->x[i] += s->xb[i];
    }

    // --- Output norm + logits ---
    float *onorm_w = (float *)malloc(d * sizeof(float));
    gguf_dequant_row(w->output_norm, 0, onorm_w);
    rmsnorm(s->xb, s->x, onorm_w, d, eps);
    free(onorm_w);

    GGUFTensor *out_W = w->output ? w->output : w->token_embd;
    matvec(s->logits, out_W, s->xb, cfg->vocab_size, d);
}

// ============================================================================
// Sampling helpers

static int sample_argmax(const float *logits, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    return best;
}

static int sample_topp(const float *logits, int n, float temperature, float top_p,
                        unsigned long long *rng_state) {
    if (temperature <= 0.0f) return sample_argmax(logits, n);

    // Temperature scaling + softmax
    float *probs = (float *)malloc(n * sizeof(float));
    float max_l = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > max_l) max_l = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { probs[i] = expf((logits[i] - max_l) / temperature); sum += probs[i]; }
    for (int i = 0; i < n; i++) probs[i] /= sum;

    // Top-p nucleus sampling
    // Sort by prob descending (simple insertion sort for small n, but vocab can be large)
    // For efficiency: find cumulative sum threshold
    float coin = random_f32(rng_state);
    if (top_p >= 1.0f) {
        // just sample from full distribution
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += probs[i];
            if (coin < cdf) { free(probs); return i; }
        }
        free(probs);
        return n - 1;
    }

    // Build sorted index array (selection sort subset up to top_p)
    int *idx = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;
    float cumsum = 0.0f;
    int nucleus_size = 0;
    for (int i = 0; i < n - 1; i++) {
        // find max in [i, n)
        int max_j = i;
        for (int j = i + 1; j < n; j++) {
            if (probs[idx[j]] > probs[idx[max_j]]) max_j = j;
        }
        int tmp = idx[i]; idx[i] = idx[max_j]; idx[max_j] = tmp;
        cumsum += probs[idx[i]];
        nucleus_size = i + 1;
        if (cumsum >= top_p) break;
    }
    if (nucleus_size == 0) nucleus_size = 1;

    // Re-normalize nucleus
    float nsum = 0.0f;
    for (int i = 0; i < nucleus_size; i++) nsum += probs[idx[i]];
    float cdf = 0.0f;
    int result = idx[nucleus_size - 1];
    for (int i = 0; i < nucleus_size; i++) {
        cdf += probs[idx[i]] / nsum;
        if (coin < cdf) { result = idx[i]; break; }
    }
    free(probs);
    free(idx);
    return result;
}

// ============================================================================
// Model loading

typedef struct {
    LlamaConfig    cfg;
    LlamaWeights   weights;
    KVCache        kv;
    RunState       state;
    BPETokenizer   tokenizer;
    GGUFContext   *gguf;
    unsigned long long rng_state;
} LlamaModel;

static GGUFTensor *require_tensor(GGUFContext *ctx, const char *name) {
    GGUFTensor *t = gguf_find_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "WARN: tensor '%s' not found\n", name);
    }
    return t;
}

static LlamaModel *llama_load(const char *path, int context_override) {
    fprintf(stderr, "Loading GGUF model: %s\n", path);
    gguf_dequant_init();  // ensure all dequant types are registered
    GGUFContext *ctx = gguf_load(path);

    const char *arch = gguf_get_str(ctx, "general.architecture", "llama");

    char key[256];
    LlamaConfig cfg;
    memset(&cfg, 0, sizeof(cfg));

    #define ARCH_KEY(k) (snprintf(key, sizeof(key), "%s.%s", arch, k), key)

    cfg.vocab_size  = (int)gguf_get_u64(ctx, ARCH_KEY("vocab_size"), 32000);
    cfg.embed_dim   = (int)gguf_get_u64(ctx, ARCH_KEY("embedding_length"), 4096);
    cfg.n_layers    = (int)gguf_get_u64(ctx, ARCH_KEY("block_count"), 32);
    cfg.n_heads     = (int)gguf_get_u64(ctx, ARCH_KEY("attention.head_count"), 32);
    cfg.n_kv_heads  = (int)gguf_get_u64(ctx, ARCH_KEY("attention.head_count_kv"), cfg.n_heads);
    cfg.ffn_dim     = (int)gguf_get_u64(ctx, ARCH_KEY("feed_forward_length"), 11008);
    cfg.context_len = (int)gguf_get_u64(ctx, ARCH_KEY("context_length"), 2048);
    cfg.rope_theta  = gguf_get_f32(ctx, ARCH_KEY("rope.freq_base"), 10000.0f);
    cfg.rms_eps     = gguf_get_f32(ctx, ARCH_KEY("attention.layer_norm_rms_epsilon"), 1e-5f);
    cfg.head_dim    = cfg.embed_dim / cfg.n_heads;

    if (context_override > 0 && context_override < cfg.context_len)
        cfg.context_len = context_override;

    // Also try vocab size from tokenizer
    {
        GGUFValue *tv = gguf_find_kv(ctx, "tokenizer.ggml.tokens");
        if (tv && tv->type == GGUF_TYPE_ARRAY && (int)tv->arr.count > cfg.vocab_size)
            cfg.vocab_size = (int)tv->arr.count;
    }

    fprintf(stderr, "  arch=%s layers=%d heads=%d kv_heads=%d embed=%d ffn=%d ctx=%d\n",
            arch, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
            cfg.embed_dim, cfg.ffn_dim, cfg.context_len);

    // Load weights
    LlamaWeights w;
    memset(&w, 0, sizeof(w));
    w.token_embd  = require_tensor(ctx, "token_embd.weight");
    w.output_norm = require_tensor(ctx, "output_norm.weight");
    w.output      = gguf_find_tensor(ctx, "output.weight");  // may be NULL (tied)

    w.attn_norm = (GGUFTensor **)calloc(cfg.n_layers, sizeof(GGUFTensor *));
    w.attn_q    = (GGUFTensor **)calloc(cfg.n_layers, sizeof(GGUFTensor *));
    w.attn_k    = (GGUFTensor **)calloc(cfg.n_layers, sizeof(GGUFTensor *));
    w.attn_v    = (GGUFTensor **)calloc(cfg.n_layers, sizeof(GGUFTensor *));
    w.attn_out  = (GGUFTensor **)calloc(cfg.n_layers, sizeof(GGUFTensor *));
    w.ffn_norm  = (GGUFTensor **)calloc(cfg.n_layers, sizeof(GGUFTensor *));
    w.ffn_gate  = (GGUFTensor **)calloc(cfg.n_layers, sizeof(GGUFTensor *));
    w.ffn_up    = (GGUFTensor **)calloc(cfg.n_layers, sizeof(GGUFTensor *));
    w.ffn_down  = (GGUFTensor **)calloc(cfg.n_layers, sizeof(GGUFTensor *));

    for (int l = 0; l < cfg.n_layers; l++) {
        snprintf(key, sizeof(key), "blk.%d.attn_norm.weight", l);
        w.attn_norm[l] = require_tensor(ctx, key);
        snprintf(key, sizeof(key), "blk.%d.attn_q.weight", l);
        w.attn_q[l] = require_tensor(ctx, key);
        snprintf(key, sizeof(key), "blk.%d.attn_k.weight", l);
        w.attn_k[l] = require_tensor(ctx, key);
        snprintf(key, sizeof(key), "blk.%d.attn_v.weight", l);
        w.attn_v[l] = require_tensor(ctx, key);
        snprintf(key, sizeof(key), "blk.%d.attn_output.weight", l);
        w.attn_out[l] = require_tensor(ctx, key);
        snprintf(key, sizeof(key), "blk.%d.ffn_norm.weight", l);
        w.ffn_norm[l] = require_tensor(ctx, key);
        snprintf(key, sizeof(key), "blk.%d.ffn_gate.weight", l);
        w.ffn_gate[l] = require_tensor(ctx, key);
        snprintf(key, sizeof(key), "blk.%d.ffn_up.weight", l);
        w.ffn_up[l] = require_tensor(ctx, key);
        snprintf(key, sizeof(key), "blk.%d.ffn_down.weight", l);
        w.ffn_down[l] = require_tensor(ctx, key);
    }

    LlamaModel *model = (LlamaModel *)calloc(1, sizeof(LlamaModel));
    model->cfg     = cfg;
    model->weights = w;
    model->gguf    = ctx;
    model->kv      = kvcache_alloc(cfg.n_layers, cfg.context_len, cfg.n_kv_heads, cfg.head_dim);
    model->state   = runstate_alloc(&cfg);
    model->rng_state = (unsigned long long)time(NULL);

    bpe_tokenizer_init(&model->tokenizer, ctx);

    fprintf(stderr, "  vocab_size=%d bos=%d eos=%d\n",
            model->tokenizer.vocab_size,
            model->tokenizer.bos_id,
            model->tokenizer.eos_id);
    fprintf(stderr, "Model loaded.\n");
    return model;
}

// ============================================================================
// Generate tokens

typedef struct {
    int    *tokens;
    int     n_tokens;
    float   temperature;
    float   top_p;
    int     max_new_tokens;
    int     stream;
    int     client_fd;
} GenParams;

// Returns generated text (malloc'd). Caller must free.
static char *generate(LlamaModel *model, GenParams *p) {
    LlamaConfig *cfg = &model->cfg;
    BPETokenizer *tok = &model->tokenizer;

    // Reset KV cache
    int kv_stride = cfg->context_len * cfg->n_kv_heads * cfg->head_dim;
    memset(model->kv.k, 0, (size_t)cfg->n_layers * kv_stride * sizeof(float));
    memset(model->kv.v, 0, (size_t)cfg->n_layers * kv_stride * sizeof(float));

    size_t out_cap = 4096;
    char *out_text = (char *)malloc(out_cap);
    if (!out_text) { fprintf(stderr, "OOM\n"); exit(EXIT_FAILURE); }
    out_text[0] = '\0';
    size_t out_len = 0;

    int pos = 0;
    int n_prompt = p->n_tokens;

    // Prefill prompt tokens
    for (int i = 0; i < n_prompt && pos < cfg->context_len; i++, pos++) {
        llama_forward(cfg, &model->weights, &model->state, &model->kv, p->tokens[i], pos);
    }

    int n_gen = 0;
    char piece_buf[BPE_MAX_TOKEN];
    char sse_buf[BPE_MAX_TOKEN + 128];

    while (n_gen < p->max_new_tokens && pos < cfg->context_len) {
        // Sample next token
        int next;
        if (p->temperature <= 0.0f) {
            next = sample_argmax(model->state.logits, cfg->vocab_size);
        } else {
            next = sample_topp(model->state.logits, cfg->vocab_size,
                               p->temperature, p->top_p, &model->rng_state);
        }

        // EOS check
        if (next == tok->eos_id) break;

        // Decode token to text
        bpe_decode_piece(tok, next, piece_buf, sizeof(piece_buf));

        // Append to output
        size_t plen = strlen(piece_buf);
        if (out_len + plen + 1 > out_cap) {
            out_cap *= 2;
            out_text = (char *)realloc(out_text, out_cap);
            if (!out_text) { fprintf(stderr, "OOM\n"); exit(EXIT_FAILURE); }
        }
        memcpy(out_text + out_len, piece_buf, plen);
        out_len += plen;
        out_text[out_len] = '\0';

        // Stream SSE if requested
        if (p->stream && p->client_fd >= 0) {
            int slen = snprintf(sse_buf, sizeof(sse_buf),
                "data: {\"choices\":[{\"text\":\"%s\",\"finish_reason\":null}]}\n\n",
                piece_buf);
            send(p->client_fd, sse_buf, slen, MSG_NOSIGNAL);
        }

        // Forward for next token
        llama_forward(cfg, &model->weights, &model->state, &model->kv, next, pos);
        pos++;
        n_gen++;
    }

    if (p->stream && p->client_fd >= 0) {
        const char *done = "data: [DONE]\n\n";
        send(p->client_fd, done, strlen(done), MSG_NOSIGNAL);
    }

    return out_text;
}

// ============================================================================
// JSON helpers

// Simple JSON string escape into dst (returns chars written, not including \0)
static int json_escape(const char *src, char *dst, int dst_cap) {
    int j = 0;
    for (int i = 0; src[i] && j < dst_cap - 4; i++) {
        unsigned char c = (unsigned char)src[i];
        if (c == '"')       { dst[j++] = '\\'; dst[j++] = '"'; }
        else if (c == '\\') { dst[j++] = '\\'; dst[j++] = '\\'; }
        else if (c == '\n') { dst[j++] = '\\'; dst[j++] = 'n'; }
        else if (c == '\r') { dst[j++] = '\\'; dst[j++] = 'r'; }
        else if (c == '\t') { dst[j++] = '\\'; dst[j++] = 't'; }
        else if (c < 0x20)  { j += snprintf(dst+j, dst_cap-j, "\\u%04X", c); }
        else                { dst[j++] = (char)c; }
    }
    dst[j] = '\0';
    return j;
}

// Extract string value for a key from JSON (very naive, no nesting)
// Returns pointer into buf (null-terminated), or NULL.
static char *json_get_str(const char *json, const char *key, char *buf, int buf_size) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return NULL;
    p += strlen(search);
    while (*p == ' ' || *p == ':' || *p == ' ') p++;
    if (*p != '"') return NULL;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < buf_size - 1) {
        if (*p == '\\' && *(p+1)) {
            p++;
            if (*p == 'n') buf[i++] = '\n';
            else if (*p == 't') buf[i++] = '\t';
            else if (*p == 'r') buf[i++] = '\r';
            else buf[i++] = *p;
        } else {
            buf[i++] = *p;
        }
        p++;
    }
    buf[i] = '\0';
    return buf;
}

static float json_get_float(const char *json, const char *key, float def) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return def;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    if (*p == '"') return def;
    return (float)atof(p);
}

static int json_get_int(const char *json, const char *key, int def) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return def;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    if (*p == '"') return def;
    return atoi(p);
}

static int json_get_bool(const char *json, const char *key, int def) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return def;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    if (strncmp(p, "true", 4) == 0) return 1;
    if (strncmp(p, "false", 5) == 0) return 0;
    return def;
}

// ============================================================================
// HTTP server

#define HTTP_MAX_REQUEST (1 << 20)  // 1 MB

static void send_response(int fd, int status, const char *content_type,
                           const char *body, int body_len, int is_stream) {
    char header[512];
    int hlen;
    if (is_stream) {
        hlen = snprintf(header, sizeof(header),
            "HTTP/1.1 %d OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "\r\n", status);
    } else {
        hlen = snprintf(header, sizeof(header),
            "HTTP/1.1 %d OK\r\n"
            "Content-Type: %s\r\n"
            "Content-Length: %d\r\n"
            "Connection: close\r\n"
            "\r\n", status, content_type, body_len);
    }
    send(fd, header, hlen, MSG_NOSIGNAL);
    if (body && body_len > 0 && !is_stream)
        send(fd, body, body_len, MSG_NOSIGNAL);
}

static void send_error(int fd, int status, const char *msg) {
    char body[256];
    int blen = snprintf(body, sizeof(body),
        "{\"error\":{\"message\":\"%s\",\"code\":%d}}", msg, status);
    char header[256];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d Error\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n", status, blen);
    send(fd, header, hlen, MSG_NOSIGNAL);
    send(fd, body, blen, MSG_NOSIGNAL);
}

static void handle_request(int fd, LlamaModel *model, const char *method,
                            const char *path, const char *body) {
    // GET /health
    if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0) {
        const char *resp = "{\"status\":\"ok\"}";
        send_response(fd, 200, "application/json", resp, (int)strlen(resp), 0);
        return;
    }

    // POST /v1/completions
    if (strcmp(method, "POST") == 0 &&
        (strcmp(path, "/v1/completions") == 0 || strcmp(path, "/completions") == 0)) {

        if (!body || body[0] == '\0') {
            send_error(fd, 400, "empty body"); return;
        }

        char prompt_buf[65536];
        if (!json_get_str(body, "prompt", prompt_buf, sizeof(prompt_buf))) {
            send_error(fd, 400, "missing prompt"); return;
        }

        float temperature = json_get_float(body, "temperature", 0.7f);
        float top_p       = json_get_float(body, "top_p", 0.9f);
        int max_tokens    = json_get_int(body, "max_tokens", 128);
        int stream        = json_get_bool(body, "stream", 0);

        if (max_tokens <= 0) max_tokens = 128;
        if (max_tokens > model->cfg.context_len) max_tokens = model->cfg.context_len;

        // Tokenize prompt
        int *prompt_tokens = (int *)malloc((strlen(prompt_buf) + 4) * sizeof(int));
        int n_prompt = bpe_encode(&model->tokenizer, prompt_buf, 1,
                                   prompt_tokens, (int)strlen(prompt_buf) + 4);

        // Limit prompt to context
        int ctx = model->cfg.context_len;
        if (n_prompt >= ctx) n_prompt = ctx - 1;

        fprintf(stderr, "Generating: prompt_tokens=%d max_new=%d temp=%.2f\n",
                n_prompt, max_tokens, temperature);

        GenParams gp;
        gp.tokens        = prompt_tokens;
        gp.n_tokens      = n_prompt;
        gp.temperature   = temperature;
        gp.top_p         = top_p;
        gp.max_new_tokens = max_tokens;
        gp.stream        = stream;
        gp.client_fd     = stream ? fd : -1;

        if (stream) {
            send_response(fd, 200, NULL, NULL, 0, 1);
        }

        char *gen_text = generate(model, &gp);
        free(prompt_tokens);

        if (!stream) {
            // Escape generated text for JSON
            size_t esc_size = strlen(gen_text) * 6 + 4;
            char *esc_text = (char *)malloc(esc_size);
            json_escape(gen_text, esc_text, (int)esc_size);

            size_t resp_size = esc_size + 256;
            char *resp = (char *)malloc(resp_size);
            int rlen = snprintf(resp, resp_size,
                "{\"choices\":[{\"text\":\"%s\",\"finish_reason\":\"stop\"}],"
                "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d}}",
                esc_text, n_prompt, (int)strlen(gen_text));

            send_response(fd, 200, "application/json", resp, rlen, 0);
            free(resp);
            free(esc_text);
        }

        free(gen_text);
        return;
    }

    send_error(fd, 404, "not found");
}

static void serve_loop(LlamaModel *model, int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); exit(EXIT_FAILURE); }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons((uint16_t)port);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 16) < 0) {
        perror("listen"); exit(EXIT_FAILURE);
    }

    fprintf(stderr, "llm-serve listening on http://0.0.0.0:%d\n", port);
    fprintf(stderr, "  POST /v1/completions\n");
    fprintf(stderr, "  GET  /health\n");

    char *req_buf = (char *)malloc(HTTP_MAX_REQUEST);
    if (!req_buf) { fprintf(stderr, "OOM\n"); exit(EXIT_FAILURE); }

    while (1) {
        struct sockaddr_in client_addr;
        socklen_t clen = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &clen);
        if (client_fd < 0) { perror("accept"); continue; }

        // Read request
        int total = 0;
        while (total < HTTP_MAX_REQUEST - 1) {
            int n = (int)recv(client_fd, req_buf + total, HTTP_MAX_REQUEST - 1 - total, 0);
            if (n <= 0) break;
            total += n;
            req_buf[total] = '\0';
            // Check if we have a complete HTTP request
            if (strstr(req_buf, "\r\n\r\n")) {
                // Check Content-Length
                const char *cl_hdr = strstr(req_buf, "Content-Length:");
                if (!cl_hdr) cl_hdr = strstr(req_buf, "content-length:");
                if (cl_hdr) {
                    int cl = atoi(cl_hdr + 15);
                    const char *body_start = strstr(req_buf, "\r\n\r\n");
                    if (body_start) {
                        int body_received = (int)(total - (body_start + 4 - req_buf));
                        if (body_received >= cl) break;
                    }
                } else {
                    break;
                }
            }
        }
        req_buf[total] = '\0';

        // Parse request line
        char method[16] = {0}, path[256] = {0};
        sscanf(req_buf, "%15s %255s", method, path);

        // Find body
        const char *body_sep = strstr(req_buf, "\r\n\r\n");
        const char *body = body_sep ? body_sep + 4 : "";

        handle_request(client_fd, model, method, path, body);
        close(client_fd);
    }

    free(req_buf);
    close(server_fd);
}

// ============================================================================
// CLI

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  -m, --model <path>     Path to GGUF model file (required)\n"
        "  -p, --port <port>      Server port (default: 8080)\n"
        "  -c, --context <size>   Context size override\n"
        "  -t, --threads <n>      Number of OpenMP threads (default: 4)\n"
        "  -h, --help             Show help\n",
        prog);
}

int main(int argc, char *argv[]) {
    const char *model_path = NULL;
    int port = 8080;
    int context_size = 0;
    int n_threads = 4;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i+1 < argc) {
            model_path = argv[++i];
        } else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--port") == 0) && i+1 < argc) {
            port = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--context") == 0) && i+1 < argc) {
            context_size = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) && i+1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: -m <model.gguf> is required\n");
        print_usage(argv[0]);
        return 1;
    }

#ifdef _OPENMP
    omp_set_num_threads(n_threads);
    fprintf(stderr, "OpenMP threads: %d\n", n_threads);
#else
    (void)n_threads;
#endif

    signal(SIGPIPE, SIG_IGN);

    LlamaModel *model = llama_load(model_path, context_size);
    serve_loop(model, port);

    return 0;
}
