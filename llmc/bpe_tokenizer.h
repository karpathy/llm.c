/*
BPE Tokenizer loaded from GGUF metadata.
Supports SentencePiece-style (LLaMA 2) and tiktoken-style (LLaMA 3) vocabularies.
*/
#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "gguf.h"

// ----------------------------------------------------------------------------
// BPE Tokenizer

#define BPE_MAX_VOCAB  200000
#define BPE_MAX_MERGES 200000
#define BPE_MAX_TOKEN  512

typedef struct {
    char   **tokens;      // vocab: token strings (null-terminated)
    float   *scores;      // token scores (for SentencePiece)
    int     *token_type;  // token type flags
    int      vocab_size;
    int      bos_id;
    int      eos_id;

    // merge rules (for explicit BPE like tiktoken)
    char   **merge_left;
    char   **merge_right;
    int      n_merges;
    int      has_merges;  // 0 = SentencePiece score-based, 1 = explicit merges

    // byte-level fallback table
    int      byte_token[256];  // byte value -> token id, -1 if none
} BPETokenizer;

// ----------------------------------------------------------------------------
// UTF-8 helpers

static int utf8_char_len(unsigned char c) {
    if (c < 0x80) return 1;
    if (c < 0xE0) return 2;
    if (c < 0xF0) return 3;
    return 4;
}

// Check if s starts with the UTF-8 encoding of U+2581 (▁, the SentencePiece space)
// U+2581 encodes as E2 96 81
static int is_sp_space(const char *s) {
    return (unsigned char)s[0] == 0xE2 &&
           (unsigned char)s[1] == 0x96 &&
           (unsigned char)s[2] == 0x81;
}

// ----------------------------------------------------------------------------
// Load from GGUF

static void bpe_tokenizer_init(BPETokenizer *tok, GGUFContext *ctx) {
    memset(tok, 0, sizeof(*tok));

    // bos/eos
    tok->bos_id = (int)gguf_get_u64(ctx, "tokenizer.ggml.bos_token_id", 1);
    tok->eos_id = (int)gguf_get_u64(ctx, "tokenizer.ggml.eos_token_id", 2);

    // tokens array
    GGUFValue *tokens_val = gguf_find_kv(ctx, "tokenizer.ggml.tokens");
    if (!tokens_val || tokens_val->type != GGUF_TYPE_ARRAY) {
        fprintf(stderr, "BPE: no tokenizer.ggml.tokens in GGUF\n");
        exit(EXIT_FAILURE);
    }
    tok->vocab_size = (int)tokens_val->arr.count;
    tok->tokens = (char **)calloc(tok->vocab_size, sizeof(char *));
    for (int i = 0; i < tok->vocab_size; i++) {
        tok->tokens[i] = strdup(tokens_val->arr.items[i].str);
    }

    // scores (optional)
    GGUFValue *scores_val = gguf_find_kv(ctx, "tokenizer.ggml.scores");
    if (scores_val && scores_val->type == GGUF_TYPE_ARRAY) {
        tok->scores = (float *)calloc(tok->vocab_size, sizeof(float));
        for (int i = 0; i < tok->vocab_size && (uint64_t)i < scores_val->arr.count; i++) {
            GGUFValue *sv = &scores_val->arr.items[i];
            tok->scores[i] = (sv->type == GGUF_TYPE_FLOAT32) ? sv->f32 : 0.0f;
        }
    }

    // token types (optional)
    GGUFValue *type_val = gguf_find_kv(ctx, "tokenizer.ggml.token_type");
    if (type_val && type_val->type == GGUF_TYPE_ARRAY) {
        tok->token_type = (int *)calloc(tok->vocab_size, sizeof(int));
        for (int i = 0; i < tok->vocab_size && (uint64_t)i < type_val->arr.count; i++) {
            GGUFValue *tv = &type_val->arr.items[i];
            tok->token_type[i] = (tv->type == GGUF_TYPE_INT32) ? tv->i32 : 0;
        }
    }

    // explicit merges (optional, tiktoken-style)
    GGUFValue *merges_val = gguf_find_kv(ctx, "tokenizer.ggml.merges");
    if (merges_val && merges_val->type == GGUF_TYPE_ARRAY && merges_val->arr.count > 0) {
        tok->has_merges = 1;
        tok->n_merges = (int)merges_val->arr.count;
        tok->merge_left  = (char **)calloc(tok->n_merges, sizeof(char *));
        tok->merge_right = (char **)calloc(tok->n_merges, sizeof(char *));
        for (int i = 0; i < tok->n_merges; i++) {
            const char *entry = merges_val->arr.items[i].str;
            // format: "left right"
            const char *sp = strchr(entry, ' ');
            if (sp) {
                size_t llen = (size_t)(sp - entry);
                tok->merge_left[i]  = (char *)malloc(llen + 1);
                memcpy(tok->merge_left[i], entry, llen);
                tok->merge_left[i][llen] = '\0';
                tok->merge_right[i] = strdup(sp + 1);
            } else {
                tok->merge_left[i]  = strdup(entry);
                tok->merge_right[i] = strdup("");
            }
        }
    }

    // build byte fallback table
    memset(tok->byte_token, -1, sizeof(tok->byte_token));
    // Look for tokens of the form <0xNN>
    for (int i = 0; i < tok->vocab_size; i++) {
        const char *t = tok->tokens[i];
        unsigned int byte_val;
        if (sscanf(t, "<0x%02X>", &byte_val) == 1 && byte_val < 256) {
            tok->byte_token[byte_val] = i;
        }
    }
}

// ----------------------------------------------------------------------------
// Decode: token_id -> string

static const char *bpe_decode(BPETokenizer *tok, int token_id) {
    if (token_id < 0 || token_id >= tok->vocab_size) return "";
    return tok->tokens[token_id];
}

// Decode a token, converting SentencePiece ▁ to space
static void bpe_decode_piece(BPETokenizer *tok, int token_id, char *out, int out_size) {
    const char *piece = bpe_decode(tok, token_id);
    int j = 0;
    const char *p = piece;
    while (*p && j < out_size - 1) {
        if (is_sp_space(p) && p[3] == '\0' || is_sp_space(p)) {
            out[j++] = ' ';
            p += 3;  // skip 3-byte UTF-8 sequence
        } else {
            out[j++] = *p++;
        }
    }
    out[j] = '\0';
}

// ----------------------------------------------------------------------------
// Symbol list for BPE encoding

typedef struct BPESym {
    int token_id;       // -1 if raw byte
    char text[BPE_MAX_TOKEN];
    int prev, next;     // doubly linked list indices
} BPESym;

// Find token id by exact string match
static int bpe_find_token(BPETokenizer *tok, const char *s) {
    // linear scan; for large vocabs a hash table would be better
    for (int i = 0; i < tok->vocab_size; i++) {
        if (strcmp(tok->tokens[i], s) == 0) return i;
    }
    return -1;
}

// Find best merge pair by score (SentencePiece style)
static int bpe_find_best_pair(BPETokenizer *tok, BPESym *syms, int n, int *out_i) {
    float best_score = -1e38f;
    int best_idx = -1;
    int merged_id = -1;

    for (int i = 0; i < n - 1; ) {
        if (syms[i].next < 0) { i++; continue; }
        int j = syms[i].next;
        // try merging syms[i] and syms[j]
        char merged[BPE_MAX_TOKEN * 2];
        int mlen = snprintf(merged, sizeof(merged), "%s%s", syms[i].text, syms[j].text);
        if (mlen < (int)sizeof(merged)) {
            int tid = bpe_find_token(tok, merged);
            if (tid >= 0) {
                float score = tok->scores ? tok->scores[tid] : 0.0f;
                if (score > best_score) {
                    best_score = score;
                    best_idx = i;
                    merged_id = tid;
                }
            }
        }
        i = j;
    }
    *out_i = best_idx;
    return merged_id;
}

// BPE encode using merge rules (tiktoken-style)
static int bpe_merge_rank(BPETokenizer *tok, const char *left, const char *right) {
    for (int i = 0; i < tok->n_merges; i++) {
        if (strcmp(tok->merge_left[i], left) == 0 &&
            strcmp(tok->merge_right[i], right) == 0) {
            return i;
        }
    }
    return INT32_MAX;
}

// Main encode function
// Returns number of tokens. out_tokens must have capacity for len+2 tokens.
static int bpe_encode(BPETokenizer *tok, const char *text, int add_bos,
                       int *out_tokens, int max_tokens) {
    int n_out = 0;
    if (add_bos && n_out < max_tokens) {
        out_tokens[n_out++] = tok->bos_id;
    }

    size_t text_len = strlen(text);
    if (text_len == 0) return n_out;

    // SentencePiece: prepend ▁ (U+2581 = E2 96 81) to represent leading space
    // We handle this by checking if first token has ▁ prefix in vocab

    // Build initial symbol list from UTF-8 characters
    // (or byte-level for unknown chars)
    // We use a simple array with a linked list overlay
    int max_syms = (int)(text_len + 4);
    BPESym *syms = (BPESym *)calloc(max_syms, sizeof(BPESym));
    if (!syms) { fprintf(stderr, "BPE: OOM\n"); exit(EXIT_FAILURE); }

    int n_syms = 0;

    // SentencePiece: treat text as having a space prepended
    // Check if vocab uses ▁ convention
    int sp_style = (bpe_find_token(tok, "\xE2\x96\x81") >= 0 ||
                    bpe_find_token(tok, "\xE2\x96\x81 ") >= 0);

    if (sp_style) {
        // Prepend ▁ to first character
        const char *p = text;
        int first = 1;
        while (*p && n_syms < max_syms) {
            int clen = utf8_char_len((unsigned char)*p);
            char buf[BPE_MAX_TOKEN];
            int blen = 0;
            if (first) {
                // prepend ▁ (3 bytes)
                buf[0] = '\xE2'; buf[1] = '\x96'; buf[2] = '\x81';
                blen = 3;
                first = 0;
            }
            for (int k = 0; k < clen && blen < (int)sizeof(buf) - 1; k++) {
                buf[blen++] = p[k];
            }
            buf[blen] = '\0';
            int tid = bpe_find_token(tok, buf);
            if (tid < 0) {
                // try without ▁ prefix
                char buf2[BPE_MAX_TOKEN];
                strncpy(buf2, buf + (blen > clen ? 3 : 0), sizeof(buf2));
                tid = bpe_find_token(tok, buf2);
                if (tid < 0 && blen == 1) {
                    // byte fallback
                    tid = tok->byte_token[(unsigned char)buf[0]];
                }
            }
            syms[n_syms].token_id = tid;
            strncpy(syms[n_syms].text, buf, BPE_MAX_TOKEN - 1);
            syms[n_syms].prev = n_syms - 1;
            syms[n_syms].next = n_syms + 1;
            n_syms++;
            p += clen;
        }
    } else {
        // Byte-level BPE (GPT-2 style or tiktoken)
        const char *p = text;
        while (*p && n_syms < max_syms) {
            int clen = utf8_char_len((unsigned char)*p);
            char buf[BPE_MAX_TOKEN];
            int blen = 0;
            for (int k = 0; k < clen && blen < (int)sizeof(buf) - 1; k++) {
                buf[blen++] = p[k];
            }
            buf[blen] = '\0';
            int tid = bpe_find_token(tok, buf);
            if (tid < 0) {
                // byte fallback
                for (int k = 0; k < blen && n_syms < max_syms; k++) {
                    syms[n_syms].token_id = tok->byte_token[(unsigned char)buf[k]];
                    syms[n_syms].text[0] = buf[k];
                    syms[n_syms].text[1] = '\0';
                    syms[n_syms].prev = n_syms - 1;
                    syms[n_syms].next = n_syms + 1;
                    n_syms++;
                }
                p += clen;
                continue;
            }
            syms[n_syms].token_id = tid;
            strncpy(syms[n_syms].text, buf, BPE_MAX_TOKEN - 1);
            syms[n_syms].prev = n_syms - 1;
            syms[n_syms].next = n_syms + 1;
            n_syms++;
            p += clen;
        }
    }

    if (n_syms > 0) {
        syms[0].prev = -1;
        syms[n_syms - 1].next = -1;
    }

    // BPE merge loop
    if (!tok->has_merges) {
        // SentencePiece score-based merges
        while (1) {
            int best_i;
            int merged_id = bpe_find_best_pair(tok, syms, n_syms, &best_i);
            if (merged_id < 0 || best_i < 0) break;
            int j = syms[best_i].next;
            // merge best_i and j into best_i
            char merged[BPE_MAX_TOKEN * 2];
            snprintf(merged, sizeof(merged), "%s%s", syms[best_i].text, syms[j].text);
            strncpy(syms[best_i].text, merged, BPE_MAX_TOKEN - 1);
            syms[best_i].token_id = merged_id;
            // unlink j
            syms[best_i].next = syms[j].next;
            if (syms[j].next >= 0) syms[syms[j].next].prev = best_i;
        }
    } else {
        // Explicit merge rules: find lowest-rank pair, merge, repeat
        while (1) {
            int best_rank = INT32_MAX;
            int best_i = -1;
            int best_j = -1;
            for (int i = 0; i >= 0 && syms[i].next >= 0; ) {
                int j = syms[i].next;
                if (j < 0) break;
                int rank = bpe_merge_rank(tok, syms[i].text, syms[j].text);
                if (rank < best_rank) {
                    best_rank = rank;
                    best_i = i;
                    best_j = j;
                }
                i = j;
            }
            if (best_i < 0 || best_rank == INT32_MAX) break;
            // merge best_i and best_j
            char merged[BPE_MAX_TOKEN * 2];
            snprintf(merged, sizeof(merged), "%s%s", syms[best_i].text, syms[best_j].text);
            strncpy(syms[best_i].text, merged, BPE_MAX_TOKEN - 1);
            int tid = bpe_find_token(tok, merged);
            syms[best_i].token_id = (tid >= 0) ? tid : -1;
            syms[best_i].next = syms[best_j].next;
            if (syms[best_j].next >= 0) syms[syms[best_j].next].prev = best_i;
        }
    }

    // Collect tokens from linked list
    for (int i = 0; i >= 0 && n_out < max_tokens; ) {
        int tid = syms[i].token_id;
        if (tid >= 0) {
            out_tokens[n_out++] = tid;
        } else {
            // byte fallback
            const char *t = syms[i].text;
            while (*t && n_out < max_tokens) {
                int bt = tok->byte_token[(unsigned char)*t];
                if (bt >= 0) out_tokens[n_out++] = bt;
                t++;
            }
        }
        if (syms[i].next < 0) break;
        i = syms[i].next;
    }

    free(syms);
    return n_out;
}

// ----------------------------------------------------------------------------
// Free

static void bpe_tokenizer_free(BPETokenizer *tok) {
    for (int i = 0; i < tok->vocab_size; i++) free(tok->tokens[i]);
    free(tok->tokens);
    free(tok->scores);
    free(tok->token_type);
    if (tok->has_merges) {
        for (int i = 0; i < tok->n_merges; i++) {
            free(tok->merge_left[i]);
            free(tok->merge_right[i]);
        }
        free(tok->merge_left);
        free(tok->merge_right);
    }
}

#endif // BPE_TOKENIZER_H
