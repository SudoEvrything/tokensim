#include "tokensimd_sys.h"

#include "tokenizer.h"
#include "tokensimd_version.h"
#include "allocator.h"
#include "batch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tsimd_sys {
    tsimd_tokenizer_t tok;
    tsimd_token_sequence_t reuse;
    uint32_t *batch_ids;
    size_t batch_ids_cap;
    size_t *batch_offsets;
    size_t batch_offsets_cap;
};

static int parse_format(const char *format) {
    if (!format || !format[0]) return TSIMD_MODEL_TOKENIZER_JSON;
    if (strcmp(format, "json") == 0 || strcmp(format, "tokenizer_json") == 0)
        return TSIMD_MODEL_TOKENIZER_JSON;
    if (strcmp(format, "tiktoken") == 0) return TSIMD_MODEL_TIKTOKEN;
    if (strcmp(format, "spm") == 0 || strcmp(format, "sentencepiece") == 0)
        return TSIMD_MODEL_SENTENCEPIECE;
    if (strcmp(format, "gpt2") == 0) return TSIMD_MODEL_GPT2;
    return -1;
}

static int parse_scheme(const char *scheme, tsimd_pretok_scheme_t *out) {
    if (!scheme || !scheme[0] || strcmp(scheme, "auto") == 0) return 0;
    if (strcmp(scheme, "o200k") == 0) *out = TSIMD_PRETOK_O200K;
    else if (strcmp(scheme, "cl100k") == 0) *out = TSIMD_PRETOK_CL100K;
    else if (strcmp(scheme, "gpt2") == 0) *out = TSIMD_PRETOK_GPT2;
    else if (strcmp(scheme, "spm") == 0) *out = TSIMD_PRETOK_SPM;
    else if (strcmp(scheme, "gem4") == 0) *out = TSIMD_PRETOK_GEM4;
    else if (strcmp(scheme, "nemo") == 0) *out = TSIMD_PRETOK_NEMO;
    else if (strcmp(scheme, "kimi") == 0) *out = TSIMD_PRETOK_KIMI;
    else if (strcmp(scheme, "qwen25") == 0) *out = TSIMD_PRETOK_QWEN25;
    else if (strcmp(scheme, "deepseek") == 0) *out = TSIMD_PRETOK_DEEPSEEK;
    else if (strcmp(scheme, "gpt2d") == 0) *out = TSIMD_PRETOK_GPT2D;
    else return -1;
    return 0;
}

tsimd_sys *tsimd_sys_open(const char *model_path, const char *format,
                          const char *scheme, char *err, size_t err_cap) {
    if (err && err_cap) err[0] = 0;
    if (!model_path) {
        if (err && err_cap) snprintf(err, err_cap, "model path is NULL");
        return NULL;
    }
    int fmt = parse_format(format);
    if (fmt < 0) {
        if (err && err_cap) snprintf(err, err_cap, "unknown format '%s'",
                                     format);
        return NULL;
    }
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (parse_scheme(scheme, &sch) != 0) {
        if (err && err_cap) snprintf(err, err_cap, "unknown scheme '%s'",
                                     scheme);
        return NULL;
    }
    tsimd_sys *h = tsimd_xmalloc(sizeof(*h));
    if (!h) {
        if (err && err_cap) snprintf(err, err_cap, "out of memory");
        return NULL;
    }
    memset(h, 0, sizeof(*h));
    if (tsimd_tokenizer_load_model(&h->tok, (tsimd_model_format_t)fmt,
                                   model_path, NULL, &sch, NULL, 0) != 0) {
        if (err && err_cap) snprintf(err, err_cap, "cannot load '%s'",
                                     model_path);
        tsimd_xfree(h);
        return NULL;
    }
    return h;
}

void tsimd_sys_close(tsimd_sys *h) {
    if (!h) return;
    tsimd_token_sequence_free(&h->reuse);
    tsimd_xfree(h->batch_ids);
    tsimd_xfree(h->batch_offsets);
    tsimd_tokenizer_free(&h->tok);
    tsimd_xfree(h);
}

size_t tsimd_sys_vocab_size(const tsimd_sys *h) {
    return h ? (size_t)h->tok.vocab.vocab_size : 0;
}

const char *tsimd_sys_version(void) { return TSIMD_VERSION_STRING; }

static int sys_encode(tsimd_sys *h, const uint8_t *data, size_t len,
                      int threads, const tsimd_token_sequence_t **out) {
    if (!h) return -1;
    size_t bound = tsimd_tokenizer_encode_bound(&h->tok, len);
    if (bound > ((size_t)1 << 28)) {
        tsimd_token_sequence_t seq;
        int big_rc = (threads != 1)
                         ? tsimd_tokenizer_encode_parallel(&h->tok, data, len,
                                                           threads, &seq)
                         : tsimd_tokenizer_encode(&h->tok, data, len, &seq);
        if (big_rc != 0) return -1;
        if (h->reuse.cap < seq.count) {
            size_t cap = h->reuse.cap ? h->reuse.cap : 4096;
            while (cap < seq.count) cap *= 2;
            uint32_t *grown =
                tsimd_xrealloc(h->reuse.ids, cap * sizeof(uint32_t));
            if (!grown) {
                tsimd_token_sequence_free(&seq);
                return -1;
            }
            h->reuse.ids = grown;
            h->reuse.cap = cap;
        }
        if (seq.count)
            memcpy(h->reuse.ids, seq.ids, seq.count * sizeof(uint32_t));
        h->reuse.count = seq.count;
        tsimd_token_sequence_free(&seq);
        *out = &h->reuse;
        return 0;
    }
    if (h->reuse.cap < bound) {
        size_t cap = h->reuse.cap ? h->reuse.cap : 4096;
        while (cap < bound) cap *= 2;
        uint32_t *grown = tsimd_xrealloc(h->reuse.ids, cap * sizeof(uint32_t));
        if (!grown) return -1;
        h->reuse.ids = grown;
        h->reuse.cap = cap;
    }
    int rc;
    if (threads != 1) {
        rc = tsimd_tokenizer_encode_parallel_into(&h->tok, data, len, threads,
                                                  h->reuse.ids, h->reuse.cap,
                                                  &h->reuse.count);
    } else {
        rc = tsimd_tokenizer_encode_into_raw(&h->tok, data, len, h->reuse.ids,
                                             h->reuse.cap, &h->reuse.count);
    }
    if (rc != 0) {
        h->reuse.count = 0;
        return -1;
    }
    *out = &h->reuse;
    return 0;
}

int tsimd_sys_encode(tsimd_sys *h, const uint8_t *data, size_t len,
                     int threads, uint32_t **ids, size_t *n_ids) {
    const tsimd_token_sequence_t *seq = NULL;
    if (sys_encode(h, data, len, threads, &seq) != 0) return -1;
    uint32_t *buf = NULL;
    if (seq->count) {
        buf = tsimd_xmalloc(seq->count * sizeof(uint32_t));
        if (!buf) return -1;
        memcpy(buf, seq->ids, seq->count * sizeof(uint32_t));
    }
    if (ids) {
        *ids = buf;
    } else {
        tsimd_xfree(buf);
        buf = NULL;
    }
    if (n_ids) *n_ids = seq->count;
    return 0;
}

int tsimd_sys_encode_into(tsimd_sys *h, const uint8_t *data, size_t len,
                          int threads, uint32_t *out, size_t cap,
                          size_t *n_out) {
    if (!h || !out) return -1;
    if (cap >= tsimd_tokenizer_encode_bound(&h->tok, len)) {
        size_t n = 0;
        if (threads != 1) {
            if (tsimd_tokenizer_encode_parallel_into(&h->tok, data, len,
                                                     threads, out, cap,
                                                     &n) == 0) {
                if (n_out) *n_out = n;
                return 0;
            }
        } else if (tsimd_tokenizer_encode_into_raw(&h->tok, data, len, out, cap,
                                                   &n) == 0) {
            if (n_out) *n_out = n;
            return 0;
        }
    }
    const tsimd_token_sequence_t *seq = NULL;
    if (sys_encode(h, data, len, threads, &seq) != 0) return -1;
    if (seq->count > cap) return -1;
    memcpy(out, seq->ids, seq->count * sizeof(uint32_t));
    if (n_out) *n_out = seq->count;
    return 0;
}

int tsimd_sys_encode_bound(tsimd_sys *h, const uint8_t *data, size_t len,
                           int threads, size_t *cap_out) {
    if (!h) return -1;
    if (cap_out) *cap_out = tsimd_tokenizer_encode_bound(&h->tok, len);
    (void)data;
    (void)threads;
    return 0;
}

int tsimd_sys_decode_parallel(tsimd_sys *h, const uint32_t *ids, size_t n,
                              int skip_special, uint8_t *buf, size_t cap,
                              size_t *out_len, int threads) {
    if (!h) return -1;
    return tsimd_tokenizer_decode_parallel(&h->tok, ids, n, skip_special, buf,
                                           cap, out_len, threads);
}

int tsimd_sys_decode_into(tsimd_sys *h, const uint32_t *ids, size_t n,
                          int skip_special, uint8_t *buf, size_t cap,
                          size_t *out_len) {
    if (!h) return -1;
    return tsimd_tokenizer_decode_into(&h->tok, ids, n, skip_special, buf,
                                       cap, out_len);
}

int tsimd_sys_decode(tsimd_sys *h, const uint32_t *ids, size_t n,
                     int skip_special, uint8_t **bytes, size_t *n_bytes) {
    if (!h) return -1;
    tsimd_byte_sequence_t dec;
    if (tsimd_tokenizer_decode(&h->tok, ids, n, skip_special, &dec) != 0)
        return -1;
    if (bytes) {
        *bytes = dec.bytes;
    } else {
        tsimd_byte_sequence_free(&dec);
    }
    if (n_bytes) *n_bytes = dec.len;
    return 0;
}

static int span_array_grow(size_t **arr, size_t *cap, size_t need) {
    if (*cap >= need) return 0;
    size_t nc = *cap ? *cap : 64;
    while (nc < need) nc *= 2;
    size_t *grown = tsimd_xrealloc(*arr, nc * sizeof(size_t));
    if (!grown) return -1;
    *arr = grown;
    *cap = nc;
    return 0;
}

int tsimd_sys_encode_batch(tsimd_sys *h, const tsimd_input_span_t *docs,
                           size_t n_docs, int threads, uint32_t **ids,
                           size_t *n_ids, size_t **offsets) {
    if (!h || n_docs == 0) {
        if (n_ids) *n_ids = 0;
        if (offsets) *offsets = NULL;
        if (ids) *ids = NULL;
        return !h ? -1 : 0;
    }
    if (span_array_grow(&h->batch_offsets, &h->batch_offsets_cap,
                        n_docs + 1) != 0)
        return -1;
    tsimd_token_batch_t batch;
    memset(&batch, 0, sizeof batch);
    int rc;
    rc = tsimd_tokenizer_encode_batch(&h->tok, docs, n_docs, &batch, threads);
    if (rc != 0) return -1;
    if (batch.id_count > h->batch_ids_cap) {
        size_t cap = h->batch_ids_cap ? h->batch_ids_cap : 4096;
        while (cap < batch.id_count) cap *= 2;
        uint32_t *grown = tsimd_xrealloc(h->batch_ids,
                                         cap * sizeof(uint32_t));
        if (!grown) {
            tsimd_token_batch_free(&batch);
            return -1;
        }
        h->batch_ids = grown;
        h->batch_ids_cap = cap;
    }
    memcpy(h->batch_ids, batch.ids, batch.id_count * sizeof(uint32_t));
    memcpy(h->batch_offsets, batch.offsets, (n_docs + 1) * sizeof(size_t));
    if (ids) *ids = h->batch_ids;
    if (n_ids) *n_ids = batch.id_count;
    if (offsets) *offsets = h->batch_offsets;
    tsimd_token_batch_free(&batch);
    return 0;
}

int tsimd_sys_encode_own(tsimd_sys *h, const uint8_t *data, size_t len,
                        int threads) {
    if (!h) return -1;
    const tsimd_token_sequence_t *seq = NULL;
    return sys_encode(h, data, len, threads, &seq) == 0 ? 0 : -1;
}

const uint32_t *tsimd_sys_ids(const tsimd_sys *h, size_t *n) {
    if (!h) {
        if (n) *n = 0;
        return NULL;
    }
    if (n) *n = h->reuse.count;
    return h->reuse.ids;
}

void tsimd_sys_free(void *p) { tsimd_xfree(p); }
