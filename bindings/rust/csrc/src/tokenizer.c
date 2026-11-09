#define _GNU_SOURCE
#include "tokenizer.h"
#include "allocator.h"
#include "model_internal.h"
#include "unicode.h"
#include "pretokenizer/internal.h"
#include "nfc_internal.h"
#include <stdatomic.h>

#include <sched.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <string.h>
#if defined(__x86_64__) || defined(__i386__)
#include <emmintrin.h>
#endif

int tsimd_model_load_gpt2(tsimd_vocab_t *v, const char *encoder_path,
                         const char *merges_path);
void tsimd_serve_state_free(tsimd_tokenizer_t *t);

int tsimd_model_load_tokenizer_json(tsimd_vocab_t *v, const char *path,
                                   tsimd_pretok_scheme_t *scheme,
                                   tsimd_added_token_t **added_out,
                                   size_t *n_added_out,
                                   char **added_blob_out,
                                   int *norm_out);
int tsimd_model_load_spm(tsimd_tokenizer_t *t, const char *path,
                         const char *fp);

static void free_added(tsimd_tokenizer_t *t) {
    if (t->added_owned) free(t->added);
    free(t->added_text_blob);
    free(t->added_next);
    free(t->added_head2);
    t->added = NULL;
    t->added_count = 0;
    t->added_next = NULL;
    t->added_head2 = NULL;
    t->added_owned = 0;
    t->added_text_blob = NULL;
}

static int rebuild_added_index(tsimd_tokenizer_t *t) {
    for (int b = 0; b < 256; b++) t->added_head[b] = 0xFFFF;
    free(t->added_next);
    free(t->added_head2);
    t->added_next = NULL;
    t->added_head2 = NULL;
    if (t->added_count == 0 || t->added_count > 65535) return 0;
    uint16_t *next = tsimd_xmalloc(t->added_count * sizeof(uint16_t));
    uint16_t *head2 =
        tsimd_xmalloc((size_t)65536 * sizeof(uint16_t));
    if (!next || !head2) {
        free(next);
        free(head2);
        return -1;
    }
    for (int k = 0; k < 65536; k++) head2[k] = 0xFFFF;

    for (size_t i = t->added_count; i-- > 0;) {
        const tsimd_added_token_t *at = &t->added[i];
        if (at->length == 0) continue;
        uint8_t b0 = (uint8_t)at->text[0];
        if (at->length >= 2) {
            size_t key = ((size_t)b0 << 8) | (uint8_t)at->text[1];
            next[i] = head2[key];
            head2[key] = (uint16_t)i;
        } else {
            next[i] = t->added_head[b0];
            t->added_head[b0] = (uint16_t)i;
        }
    }
    t->added_next = next;
    t->added_head2 = head2;
    return 0;
}

static uint32_t auto_id_floor(const tsimd_tokenizer_t *t) {
    uint32_t floor = t->vocab.vocab_size;
    for (size_t i = 0; i < t->added_count; i++)
        if (t->added[i].id != 0 && t->added[i].id + 1 > floor)
            floor = t->added[i].id + 1;
    return floor;
}

static int append_added(tsimd_tokenizer_t *t, const tsimd_added_token_t *extra,
                        size_t n_extra, int auto_ids) {
    if (n_extra == 0) return 0;
    size_t base = t->added_count;
    tsimd_added_token_t *grown =
        tsimd_xrealloc(t->added, (base + n_extra) * sizeof(tsimd_added_token_t));
    if (!grown) return -1;
    t->added = grown;
    t->added_owned = 1;
    uint32_t next_auto = auto_id_floor(t);
    for (size_t i = 0; i < n_extra; i++) {
        t->added[base + i] = extra[i];
        t->added[base + i].length = (uint32_t)strlen(extra[i].text);
        if (auto_ids && extra[i].id == 0)
            t->added[base + i].id = next_auto++;
    }
    t->added_count = base + n_extra;
    return rebuild_added_index(t);
}

static int tiktoken_scheme_ok(tsimd_pretok_scheme_t sch, size_t vocab_size) {
    if (sch != TSIMD_PRETOK_GPT2 && sch != TSIMD_PRETOK_GPT2D) return 1;
    if (vocab_size < 60000) return 1;
    fprintf(stderr,
            "tsimd: scheme gpt2 requested for a %zu-token tiktoken vocab "
            "(cl100k/o200k size signature); the gpt2 regex does not match "
            "this vocab — pass cl100k or o200k explicitly\n",
            vocab_size);
    return 0;
}

static void decode_tables_build(tsimd_tokenizer_t *t) {
    size_t max_id = 0;
    for (size_t s = 0; s < t->added_count; s++)
        if (t->added[s].id > max_id) max_id = t->added[s].id;
    if (t->added_count) {
        t->added_by_id = calloc(max_id + 1, sizeof(tsimd_added_token_t *));
        if (t->added_by_id) {
            t->added_by_id_n = max_id + 1;
            for (size_t s = 0; s < t->added_count; s++)
                t->added_by_id[t->added[s].id] = &t->added[s];
        }
    }
    if (t->scheme == TSIMD_PRETOK_SPM && t->vocab.spm_control_n) {
        uint32_t mx = 0;
        for (uint32_t k = 0; k < t->vocab.spm_control_n; k++)
            if (t->vocab.spm_control_ids[k] > mx)
                mx = t->vocab.spm_control_ids[k];
        size_t bytes = (size_t)mx / 8 + 1;
        t->spm_ctl_flags = calloc(bytes, 1);
        if (t->spm_ctl_flags) {
            t->spm_ctl_n = bytes;
            for (uint32_t k = 0; k < t->vocab.spm_control_n; k++) {
                uint32_t id = t->vocab.spm_control_ids[k];
                t->spm_ctl_flags[id >> 3] |= (uint8_t)(1u << (id & 7));
            }
        }
    }
}

int tsimd_tokenizer_load_model(tsimd_tokenizer_t *t, tsimd_model_format_t fmt,
                              const char *path1, const char *path2,
                              tsimd_pretok_scheme_t *scheme,
                              const tsimd_added_token_t *extra,
                              size_t n_extra) {

    tsimd_tokenizer_free(t);
    memset(t, 0, sizeof(*t));
    t->magic_ = TSIMD_TOKENIZER_MAGIC;
    tsimd_pretok_scheme_t det = scheme ? *scheme : TSIMD_PRETOK_GPT2;
    int rc = -1;
    switch (fmt) {
        case TSIMD_MODEL_TIKTOKEN:
            if (tsimd_vocab_load_file(path1, &t->vocab) != 0) {
                fprintf(stderr, "tsimd: cannot load tiktoken vocab %s\n", path1);
                goto out;
            }
            if (!tiktoken_scheme_ok(det, t->vocab.vocab_size)) {
                rc = -1;
                goto out;
            }
            t->scheme = det;
            rc = 0;
            break;
        case TSIMD_MODEL_GPT2:
            if (tsimd_model_load_gpt2(&t->vocab, path1, path2) != 0) goto out;
            t->scheme = det;
            rc = 0;
            break;
        case TSIMD_MODEL_TOKENIZER_JSON: {
            tsimd_added_token_t *file_added = NULL;
            size_t n_file = 0;
            char *file_blob = NULL;

            if (tsimd_model_load_tokenizer_json(&t->vocab, path1, &det,
                                               &file_added, &n_file,
                                               &file_blob, &t->norm) != 0)
                goto out;
            t->added = file_added;
            t->added_count = n_file;
            t->added_owned = 1;
            t->added_text_blob = file_blob;
            if (rebuild_added_index(t) != 0) {
                rc = -1;
                goto out;
            }
            t->scheme = det;
            rc = 0;
            break;
        }
        case TSIMD_MODEL_SENTENCEPIECE:
            if (tsimd_model_load_spm(t, path1, NULL) != 0) {
                fprintf(stderr, "tsimd: cannot load sentencepiece model %s\n",
                        path1);
                goto out;
            }
            det = TSIMD_PRETOK_SPM;

            t->scheme = det;
            rc = 0;
            break;
        }
    if (rc == 0) {
        size_t entries = 1u << 22;
        int bits = tsimd_pcache_bits_env();
        if (bits > 0) entries = (size_t)1 << bits;

        enum { TSIMD_SPM_WC_SLOTS = 64 };
        if (det == TSIMD_PRETOK_GEM4 || det == TSIMD_PRETOK_SPM ||
            det == TSIMD_PRETOK_KIMI) {
            t->spm_caches = (tsimd_pretok_cache_t *)calloc(
                TSIMD_SPM_WC_SLOTS, sizeof(*t->spm_caches));
            if (t->spm_caches) t->spm_caches_n = TSIMD_SPM_WC_SLOTS;
        }

        if (append_added(t, extra, n_extra, 1) != 0 ||
            tsimd_pretok_cache_init(&t->cache, entries) != 0 ||

            tsimd_pcache_seed_vocab(&t->cache, &t->vocab) != 0) {
            rc = -1;
        }
    }
out:
    if (scheme) *scheme = det;
    if (rc == 0) decode_tables_build(t);
    if (rc != 0) tsimd_tokenizer_free(t);
    return rc;
}

int tsimd_tokenizer_load(tsimd_tokenizer_t *t, const char *vocab_path,
                        tsimd_pretok_scheme_t scheme,
                        const tsimd_added_token_t *specials,
                        size_t n_specials) {
    tsimd_pretok_scheme_t s = scheme;
    return tsimd_tokenizer_load_model(t, TSIMD_MODEL_TIKTOKEN, vocab_path, NULL,
                                     &s, specials, n_specials);
}

static void spm_caches_free(tsimd_tokenizer_t *t) {
    for (size_t k = 0; k < t->spm_caches_n; k++)
        if (t->spm_caches[k].entries)
            tsimd_pretok_cache_free(&t->spm_caches[k]);
    free(t->spm_caches);
    t->spm_caches = NULL;
    t->spm_caches_n = 0;
}

tsimd_pretok_cache_t *tsimd_spm_worker_cache(tsimd_tokenizer_t *t, size_t idx,
                                            size_t chunk_len) {
    enum { TSIMD_SPM_WC_SLOTS = 64 };
    if (idx >= TSIMD_SPM_WC_SLOTS) return &t->cache;
    if (!t->spm_caches) return &t->cache;
    size_t want = 16;
    while (want < 20 && ((size_t)1 << (want - 1)) < chunk_len / 16) want++;
    static long wc_bits = -1;
    if (wc_bits < 0) {
        const char *e = getenv("TSIMD_SPM_WC_BITS");
        wc_bits = e && e[0] ? strtol(e, NULL, 10) : 0;
    }
    if (wc_bits >= 12 && wc_bits <= 22) want = (size_t)wc_bits;
    tsimd_pretok_cache_t *c = &t->spm_caches[idx];
    if (c->entries) {
        size_t have = 0;
        while (((size_t)1 << have) < c->mask + 1) have++;
        if (have >= want) return c;
        tsimd_pretok_cache_free(c);
        memset(c, 0, sizeof(*c));
    }
    if (!c->entries) {
        if (tsimd_pretok_cache_init(c, (size_t)1 << want) != 0) {
            memset(c, 0, sizeof(*c));
            return &t->cache;
        }
        if (tsimd_pcache_seed_vocab(c, &t->vocab) != 0) {
            tsimd_pretok_cache_free(c);
            memset(c, 0, sizeof(*c));
            return &t->cache;
        }
    }
    return c;
}

void tsimd_tokenizer_free(tsimd_tokenizer_t *t) {
    if (t->magic_ != TSIMD_TOKENIZER_MAGIC) {
        memset(t, 0, sizeof(*t));
        return;
    }
    tsimd_tokenizer_release_pool(t);
    tsimd_serve_state_free(t);
    tsimd_vocab_free(&t->vocab);
    spm_caches_free(t);
    tsimd_pretok_cache_free(&t->cache);
    tsimd_pretok_cache_free(&t->memo_cache);
    free(t->added_by_id);
    free(t->spm_ctl_flags);
    free_added(t);
    memset(t, 0, sizeof(*t));
}

static int emit_out(tsimd_bpe_workspace_t *ws, uint32_t id) {
    if (ws->out_count == ws->out_capacity) {
        if (tsimd_ws_out_ensure(ws, ws->out_count + 1) != 0) return -1;
    }
    ws->out[ws->out_count++] = id;
    return 0;
}

size_t tsimd_tokenizer_next_added(const tsimd_tokenizer_t *t,
                                 const uint8_t *data, size_t len,
                                 size_t pos, size_t *added_index) {
    size_t best = (size_t)-1;
    size_t best_len = 0;
    for (size_t s = 0; s < t->added_count; s++) {
        const tsimd_added_token_t *at = &t->added[s];
        size_t nlen = at->length;
        if (nlen == 0 || nlen > len) continue;
        const uint8_t *last = data + len - nlen;
        const uint8_t *h = data + pos;
        if (h > last) continue;
        while (h <= last) {
            const uint8_t *r = memchr(h, at->text[0],
                                      (size_t)(last - h) + 1);
            if (!r) break;
            if (memcmp(r, at->text, nlen) == 0) {
                size_t i = (size_t)(r - data);
                if (i < best || (i == best && nlen > best_len)) {
                    best = i;
                    best_len = nlen;
                    *added_index = s;
                }
                break;
            }
            h = r + 1;
        }
    }
    return best;
}

int tsimd_serve_segment(tsimd_tokenizer_t *t, tsimd_bpe_workspace_t *out_ws,
                        const uint8_t *data, size_t len);
int tsimd_serve_batched_segment(tsimd_tokenizer_t *t,
                                tsimd_bpe_workspace_t *out_ws,
                                const uint8_t *data, size_t len);
void tsimd_serve_state_free(tsimd_tokenizer_t *t);
int tsimd_batched_serve_disabled(void);

static int kimi_masks_off(void) {
#if defined(__x86_64__) || defined(__i386__)
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_KIMI_MASKS");
        atomic_store(&state, (e && e[0] == '1') ? 1 : 2);
        v = atomic_load(&state);
    }
    return v != 1;
#else
    return 1;
#endif
}
void tsimd_serve_env_refresh(void);
int tsimd_serve_env_per_call(void);
int tsimd_serve_qwen25_scalar_cached(void);
int tsimd_gpt2d_span_on(void);
typedef struct {
    tsimd_tokenizer_t *t;
    tsimd_bpe_workspace_t *ws;
    const uint8_t *data;
    const uint8_t *end;

} fuse_ctx_t;

static int fused_span(void *ud, size_t s, size_t e) {
    fuse_ctx_t *f = ud;
    const uint8_t *p = f->data + s;
    size_t n = e - s;
    if (n > UINT32_MAX) {
        fprintf(stderr, "tsimd: single pretoken exceeds UINT32_MAX bytes\n");
        return -1;
    }
    uint32_t n_ids = 0;
    uint32_t ids[TSIMD_PCACHE_MAX_IDS];
    uint32_t *cached = tsimd_pcache_lookup_into_bounded(
        &f->t->cache, p, (uint32_t)n, f->end, ids, &n_ids)
                           ? ids
                           : NULL;
    if (cached) {
        if (n_ids > 2 && f->ws->out_count + n_ids <= f->ws->out_capacity) {
            memcpy(f->ws->out + f->ws->out_count, cached,
                   n_ids * sizeof(uint32_t));
            f->ws->out_count += n_ids;
            return 0;
        }
        for (uint32_t k = 0; k < n_ids; k++)
            if (emit_out(f->ws, cached[k]) != 0) return -1;
        return 0;
    }
    size_t base = f->ws->out_count;
    if (tsimd_bpe_encode(&f->t->vocab, f->ws, p, (uint32_t)n) != 0) return -1;
    tsimd_pcache_insert(&f->t->cache, p, (uint32_t)n, f->ws->out + base,
                        (uint32_t)(f->ws->out_count - base));
    return 0;
}

void tsimd_pcache_upgrade_for(tsimd_tokenizer_t *t, size_t len);
void tsimd_pcache_upgrade_for(tsimd_tokenizer_t *t, size_t len) {
    if (tsimd_pcache_bits_env() > 0) return;
    t->cache_bytes += len;
    size_t lo = (size_t)256 << 20;
    {
        static atomic_int thr_state;
        int v = atomic_load(&thr_state);
        if (v == 0) {
            long mb = 256;
            const char *e = getenv("TSIMD_PCACHE_UPGRADE_MB");
            if (e && e[0]) {
                long p = strtol(e, NULL, 10);
                if (p >= 0 && p <= 65536) mb = p;
            }
            v = (int)(mb > 0 ? mb : 1);
            atomic_store(&thr_state, v);
        }
        lo = (size_t)v << 20;
    }
    size_t want = 0;
    if (t->cache_bytes >= 2 * lo) want = 1u << 22;
    else if (t->cache_bytes >= lo) want = 1u << 21;
    if (want == 0 || t->cache.mask + 1 >= want) return;
    tsimd_pretok_cache_free(&t->cache);
    if (tsimd_pretok_cache_init(&t->cache, want) != 0) return;
    tsimd_pcache_seed_vocab(&t->cache, &t->vocab);
}

static int tsimd_spm_norm_off(void) {
    static atomic_int state = 0;
    int v = atomic_load(&state);
    if (v == 0) {
        const char *e = getenv("TSIMD_SPM_NORM_SEQ");
        atomic_store(&state, (e && e[0] == '1') ? 2 : 1);
        v = atomic_load(&state);
    }
    return v == 2;
}

static size_t seam_block_target(void) {
    static size_t cached = 0;
    static _Atomic int done = 0;
    if (atomic_load(&done)) return cached;
    size_t v;
    const char *e = getenv("TSIMD_NO_SEAM_BLOCK");
    if (e && e[0] == '1') {
        v = 0;
    } else {
        v = (size_t)(1u << 14);
        const char *k = getenv("TSIMD_SEAM_BLOCK_KB");
        if (k && k[0]) {
            long kb = strtol(k, NULL, 10);
            if (kb > 0) v = (size_t)kb << 10;
        }
    }
    cached = v;
    atomic_store(&done, 1);
    return cached;
}

static uint8_t *tsimd_metaspace_scratch(size_t need) {
    static _Thread_local uint8_t *buf;
    static _Thread_local size_t cap;
    if (cap < need) {
        tsimd_xfree(buf);
        buf = tsimd_xmalloc(need);
        cap = buf ? need : 0;
    }
    return buf;
}

static uint64_t tsimd_dbg_tsc(void) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#else
    return 0;
#endif
}

static int spm_encode_par_cached(tsimd_tokenizer_t *t,
                                tsimd_bpe_workspace_t *ws,
                                const uint8_t *norm, size_t nn, int threads) {
    enum { TSIMD_SPM_WC_MAX = 64 };
    if (threads < 1) threads = 1;
    if (threads > TSIMD_SPM_WC_MAX) threads = TSIMD_SPM_WC_MAX;
    tsimd_pretok_cache_t *wc[TSIMD_SPM_WC_MAX];
    size_t chunk = nn / (size_t)threads + 1;
    for (int i = 0; i < threads; i++)
        wc[i] = tsimd_spm_worker_cache(t, (size_t)i, chunk);
    uint64_t dbg0 = getenv("TSIMD_POOL_DBG") ? tsimd_dbg_tsc() : 0;
    int rc = tsimd_bpe_encode_spm_blocked_par_cached(
        &t->vocab, &t->cache, wc, ws, norm, (uint32_t)nn,
        seam_block_target(), threads);
    if (getenv("TSIMD_WC_DBG"))
        for (int i = 0; i < threads; i++)
            fprintf(stderr, "wc[%d] ptr=%s count=%zu\n", i,
                    wc[i] == &t->cache ? "SHARED" : "own",
                    wc[i]->count);
    if (dbg0)
        fprintf(stderr, "tsimd spm par_dbg: par %.2f Mcyc\n",
                (double)(tsimd_dbg_tsc() - dbg0) / 1e6);
    return rc;
}

typedef struct {
    const uint8_t *in;
    uint8_t *out;
    size_t in_len;
    size_t out_off;
    int dummy;
    size_t actual;
} meta_job_t;

static void *meta_worker(void *arg) {
    meta_job_t *mj = arg;
    size_t mcap = mj->in_len * 3 + 16;
    uint8_t *buf = tsimd_xmalloc(mcap);
    if (!buf) {
        mj->actual = (size_t)-1;
        return NULL;
    }
    size_t nn = tsimd_metaspace_ex(mj->in, mj->in_len, buf, mcap, mj->dummy);
    mj->out = buf;
    mj->actual = nn;
    return NULL;
}

static int tsimd_metaspace_par(const uint8_t *data, size_t len,
                               uint8_t *out, size_t cap, int dummy_prefix,
                               int threads, size_t chunk, size_t *out_len) {
    if (threads <= 1 || len < 4 * chunk) {
        size_t nn = tsimd_metaspace_ex(data, len, out, cap, dummy_prefix);
        if (nn == (size_t)-1) return -1;
        *out_len = nn;
        return 0;
    }
    enum { TSIMD_META_MAXC = 64 };
    meta_job_t mj[TSIMD_META_MAXC];
    pthread_t th[TSIMD_META_MAXC];
    for (int i = 0; i < TSIMD_META_MAXC; i++) th[i] = 0;
    size_t n = 0, pos = 0;
    while (pos < len && n < TSIMD_META_MAXC) {
        size_t end = pos + chunk;
        if (end < len) {
            const uint8_t *sp = (const uint8_t *)memchr(data + end, ' ',
                                                        len - end);
            end = sp ? (size_t)(sp - data) : len;
        } else {
            end = len;
        }
        if (end <= pos) end = pos + 1;
        mj[n].in = data + pos;
        mj[n].in_len = end - pos;
        mj[n].out = NULL;
        mj[n].out_off = 0;
        mj[n].actual = 0;
        mj[n].dummy = (n == 0) ? dummy_prefix : 0;
        n++;
        pos = end;
    }
    for (size_t i = 1; i < n; i++)
        if (pthread_create(&th[i], NULL, meta_worker, &mj[i]) != 0)
            meta_worker(&mj[i]);
    meta_worker(&mj[0]);
    for (size_t i = 1; i < n; i++)
        if (th[i]) pthread_join(th[i], NULL);
    size_t cur = 0;
    int rc = 0;
    for (size_t i = 0; i < n; i++)
        if (mj[i].actual == (size_t)-1) rc = -1;
    for (size_t i = 0; i < n && rc == 0; i++) {
        if (cur + mj[i].actual > cap || cur + mj[i].actual < cur) {
            rc = -1;
            break;
        }
        memcpy(out + cur, mj[i].out, mj[i].actual);
        cur += mj[i].actual;
    }
    for (size_t i = 0; i < n; i++) tsimd_xfree(mj[i].out);
    *out_len = cur;
    return rc;
}

static int tsimd_tokenizer_encode_whole_chain_par_impl(tsimd_tokenizer_t *t,
                                                       const uint8_t *data,
                                                       size_t len, int threads,
                                                       uint32_t *buf,
                                                       size_t cap,
                                                       uint32_t **out_ids,
                                                       size_t *out_count);

static uint8_t *spm_norm_alloc(tsimd_tokenizer_t *t, const uint8_t *p,
                               size_t n, size_t *out_len);

int tsimd_tokenizer_encode_whole_chain_par(tsimd_tokenizer_t *t,
                                           const uint8_t *data, size_t len,
                                           int threads,
                                           tsimd_token_sequence_t *out) {
    if (t->scheme == TSIMD_PRETOK_SPM && t->spm_kind) {
        if (threads <= 1 || len < (2u << 20))
            return tsimd_tokenizer_encode(t, data, len, out);
        tsimd_bpe_workspace_t ws;
        if (tsimd_bpe_workspace_init(&ws, t->vocab.max_token_len) != 0)
            return -1;
        size_t nn = 0;
        int rc;
        if (t->added_count == 0) {
            uint8_t *norm = spm_norm_alloc(t, data, len, &nn);
            rc = norm ? tsimd_unigram_encode_par(&t->vocab, &ws, norm,
                                                 (uint32_t)nn, threads)
                      : -1;
        } else {
            tsimd_added_seam_list_t seams;
            memset(&seams, 0, sizeof seams);
            rc = tsimd_tokenizer_added_seams(t, data, len, &seams);
            if (rc == 0) {
                size_t seg = 0;
                for (size_t i = 0; i < seams.count && rc == 0; i++) {
                    const tsimd_added_seam_t *sm = &seams.items[i];
                    if (sm->off > seg) {
                        uint8_t *norm =
                            spm_norm_alloc(t, data + seg, sm->off - seg, &nn);
                        rc = norm ? tsimd_unigram_encode_par(
                                        &t->vocab, &ws, norm,
                                        (uint32_t)nn, threads)
                                  : -1;
                    }
                    if (rc == 0) rc = emit_out(&ws, t->added[sm->idx].id);
                    seg = sm->off + sm->len;
                }
                if (rc == 0 && len > seg) {
                    uint8_t *norm = spm_norm_alloc(t, data + seg, len - seg,
                                                   &nn);
                    rc = norm ? tsimd_unigram_encode_par(&t->vocab, &ws, norm,
                                                         (uint32_t)nn,
                                                         threads)
                              : -1;
                }
            }
            tsimd_added_seam_list_free(&seams);
        }
        if (rc == 0) {
            out->ids = NULL;
            out->count = ws.out_count;
            out->cap = ws.out_count;
            if (ws.out_count) {
                out->ids = (uint32_t *)malloc(
                    ws.out_count * sizeof(uint32_t));
                if (out->ids)
                    memcpy(out->ids, ws.out,
                           ws.out_count * sizeof(uint32_t));
                else
                    rc = -1;
            }
        }
        tsimd_bpe_workspace_free(&ws);
        return rc;
    }
    uint32_t *ids = NULL;
    size_t count = 0;
    int rc = tsimd_tokenizer_encode_whole_chain_par_impl(
        t, data, len, threads, NULL, 0, &ids, &count);
    if (rc != 0) return rc;
    out->ids = ids;
    out->count = count;
    out->cap = count;
    return 0;
}

int tsimd_tokenizer_encode_whole_chain_par_into(tsimd_tokenizer_t *t,
                                                const uint8_t *data, size_t len,
                                                int threads, uint32_t *buf,
                                                size_t cap, size_t *count) {
    if (!buf || !count) return -1;
    if (cap < tsimd_tokenizer_encode_bound(t, len)) return -1;
    *count = 0;
    if (t->scheme == TSIMD_PRETOK_SPM && t->spm_kind)
        return tsimd_tokenizer_encode_into_raw(t, data, len, buf, cap, count);
    uint32_t *ids = NULL;
    size_t n = 0;
    int rc = tsimd_tokenizer_encode_whole_chain_par_impl(t, data, len, threads,
                                                         buf, cap, &ids, &n);
    if (rc != 0) return rc;
    *count = n;
    return 0;
}

static uint8_t *tsimd_spm_norm_scratch(size_t need);

static int whole_chain_span(tsimd_tokenizer_t *t, const uint8_t *data,
                            size_t len, int threads, int dummy,
                            tsimd_bpe_workspace_t *ws) {
    if (len == 0) return 0;
    if (t->scheme == TSIMD_PRETOK_SPM && !t->spm_kind) {
        size_t cap = len * 3 + 16;
        uint8_t *norm = tsimd_spm_norm_scratch(cap);
        if (!norm) return -1;
        size_t nn = tsimd_spm_normalize(data, len, norm, cap, dummy,
                                        t->spm_remove_extra);
        if (nn == (size_t)-1) return -1;
        return spm_encode_par_cached(t, ws, norm, nn, threads);
    }
    size_t mcap = len * 3 + 16;
    uint8_t *norm = tsimd_metaspace_scratch(mcap);
    if (!norm) return -1;
    size_t mchunk = len / (size_t)(threads > 0 ? threads * 4 : 4) + 1;
    if (mchunk < 64 * 1024) mchunk = 64 * 1024;
    size_t nn = 0;
    int mrc = tsimd_metaspace_par(data, len, norm, mcap, dummy, threads,
                                  mchunk, &nn);
    if (mrc != 0) return -1;
    return spm_encode_par_cached(t, ws, norm, nn, threads);
}

static int tsimd_tokenizer_encode_whole_chain_par_impl(tsimd_tokenizer_t *t,
                                                       const uint8_t *data,
                                                       size_t len, int threads,
                                                       uint32_t *buf,
                                                       size_t cap,
                                                       uint32_t **out_ids,
                                                       size_t *out_count) {
    tsimd_serve_env_refresh();
    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, t->vocab.max_token_len) != 0) return -1;
    int dummy0 = t->scheme == TSIMD_PRETOK_GEM4 ? 0 : t->spm_dummy_prefix;
    int rc;
    if (t->added_count == 0) {
        rc = whole_chain_span(t, data, len, threads, dummy0, &ws);
    } else {
        tsimd_added_seam_list_t seams;
        memset(&seams, 0, sizeof seams);
        rc = tsimd_tokenizer_added_seams(t, data, len, &seams);
        if (rc == 0) {
            size_t seg = 0;
            for (size_t i = 0; i < seams.count && rc == 0; i++) {
                const tsimd_added_seam_t *sm = &seams.items[i];
                if (sm->off > seg)
                    rc = whole_chain_span(t, data + seg, sm->off - seg,
                                          threads, dummy0, &ws);
                if (rc == 0) rc = emit_out(&ws, t->added[sm->idx].id);
                seg = sm->off + sm->len;
            }
            if (rc == 0 && len > seg)
                rc = whole_chain_span(t, data + seg, len - seg, threads,
                                      dummy0, &ws);
        }
        tsimd_added_seam_list_free(&seams);
    }
    if (rc != 0) { tsimd_bpe_workspace_free(&ws); return -1; }
    if (buf) {
        if (ws.out_count > cap) { tsimd_bpe_workspace_free(&ws); return -1; }
        if (ws.out_count)
            memcpy(buf, ws.out, ws.out_count * sizeof(uint32_t));
        *out_ids = buf;
    } else {
        *out_ids = NULL;
        if (ws.out_count) {
            *out_ids = tsimd_xmalloc(ws.out_count * sizeof(uint32_t));
            if (!*out_ids) { tsimd_bpe_workspace_free(&ws); return -1; }
            memcpy(*out_ids, ws.out, ws.out_count * sizeof(uint32_t));
        }
    }
    *out_count = ws.out_count;
    tsimd_bpe_workspace_free(&ws);
    return 0;
}

static uint8_t *tsimd_spm_norm_scratch(size_t need) {
    static _Thread_local uint8_t *buf;
    static _Thread_local size_t cap;
    if (cap < need) {
        tsimd_xfree(buf);
        buf = tsimd_xmalloc(need);
        cap = buf ? need : 0;
    }
    return buf;
}

static uint8_t *spm_norm_alloc(tsimd_tokenizer_t *t, const uint8_t *p,
                               size_t n, size_t *out_len) {
    size_t cap = n * 3 + 16;
    for (;;) {
        uint8_t *buf = tsimd_spm_norm_scratch(cap);
        if (!buf) return NULL;
        size_t nn = t->spm_kind
                        ? tsimd_spm_normalize(p, n, buf, cap,
                                              t->spm_dummy_prefix,
                                              t->spm_remove_extra)
                        : tsimd_metaspace_ex(p, n, buf, cap,
                                             t->spm_dummy_prefix);
        if (nn != (size_t)-1) {
            *out_len = nn;
            return buf;
        }
        if (cap > ((size_t)1 << 30)) return NULL;
        cap <<= 1;
    }
}

static int encode_spm_segment_c(tsimd_tokenizer_t *t, const uint8_t *data,
                                size_t len, tsimd_bpe_workspace_t *ws,
                                tsimd_pretok_cache_t *c,
                                tsimd_pretok_cache_t *shared) {
    size_t nn = 0;
    uint8_t *norm = spm_norm_alloc(t, data, len, &nn);
    if (!norm) return -1;
    if (getenv("TSIMD_ESSC_DBG"))
        fprintf(stderr, "ESSC len=%zu nn=%zu kind=%d scheme=%d\n", len, nn,
                t->spm_kind, (int)t->scheme);
    return t->spm_kind
               ? tsimd_unigram_encode(&t->vocab, ws, norm, (uint32_t)nn)
               : tsimd_bpe_encode_spm_blocked_cached(
                     &t->vocab, c, shared, ws, norm, (uint32_t)nn,
                     tsimd_spm_word_target(&t->vocab, seam_block_target()));
}

static int encode_spm_segment(tsimd_tokenizer_t *t, const uint8_t *data,
                              size_t len, tsimd_bpe_workspace_t *ws) {
    return encode_spm_segment_c(t, data, len, ws, &t->cache, NULL);
}

static int uni_memo_ws(const uint8_t *d, size_t len, size_t i,
                       size_t *adv_out) {
    uint8_t b = d[i];
    if (b < 0x80) {
        if (b == ' ' || b == '\t' || b == '\n' || b == '\r' ||
            b == 0x0b || b == 0x0c) {
            *adv_out = 1;
            return 1;
        }
        return 0;
    }
    if (b < 0xC0) return 0;

    uint32_t cp;
    tsimd_char_class_t cls;
    size_t adv = tsimd_utf8_next(d + i, len - i, &cp, &cls);
    if (adv == 0 || cls != TSIMD_CLASS_SPACE) return 0;
    *adv_out = adv;
    return 1;
}

static size_t uni_memo_next_split(const tsimd_tokenizer_t *t,
                                  const uint8_t *d, size_t len,
                                  size_t from) {
    size_t i = from;
    while (i < len) {
        size_t adv = 0;
        if (!uni_memo_ws(d, len, i, &adv)) {
            i++;
            continue;
        }
        if (i > 0) {
            size_t ladv = 0;
            if (uni_memo_ws(d, len, i - 1, &ladv)) {
                i += adv;
                continue;
            }

            size_t s = i - 1;
            while (s > 0 && d[s - 1] >= 0x80 && (d[s] & 0xC0) == 0x80)
                s--;
            size_t a = tsimd_utf8_decode(d + s, len - s, &(uint32_t){0});
            if (a == 0 ||
                tsimd_vocab_find(&t->vocab, d + s, (uint32_t)a) ==
                    UINT32_MAX) {
                i += adv;
                continue;
            }
        }
        return i;
    }
    return len;
}

static int uni_memo_disabled(void) {

    const char *off = getenv("TSIMD_DISABLE_UNIGRAM_MEMO");
    if (off && off[0] == '1') return 1;
    const char *on = getenv("TSIMD_ENABLE_UNIGRAM_MEMO");
    return !(on && on[0] == '1');
}

void tsimd_memo_size_for(tsimd_tokenizer_t *t, size_t total_len) {
    if (t->scheme != TSIMD_PRETOK_SPM || !t->spm_kind) return;
    if (uni_memo_disabled()) return;
    if (t->memo_cache.entries) return;
    size_t entries = 1u << 22;
    if (total_len < (8u << 20)) entries = 1u << 20;
    int bits = tsimd_pcache_bits_env();
    if (bits > 0) entries = (size_t)1 << bits;
    (void)tsimd_pretok_cache_init(&t->memo_cache, entries);
}

static int encode_unigram_memo(tsimd_tokenizer_t *t, const uint8_t *data,
                               size_t len, tsimd_bpe_workspace_t *ws) {

    int use_cache = !uni_memo_disabled();
    if (use_cache && !t->memo_cache.entries) {

        size_t entries = 1u << 22;
        int bits = tsimd_pcache_bits_env();
        if (bits > 0) entries = (size_t)1 << bits;
        if (tsimd_pretok_cache_init(&t->memo_cache, entries) != 0)
            return -1;
    }
    tsimd_pretok_cache_t *c = &t->memo_cache;
    size_t pos = 0;

    float acc = 0.0f;
    while (pos < len) {
        float uni_acc_in = acc;
        size_t end = uni_memo_next_split(t, data, len, pos);

        if (end <= pos) {
            size_t wadv = 0;
            end = uni_memo_ws(data, len, pos, &wadv) ? pos + wadv
                                                     : pos + 1;
        }
        const uint8_t *p = data + pos;
        size_t n = end - pos;
        uint32_t ids[TSIMD_PCACHE_MAX_IDS];
        uint32_t n_ids = 0;
        const uint32_t *hit_ids = ids;
        int hit = 0;
        if (use_cache && n <= TSIMD_PCACHE_MAX_SPAN) {
            hit = tsimd_pcache_lookup_into_bounded(
                c, p, (uint32_t)n, data + len, ids, &n_ids) ? 1 : 0;
        } else if (use_cache) {
            uint32_t ln = 0;
            const uint32_t *lid = tsimd_plong_probe_ex(c, p, n, &ln);
            if (lid) {
                hit = 1;
                hit_ids = lid;
                n_ids = ln;
            }
        }

            if (hit) {
                float v = acc;
                for (uint32_t k = 0; k < n_ids; k++) {
                    if (v < -90000.0f || v > 90000.0f) {
                        hit = 0;
                        break;
                    }
                    v += t->vocab.scores[hit_ids[k]];
                }
                if (hit) {
                    for (uint32_t k = 0; k < n_ids; k++) {
                        if (emit_out(ws, hit_ids[k]) != 0) return -1;
                        acc += t->vocab.scores[hit_ids[k]];
                    }
                }
            }
        int was_hit = hit;
        if (!hit) {

            size_t nn = 0;
            uint8_t *norm = spm_norm_alloc(t, p, n, &nn);
            if (!norm) return -1;
            size_t base = ws->out_count;
            float score, margin = 0.0f;
            if (tsimd_unigram_encode_ex(&t->vocab, ws, norm, (uint32_t)nn,
                                        acc, &score, &margin) != 0)
                return -1;
            acc = score;
            uint32_t got = (uint32_t)(ws->out_count - base);

            int cacheable = !was_hit;
            {
                float chk = uni_acc_in;
                for (uint32_t k = 0; k < got; k++)
                    chk += t->vocab.scores[ws->out[base + k]];
                float mag = fabsf(uni_acc_in) + fabsf(score) + 1.0f;
                (void)mag;

                float need = 2.0f * (float)(got + 2) * 100001.0f *
                             1.1920929e-7f;
                if (chk != score || !(margin > need)) cacheable = 0;
            }

            {
                static int noguard = -1;
                if (noguard < 0)
                    noguard = getenv("TSIMD_UNI_NOGUARD") != NULL;
                if (noguard) cacheable = 1;
            }

            if (n <= TSIMD_PCACHE_MAX_SPAN) {
                if (cacheable)
                    if (use_cache)
                        tsimd_pcache_insert(c, p, (uint32_t)n,
                                        ws->out + base, got);
            } else if (cacheable) {
                if (use_cache)
                    tsimd_plong_insert(c, p, n, ws->out + base, got);
            }
        }
        {
            static int memo_dbg = -1;
            if (memo_dbg < 0) {
                const char *e = getenv("TSIMD_MEMO_DEBUG");
                memo_dbg = (e && e[0] == '1') ? 1 : 0;
            }
            if (memo_dbg && pos < 64)
                fprintf(stderr, "UNIT pos=%zu end=%zu acc=%.4f hit=%d\n",
                        pos, end, acc, hit);
        }
        pos = end;

        if (acc < -100000.0f || acc > 100000.0f) acc = 0.0f;
    }
    return 0;
}

static int qwen25_scalar_forced(void) {
    if (tsimd_serve_env_per_call()) {
        const char *e = getenv("TSIMD_QWEN25_SCALAR");
        return e && e[0] == '1';
    }
    return tsimd_serve_qwen25_scalar_cached();
}

int tsimd_encode_segment_pool(tsimd_tokenizer_t *t, const uint8_t *data,
                              size_t len, tsimd_bpe_workspace_t *ws,
                              tsimd_pretok_cache_t *c,
                              tsimd_pretok_cache_t *shared) {
    if (len == 0) return 0;
    if (!c) c = &t->cache;
    if (t->scheme == TSIMD_PRETOK_GEM4) {
        size_t cap = len * 3 + 16;
        uint8_t *norm = tsimd_metaspace_scratch(cap);
        if (!norm) return -1;
        size_t nn = tsimd_metaspace_ex(data, len, norm, cap, 0);
        if (nn == (size_t)-1) return -1;
        if (getenv("TSIMD_ESSC_DBG"))
            fprintf(stderr, "SEGPOOL-GEM4 len=%zu nn=%zu\n", len, nn);
        return tsimd_bpe_encode_spm_blocked_cached(
                   &t->vocab, c, shared, ws, norm, (uint32_t)nn,
                   tsimd_spm_word_target(&t->vocab, seam_block_target()));
    }
    return encode_spm_segment_c(t, data, len, ws, c, shared);
}

static int encode_segment(tsimd_tokenizer_t *t, const uint8_t *data,
                          size_t len, tsimd_bpe_workspace_t *ws) {
    if (t->scheme == TSIMD_PRETOK_O200K ||
        t->scheme == TSIMD_PRETOK_CL100K ||

        t->scheme == TSIMD_PRETOK_DEEPSEEK ||

        t->scheme == TSIMD_PRETOK_NEMO ||

        t->scheme == TSIMD_PRETOK_GPT2 ||
        (t->scheme == TSIMD_PRETOK_GPT2D && tsimd_gpt2d_span_on()) ||

        (t->scheme == TSIMD_PRETOK_QWEN25 && !qwen25_scalar_forced()))
        return tsimd_serve_segment(t, ws, data, len);
    if (t->scheme == TSIMD_PRETOK_KIMI) {
        fuse_ctx_t f = {t, ws, data, data + len};
        if (!kimi_masks_off())
            return tsimd_pretokenize_kimi_masks_ex(data, len, fused_span, &f);
        return tsimd_pretokenize_stream(t->scheme, data, len, fused_span, &f,
                                        NULL);
    }
    if (t->scheme == TSIMD_PRETOK_SPM) {

        if (t->spm_kind && !uni_memo_disabled())
            return encode_unigram_memo(t, data, len, ws);
        return encode_spm_segment(t, data, len, ws);
    }
    if (t->scheme == TSIMD_PRETOK_GEM4) {

        size_t cap = len * 3 + 16;
        uint8_t *norm = tsimd_metaspace_scratch(cap);
        if (!norm) return -1;
        size_t nn = tsimd_metaspace_ex(data, len, norm, cap, 0);
        if (nn == (size_t)-1) return -1;
        return tsimd_bpe_encode_spm_blocked_cached(
                   &t->vocab, &t->cache, NULL, ws, norm, (uint32_t)nn,
                   tsimd_spm_word_target(&t->vocab, seam_block_target()));
    }
    if (!tsimd_batched_serve_disabled())
        return tsimd_serve_batched_segment(t, ws, data, len);
    fuse_ctx_t f = {t, ws, data, data + len};
    return tsimd_pretokenize_stream(t->scheme, data, len, fused_span, &f,
                                   NULL);
}

size_t tsimd_tokenizer_added_len_at(const tsimd_tokenizer_t *t,
                                    const uint8_t *data, size_t len,
                                    size_t pos, size_t *added_index) {
    size_t best_len = 0;

    if (t->added_next != NULL && pos < len) {

        if (pos + 1 < len) {
            size_t key = ((size_t)data[pos] << 8) | data[pos + 1];
            for (uint16_t s = t->added_head2[key]; s != 0xFFFF;
                 s = t->added_next[s]) {
                const tsimd_added_token_t *at = &t->added[s];
                size_t nlen = at->length;
                if (nlen <= best_len || nlen > len - pos) continue;
                if (memcmp(data + pos, at->text, nlen) == 0) {
                    best_len = nlen;
                    if (added_index) *added_index = s;
                }
            }
        }
        for (uint16_t s = t->added_head[data[pos]]; s != 0xFFFF;
             s = t->added_next[s]) {
            const tsimd_added_token_t *at = &t->added[s];
            size_t nlen = at->length;
            if (nlen == 0 || nlen <= best_len || nlen > len - pos)
                continue;
            if (memcmp(data + pos, at->text, nlen) == 0) {
                best_len = nlen;
                if (added_index) *added_index = s;
            }
        }
        return best_len;
    }
    for (size_t s = 0; s < t->added_count; s++) {
        const tsimd_added_token_t *at = &t->added[s];
        size_t nlen = at->length;
        if (nlen == 0 || nlen <= best_len || nlen > len - pos) continue;
        if (at->text[0] != (char)data[pos]) continue;
        if (memcmp(data + pos, at->text, nlen) == 0) {
            best_len = nlen;
            if (added_index) *added_index = s;

        }
    }
    return best_len;
}

#define TSIMD_NFC_CHUNK ((size_t)1 << 20)

static int encode_buffer(tsimd_tokenizer_t *t, const uint8_t *data,
                         size_t len, tsimd_bpe_workspace_t *ws);

static int encode_chunk_nfc(tsimd_tokenizer_t *t, tsimd_bpe_workspace_t *ws,
                            const uint8_t *p, size_t n, int compat,
                            uint8_t *scratch, size_t scratch_cap) {
    if (!tsimd_nfc_may_change(p, n))
        return encode_buffer(t, p, n, ws);
    size_t nn = compat ? tsimd_nfkc_normalize(p, n, scratch, scratch_cap)
                       : tsimd_nfc_normalize(p, n, scratch, scratch_cap);
    if (nn == (size_t)-1) return -1;
    return encode_buffer(t, scratch, nn, ws);
}

static size_t nfc_seam(const tsimd_tokenizer_t *t, const uint8_t *data,
                       size_t len, size_t from);

int tsimd_encode_chunk_nfc_fused(tsimd_tokenizer_t *t,
                                 tsimd_bpe_workspace_t *ws,
                                 const uint8_t *p, size_t n, int compat) {
    if (compat ? !tsimd_nfkc_may_change(p, n)
               : !tsimd_nfc_may_change(p, n))
        return encode_buffer(t, p, n, ws);
    size_t need = n * 4 + 64;
    if (ws->nfc_cap < need) {
        uint8_t *nb = tsimd_xrealloc(ws->nfc_buf, need);
        if (!nb) return -1;
        ws->nfc_buf = nb;
        ws->nfc_cap = need;
    }
    size_t nn = compat ? tsimd_nfkc_normalize(p, n, ws->nfc_buf, ws->nfc_cap)
                       : tsimd_nfc_normalize(p, n, ws->nfc_buf, ws->nfc_cap);
    if (nn == (size_t)-1) return -1;
    return encode_buffer(t, ws->nfc_buf, nn, ws);
}

size_t tsimd_nfc_seam_at(const tsimd_tokenizer_t *t, const uint8_t *data,
                         size_t len, size_t from) {
    return nfc_seam(t, data, len, from);
}

static size_t nfc_seam(const tsimd_tokenizer_t *t, const uint8_t *data,
                       size_t len, size_t from) {
    size_t maxlen = 0;
    for (size_t i = 0; i < t->added_count; i++)
        if (t->added[i].length > maxlen) maxlen = t->added[i].length;
    while (from + 1 < len) {

        long p = tsimd_pretok_safe_split(t->scheme, data, len, from, from);
        if (p < 0) break;
        size_t seam = (size_t)p;
        int inside = 0;
        size_t lo = seam > maxlen ? seam - maxlen : 0;
        for (size_t q = lo; q < seam; q++) {
            size_t hit = tsimd_tokenizer_added_len_at(t, data, len, q, NULL);
            if (hit > 0 && q + hit > seam) {
                inside = 1;
                seam = q + hit;
                break;
            }
        }
        if (!inside && seam > from) return seam;

        if (seam > from)
            from = seam;
        else
            from = from + 1;
    }
    return len;
}

#define TSIMD_PAR_NFC_CHUNK ((size_t)1 << 20)

typedef struct {
    const tsimd_tokenizer_t *t;
    const uint8_t *data;
    uint8_t *out;
    const size_t *seams;
    size_t n_chunks;
    size_t *sizes;
    uint8_t **keep;
    const size_t *doffs;
    uint8_t *scratch;
    size_t scratch_cap;
    int compat;
    uint64_t t_scan, t_norm, t_copy;
} pnorm_job_t;

#if defined(__x86_64__) || defined(__i386__)
static inline uint64_t pnorm_rdtsc(void) { return pnorm_rdtsc(); }
#else
static inline uint64_t pnorm_rdtsc(void) { return 0; }
#endif

static void *pnorm_worker(void *arg) {
    pnorm_job_t *j = arg;

    tsimd_restore_process_affinity();
    const int dbg = getenv("TSIMD_NFC_DBG") != NULL;
    for (size_t c = 0; c < j->n_chunks; c++) {
        const uint8_t *p = j->data + j->seams[c];
        size_t n = j->seams[c + 1] - j->seams[c];
        uint64_t d0 = dbg ? pnorm_rdtsc() : 0;
        int changed = j->compat ? tsimd_nfkc_may_change(p, n)
                                : tsimd_nfc_may_change(p, n);
        if (dbg) { j->t_scan += pnorm_rdtsc() - d0; d0 = pnorm_rdtsc(); }
        if (changed) {
            size_t nn;
            {
                uint64_t n0 = dbg ? pnorm_rdtsc() : 0;
                nn = tsimd_nfc_normalize(p, n, j->scratch,
                                         j->scratch_cap);
                if (dbg) j->t_norm += pnorm_rdtsc() - n0;
            }
            if (nn == (size_t)-1) { j->sizes[c] = (size_t)-1; continue; }
            if (nn > n) {
                uint8_t *kb = tsimd_xmalloc(nn);
                if (!kb) { j->sizes[c] = (size_t)-1; continue; }
                memcpy(kb, j->scratch, nn);
                j->keep[c] = kb;
                j->sizes[c] = nn;
            } else {
                memcpy(j->out + j->seams[c], j->scratch, nn);
                j->sizes[c] = nn;
            }
        } else {
            memcpy(j->out + j->seams[c], p, n);
            j->sizes[c] = n;
        }
        if (dbg) j->t_copy += pnorm_rdtsc() - d0;
    }
    tsimd_nfc_tls_release();
    return NULL;

}

static uint8_t *nfc_normalize_all_impl(const tsimd_tokenizer_t *t,
                                      const uint8_t *data, size_t len,
                                      int max_threads, size_t *out_len,
                                      int reuse) {
    if (t->norm == TSIMD_NORM_NONE) {
        uint8_t *out = tsimd_xmalloc(len ? len : 1);
        if (!out) return NULL;
        if (len) memcpy(out, data, len);
        *out_len = len;
        return out;
    }

    size_t seam_cap = len / TSIMD_PAR_NFC_CHUNK + 8;
    size_t *seams = tsimd_xmalloc(seam_cap * sizeof(size_t));
    if (!seams) return NULL;
    size_t n_chunks = 0;
    seams[n_chunks++] = 0;
    size_t pos = 0;
    while (pos < len) {
        size_t endm = pos + TSIMD_PAR_NFC_CHUNK >= len
                          ? len
                          : nfc_seam(t, data, len,
                                     pos + TSIMD_PAR_NFC_CHUNK);
        if (endm <= pos) endm = pos + 1;
        if (n_chunks >= seam_cap) {
            tsimd_xfree(seams);
            return NULL;
        }
        seams[n_chunks++] = endm;
        pos = endm;
    }

    static _Thread_local uint8_t *tl_norm;
    static _Thread_local size_t tl_norm_cap;
    size_t norm_cap = len + len / 16 + 64;
    uint8_t *norm;
    if (reuse) {
        if (tl_norm_cap < norm_cap) {
            tsimd_xfree(tl_norm);
            tl_norm = tsimd_xmalloc(norm_cap);
            tl_norm_cap = tl_norm ? norm_cap : 0;
        }
        norm = tl_norm;
    } else {
        norm = tsimd_xmalloc(norm_cap);
    }
    size_t *sizes = tsimd_xmalloc(n_chunks * sizeof(size_t));
    size_t *doffs = tsimd_xmalloc(n_chunks * sizeof(size_t));
    uint64_t dbg_scan = 0, dbg_norm = 0, dbg_copy = 0;
    uint8_t **keep = tsimd_xmalloc(n_chunks * sizeof(uint8_t *));
    if (keep) memset(keep, 0, n_chunks * sizeof(uint8_t *));
    if (!norm || !sizes || !keep || !doffs) {
        tsimd_xfree(seams);
        tsimd_xfree(norm);
        tsimd_xfree(sizes);
        tsimd_xfree(keep);
        return NULL;
    }

    size_t max_chunk = 0;
    for (size_t q = 0; q + 1 < n_chunks; q++) {
        size_t l = seams[q + 1] - seams[q];
        if (l > max_chunk) max_chunk = l;
    }
    size_t scratch_cap = max_chunk * 4 + 64;
    int nthreads = (int)n_chunks < 1 ? 1
                   : ((int)n_chunks < max_threads ? (int)n_chunks
                                                  : max_threads);
    if (nthreads <= 1 || n_chunks == 1) {

        pnorm_job_t j = {t, data, norm, seams, n_chunks - 1, sizes,
                         keep, NULL, NULL, scratch_cap};
        j.compat = t->norm == TSIMD_NORM_NFKC;
        j.scratch = tsimd_xmalloc(j.scratch_cap);
        if (!j.scratch) {
            tsimd_xfree(seams); tsimd_xfree(norm); tsimd_xfree(sizes);
            return NULL;
        }
        pnorm_worker(&j);
        tsimd_xfree(j.scratch);
    } else {
        pthread_t th[64];
        pnorm_job_t jobs[64];
        uint8_t *scratch[64];
        size_t per = (n_chunks - 1) / (size_t)nthreads;
        size_t extra = (n_chunks - 1) % (size_t)nthreads;

        memset(sizes, 0xFF, n_chunks * sizeof(size_t));
        size_t c = 0;
        int ok = 1;
        for (int k = 0; k < nthreads; k++) {
            size_t cnt = per + ((size_t)k < extra ? 1 : 0);
            jobs[k].t = t;
            jobs[k].data = data;
            jobs[k].out = norm;
            jobs[k].seams = seams + c;
            jobs[k].n_chunks = cnt;
            jobs[k].sizes = sizes + c;
            jobs[k].keep = keep + c;
            jobs[k].scratch_cap = scratch_cap;
            scratch[k] = tsimd_xmalloc(jobs[k].scratch_cap);
            jobs[k].scratch = scratch[k];
            jobs[k].t_scan = jobs[k].t_norm = jobs[k].t_copy = 0;
            c += cnt;
            if (!scratch[k]) { ok = 0; break; }
        }
        if (ok) {
            for (int k = 0; k < nthreads; k++)
                if (jobs[k].n_chunks &&
                    pthread_create(&th[k], NULL, pnorm_worker, &jobs[k]) != 0)
                    ok = 0;
            for (int k = 0; k < nthreads; k++)
                if (jobs[k].n_chunks && ok) pthread_join(th[k], NULL);
            for (int k = 0; ok && k < nthreads; k++)
                for (size_t q = 0; q < jobs[k].n_chunks; q++)
                    if (jobs[k].sizes[q] == (size_t)-1) ok = 0;
        }
        for (int k = 0; k < nthreads; k++) {
            dbg_scan += jobs[k].t_scan;
            dbg_norm += jobs[k].t_norm;
            dbg_copy += jobs[k].t_copy;
        }
        for (int k = 0; k < nthreads; k++) tsimd_xfree(scratch[k]);
        if (!ok) {
            for (size_t q = 0; q < n_chunks; q++) tsimd_xfree(keep[q]);
            tsimd_xfree(seams); tsimd_xfree(norm); tsimd_xfree(sizes);
            tsimd_xfree(keep); tsimd_xfree(doffs);
            return NULL;
        }
    }

    size_t off = 0;
    for (size_t c2 = 0; c2 + 1 < n_chunks; c2++) {
        size_t nn = sizes[c2];
        if (nn == (size_t)-1) {
            for (size_t q = 0; q < n_chunks; q++) tsimd_xfree(keep[q]);
            tsimd_xfree(seams); tsimd_xfree(norm); tsimd_xfree(sizes);
            tsimd_xfree(keep); tsimd_xfree(doffs);
            return NULL;
        }
    }
    off = 0;
    for (size_t c2 = 0; c2 + 1 < n_chunks; c2++) {
        size_t nn = sizes[c2];
        const uint8_t *src = keep[c2] ? keep[c2] : norm + seams[c2];
        if (src != norm + off) memmove(norm + off, src, nn);
        off += nn;
    }
    if (getenv("TSIMD_NFC_DBG") && nthreads > 1)
        fprintf(stderr,
                "tsimd nfc chunk: scan %.2f Mcyc, norm %.2f Mcyc, "
                "copy %.2f Mcyc\n",
                (double)dbg_scan / 1e6, (double)dbg_norm / 1e6,
                (double)dbg_copy / 1e6);
    for (size_t q = 0; q < n_chunks; q++) tsimd_xfree(keep[q]);
    tsimd_xfree(seams);
    tsimd_xfree(sizes);
    tsimd_xfree(keep);
    tsimd_xfree(doffs);
    *out_len = off;
    return norm;
}

static int seam_push(tsimd_added_seam_list_t *l, size_t off, uint32_t len,
                     uint32_t idx) {
    if (l->count == l->cap) {
        size_t ncap = l->cap * 2;
        tsimd_added_seam_t *ni;
        if (l->items == l->inline_items) {
            ni = tsimd_xmalloc(ncap * sizeof(*ni));
            if (!ni) return -1;
            memcpy(ni, l->items, l->count * sizeof(*ni));
        } else {
            ni = tsimd_xrealloc(l->items, ncap * sizeof(*ni));
            if (!ni) return -1;
        }
        l->items = ni;
        l->cap = ncap;
    }
    l->items[l->count].off = off;
    l->items[l->count].len = len;
    l->items[l->count].idx = idx;
    l->count++;
    return 0;
}

int tsimd_tokenizer_added_seams(const tsimd_tokenizer_t *t,
                                const uint8_t *data, size_t len,
                                tsimd_added_seam_list_t *out) {
    out->cap = sizeof(out->inline_items) / sizeof(out->inline_items[0]);
    out->items = out->inline_items;
    out->count = 0;
    if (len == 0 || t->added_count == 0) return 0;
    uint8_t first[256] = {0};
    for (size_t s = 0; s < t->added_count; s++)
        if (t->added[s].length > 0)
            first[(uint8_t)t->added[s].text[0]] = 1;
    uint8_t fb[256];
    int nfb = 0;
    for (int b = 0; b < 256; b++)
        if (first[b]) fb[nfb++] = (uint8_t)b;
    if (nfb == 0) return 0;
#if defined(__x86_64__) || defined(__i386__)
    if (nfb >= 1 && nfb <= 8) {
        __m128i vb[8];
        for (int k = 0; k < nfb; k++)
            vb[k] = _mm_set1_epi8((char)fb[k]);
        size_t pos = 0;
        while (pos + 16 <= len) {
            __m128i v = _mm_loadu_si128(
                (const __m128i *)(const void *)(data + pos));
            __m128i m = _mm_cmpeq_epi8(v, vb[0]);
            for (int k = 1; k < nfb; k++)
                m = _mm_or_si128(m, _mm_cmpeq_epi8(v, vb[k]));
            unsigned mask = (unsigned)_mm_movemask_epi8(m);
            int hit_in_block = 0;
            while (mask) {
                unsigned k2 = (unsigned)__builtin_ctz(mask);
                size_t cand = pos + k2;
                size_t idx = 0;
                size_t hit = tsimd_tokenizer_added_len_at(t, data, len, cand,
                                                          &idx);
                if (hit) {
                    if (seam_push(out, cand, (uint32_t)hit,
                                  (uint32_t)idx) != 0)
                        return -1;
                    pos = cand + hit;
                    hit_in_block = 1;
                    break;
                }
                mask &= mask - 1;
            }
            if (!hit_in_block) pos += 16;
        }
        for (; pos < len; pos++) {
            if (!first[data[pos]]) continue;
            size_t idx = 0;
            size_t hit = tsimd_tokenizer_added_len_at(t, data, len, pos, &idx);
            if (hit) {
                if (seam_push(out, pos, (uint32_t)hit, (uint32_t)idx) != 0)
                    return -1;
                pos += hit - 1;
            }
        }
        return 0;
    }
#endif
    {
        size_t pos = 0;
        while (pos < len) {
            const uint8_t *hit = NULL;
            for (int k = 0; k < nfb; k++) {
                const uint8_t *h = memchr(data + pos, fb[k], len - pos);
                if (h && (!hit || h < hit)) hit = h;
            }
            if (!hit) break;
            size_t cand = (size_t)(hit - data);
            size_t idx = 0;
            size_t l = tsimd_tokenizer_added_len_at(t, data, len, cand, &idx);
            if (l) {
                if (seam_push(out, cand, (uint32_t)l, (uint32_t)idx) != 0)
                    return -1;
                pos = cand + l;
            } else {
                pos = cand + 1;
            }
        }
    }
    return 0;
}

void tsimd_added_seam_list_free(tsimd_added_seam_list_t *list) {
    if (list->items != list->inline_items) tsimd_xfree(list->items);
    list->items = list->inline_items;
    list->count = 0;
    list->cap = sizeof(list->inline_items) / sizeof(list->inline_items[0]);
}

static int encode_buffer(tsimd_tokenizer_t *t, const uint8_t *data,
                         size_t len, tsimd_bpe_workspace_t *ws) {
    if (len == 0) return 0;
    if (t->added_count == 0) return encode_segment(t, data, len, ws);

    tsimd_added_seam_list_t seams;
    if (tsimd_tokenizer_added_seams(t, data, len, &seams) != 0) return -1;
    size_t seg = 0;
    int rc = 0;
    for (size_t i = 0; i < seams.count; i++) {
        const tsimd_added_seam_t *sm = &seams.items[i];
        if (sm->off > seg &&
            encode_segment(t, data + seg, sm->off - seg, ws) != 0) {
            rc = -1;
            break;
        }
        const tsimd_added_token_t *at = &t->added[sm->idx];
        if (at->length > 0 && emit_out(ws, at->id) != 0) {
            rc = -1;
            break;
        }
        seg = sm->off + sm->len;
    }
    if (rc == 0 && len > seg &&
        encode_segment(t, data + seg, len - seg, ws) != 0)
        rc = -1;
    tsimd_added_seam_list_free(&seams);
    return rc;
}

int tsimd_encode_buffer_pool(tsimd_tokenizer_t *t, const uint8_t *data,
                             size_t len, tsimd_bpe_workspace_t *ws,
                             tsimd_pretok_cache_t *c,
                             tsimd_pretok_cache_t *shared) {
    if (len == 0) return 0;
    if (t->added_count == 0)
        return tsimd_encode_segment_pool(t, data, len, ws, c, shared);
    if (t->scheme == TSIMD_PRETOK_GEM4 && !tsimd_spm_norm_off()) {
        tsimd_added_seam_list_t seams;
        if (tsimd_tokenizer_added_seams(t, data, len, &seams) != 0) return -1;
        size_t nseg = seams.count + 1;
        size_t *hard = tsimd_xmalloc(nseg * sizeof(size_t));
        size_t *counts = tsimd_xmalloc(nseg * sizeof(size_t));
        size_t cap = len * 3 + 16;
        uint8_t *norm = tsimd_xmalloc(cap);
        uint32_t *added_ids = tsimd_xmalloc(seams.count * sizeof(uint32_t));
        if (!hard || !counts || !norm || !added_ids) {
            tsimd_xfree(hard);
            tsimd_xfree(counts);
            tsimd_xfree(norm);
            tsimd_xfree(added_ids);
            tsimd_added_seam_list_free(&seams);
            return -1;
        }
        size_t noff = 0, seg = 0, nseg_real = 0;
        int rc = 0;
        for (size_t i = 0; i < seams.count && rc == 0; i++) {
            const tsimd_added_seam_t *sm = &seams.items[i];
            if (sm->off > seg) {
                size_t nn = tsimd_metaspace_ex(data + seg, sm->off - seg,
                                               norm + noff, cap - noff, 0);
                if (nn == (size_t)-1) {
                    rc = -1;
                    break;
                }
                noff += nn;
            }
            hard[nseg_real++] = noff;
            added_ids[i] = t->added[sm->idx].id;
            seg = sm->off + sm->len;
        }
        if (rc == 0 && len > seg) {
            size_t nn = tsimd_metaspace_ex(data + seg, len - seg, norm + noff,
                                           cap - noff, 0);
            if (nn == (size_t)-1)
                rc = -1;
            else
                noff += nn;
        }
        if (rc == 0) hard[nseg_real++] = noff;
        if (rc == 0) {
            size_t noff2 = 0;
            for (size_t i = 0; i < nseg_real && rc == 0; i++) {
                size_t seglen = hard[i] - noff2;
                if (seglen &&
                    tsimd_bpe_encode_spm_blocked_cached(
                        &t->vocab, c, shared, ws, norm + noff2,
                        (uint32_t)seglen,
                        tsimd_spm_word_target(&t->vocab,
                                              seam_block_target())) != 0)
                    rc = -1;
                noff2 += seglen;
                if (i < seams.count) {
                    if (emit_out(ws, added_ids[i]) != 0) rc = -1;
                }
            }
        }
        tsimd_xfree(hard);
        tsimd_xfree(counts);
        tsimd_xfree(norm);
        tsimd_xfree(added_ids);
        tsimd_added_seam_list_free(&seams);
        return rc;
    }
    tsimd_added_seam_list_t seams;
    if (tsimd_tokenizer_added_seams(t, data, len, &seams) != 0) return -1;
    size_t seg = 0;
    int rc = 0;
    for (size_t i = 0; i < seams.count; i++) {
        const tsimd_added_seam_t *sm = &seams.items[i];
        if (sm->off > seg &&
            tsimd_encode_segment_pool(t, data + seg, sm->off - seg, ws, c,
                                      shared) != 0) {
            rc = -1;
            break;
        }
        const tsimd_added_token_t *at = &t->added[sm->idx];
        if (at->length > 0 && emit_out(ws, at->id) != 0) {
            rc = -1;
            break;
        }
        seg = sm->off + sm->len;
    }
    if (rc == 0 && len > seg &&
        tsimd_encode_segment_pool(t, data + seg, len - seg, ws, c, shared) !=
            0)
        rc = -1;
    tsimd_added_seam_list_free(&seams);
    return rc;
}

static int encode_body(tsimd_tokenizer_t *t, const uint8_t *data, size_t len,
                       tsimd_bpe_workspace_t *ws) {
    int rc;
    if (t->norm == TSIMD_NORM_NFC || t->norm == TSIMD_NORM_NFKC) {
        int compat = t->norm == TSIMD_NORM_NFKC;
        rc = 0;
        size_t pos = 0;
        uint8_t *scratch = NULL;
        size_t scratch_cap = 0;
        while (rc == 0 && pos < len) {
            size_t end = pos + TSIMD_NFC_CHUNK >= len
                             ? len
                             : nfc_seam(t, data, len,
                                        pos + TSIMD_NFC_CHUNK);
            const uint8_t *p = data + pos;
            size_t n = end - pos;
            if (compat ? tsimd_nfkc_may_change(p, n)
                       : tsimd_nfc_may_change(p, n)) {
                size_t need = n * (compat ? 24 : 4) + 64;
                if (need > scratch_cap) {
                    tsimd_xfree(scratch);
                    scratch = tsimd_xmalloc(need);
                    scratch_cap = need;
                }
                rc = encode_chunk_nfc(t, ws, p, n, compat, scratch,
                                      scratch_cap);
                pos = end;
                continue;
            }
            rc = encode_buffer(t, p, n, ws);
            pos = end;
        }
        tsimd_xfree(scratch);
    } else {
        rc = encode_buffer(t, data, len, ws);
    }
    return rc;
}

static int encode_into_impl(tsimd_tokenizer_t *t, const uint8_t *data,
                            size_t len, uint32_t *ids, size_t cap,
                            int foreign, uint32_t **out_ids, size_t *out_cap,
                            size_t *out_count) {
    tsimd_serve_env_refresh();
    tsimd_pcache_upgrade_for(t, len);
    tsimd_memo_size_for(t, len);
    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, t->vocab.max_token_len) != 0) return -1;
    tsimd_xfree(ws.out);
    ws.out = ids;
    ws.out_capacity = cap;
    ws.out_foreign = ids != NULL && foreign;
    int rc = encode_body(t, data, len, &ws);
    *out_ids = ws.out;
    *out_cap = ws.out_capacity;
    *out_count = rc == 0 ? ws.out_count : 0;
    ws.out = NULL;
    ws.out_capacity = 0;
    ws.out_count = 0;
    tsimd_bpe_workspace_free(&ws);
    return rc;
}

int tsimd_tokenizer_encode(tsimd_tokenizer_t *t, const uint8_t *data,
                          size_t len, tsimd_token_sequence_t *out) {
    uint32_t *ids = NULL;
    size_t cap = 0, count = 0;
    int rc = encode_into_impl(t, data, len, NULL, 0, 0, &ids, &cap, &count);
    if (rc != 0) {
        tsimd_xfree(ids);
        out->ids = NULL;
        out->count = 0;
        out->cap = 0;
        return -1;
    }
    out->ids = ids;
    out->count = count;
    out->cap = cap;
    return 0;
}

int tsimd_tokenizer_encode_into(tsimd_tokenizer_t *t, const uint8_t *data,
                               size_t len, tsimd_token_sequence_t *seq) {
    uint32_t *ids = NULL;
    size_t cap = 0, count = 0;
    int rc = encode_into_impl(t, data, len, seq->ids, seq->cap, 0, &ids, &cap,
                              &count);
    seq->ids = ids;
    seq->count = count;
    seq->cap = cap;
    return rc;
}

size_t tsimd_tokenizer_encode_bound(const tsimd_tokenizer_t *t, size_t len) {
    int expanding = t->norm == TSIMD_NORM_NFC || t->norm == TSIMD_NORM_NFKC ||
                    t->scheme == TSIMD_PRETOK_SPM ||
                    t->scheme == TSIMD_PRETOK_GEM4;
    size_t per = expanding ? 18u : 1u;
    return len * per + (size_t)t->added_count + 16;
}

int tsimd_tokenizer_encode_into_raw(tsimd_tokenizer_t *t, const uint8_t *data,
                                   size_t len, uint32_t *buf, size_t cap,
                                   size_t *count) {
    if (!buf || cap < tsimd_tokenizer_encode_bound(t, len)) return -1;
    uint32_t *ids = NULL;
    size_t out_cap = 0, out_count = 0;
    int rc = encode_into_impl(t, data, len, buf, cap, 1, &ids, &out_cap,
                              &out_count);
    if (rc != 0) {
        if (ids && ids != buf) tsimd_xfree(ids);
        return -1;
    }
    if (ids != buf) {
        if (out_count > cap) {
            tsimd_xfree(ids);
            return -1;
        }
        if (out_count) memcpy(buf, ids, out_count * sizeof(uint32_t));
        tsimd_xfree(ids);
    }
    if (count) *count = out_count;
    return 0;
}

uint8_t *tsimd_nfc_normalize_all(const tsimd_tokenizer_t *t,
                                 const uint8_t *data, size_t len,
                                 int max_threads, size_t *out_len) {
    return nfc_normalize_all_impl(t, data, len, max_threads, out_len, 0);
}

uint8_t *tsimd_nfc_normalize_all_tl(const tsimd_tokenizer_t *t,
                                    const uint8_t *data, size_t len,
                                    int max_threads, size_t *out_len) {
    return nfc_normalize_all_impl(t, data, len, max_threads, out_len, 1);
}

int tsimd_tokenizer_decode_into(const tsimd_tokenizer_t *t,
                                const uint32_t *ids, size_t n,
                                int skip_special, uint8_t *buf, size_t cap,
                                size_t *out_len) {
    if (out_len) *out_len = 0;
    if (!buf && cap) return -1;

    size_t total = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t id = ids[i];
        const tsimd_added_token_t *at0 =
            id < t->added_by_id_n ? t->added_by_id[id] : NULL;
        if (at0) {
            if (!at0->is_special || !skip_special) total += at0->length;
            continue;
        }
        if (id < t->vocab.vocab_size) {
            if (t->scheme == TSIMD_PRETOK_SPM) {
                if (id < t->spm_ctl_n * 8 &&
                    (t->spm_ctl_flags[id >> 3] >> (id & 7)) & 1)
                    continue;
                if (id == t->vocab.spm_unk_id) {
                    total += t->vocab.spm_unk_surface_len;
                    continue;
                }
            }
            total += t->vocab.offsets[id + 1] - t->vocab.offsets[id];
            continue;
        }
        fprintf(stderr, "tsimd: cannot decode id %u\n", id);
        return -1;
    }
    if (out_len) *out_len = total;
    if (!buf) return 0;
    if (total > cap) return -1;
    if (total == 0) return 0;
    size_t off = 0;
    for (size_t i = 0; i < n; i++) {
        size_t pf = i + 16;
        if (pf < n && ids[pf] < t->vocab.vocab_size) {
            uint32_t pb = t->vocab.offsets[ids[pf]];
            __builtin_prefetch(t->vocab.bytes + pb);
            __builtin_prefetch(t->vocab.bytes + pb + 64);
        }
        uint32_t id = ids[i];
        const tsimd_added_token_t *at1 =
            id < t->added_by_id_n ? t->added_by_id[id] : NULL;
        if (at1) {
            if (!at1->is_special || !skip_special) {
                memcpy(buf + off, at1->text, at1->length);
                off += at1->length;
            }
            continue;
        }
        if (id < t->vocab.vocab_size) {
            if (t->scheme == TSIMD_PRETOK_SPM) {
                if (id < t->spm_ctl_n * 8 &&
                    (t->spm_ctl_flags[id >> 3] >> (id & 7)) & 1)
                    continue;
                if (id == t->vocab.spm_unk_id) {
                    memcpy(buf + off, t->vocab.spm_unk_surface,
                           t->vocab.spm_unk_surface_len);
                    off += t->vocab.spm_unk_surface_len;
                    continue;
                }
            }
            uint32_t b = t->vocab.offsets[id], e = t->vocab.offsets[id + 1];
            memcpy(buf + off, t->vocab.bytes + b, e - b);
            off += e - b;
            continue;
        }
    }
    if (t->scheme == TSIMD_PRETOK_SPM || t->scheme == TSIMD_PRETOK_GEM4) {
        size_t r = 0, w = 0;
        if ((t->spm_dummy_prefix || t->spm_remove_extra) && off >= 3 &&
            buf[0] == 0xE2 && buf[1] == 0x96 && buf[2] == 0x81)
            r = 3;
        for (; r < off;) {
            if (r + 3 <= off && buf[r] == 0xE2 && buf[r + 1] == 0x96 &&
                buf[r + 2] == 0x81) {
                buf[w++] = ' ';
                r += 3;
            } else {
                buf[w++] = buf[r++];
            }
        }
        off = w;
    }
    if (out_len) *out_len = off;
    return 0;
}

int tsimd_tokenizer_decode(const tsimd_tokenizer_t *t,
                           const uint32_t *ids, size_t n, int skip_special,
                           tsimd_byte_sequence_t *out) {
    out->bytes = NULL;
    out->len = 0;
    size_t total = 0;
    if (tsimd_tokenizer_decode_into(t, ids, n, skip_special, NULL, 0,
                                    &total) != 0)
        return -1;
    if (total == 0) return 0;
    uint8_t *buf = tsimd_xmalloc(total);
    if (!buf) return -1;
    size_t off = 0;
    if (tsimd_tokenizer_decode_into(t, ids, n, skip_special, buf, total,
                                    &off) != 0) {
        free(buf);
        return -1;
    }
    out->bytes = buf;
    out->len = off;
    return 0;
}

void tsimd_byte_sequence_free(tsimd_byte_sequence_t *seq) {
    tsimd_xfree(seq->bytes);
    seq->bytes = NULL;
    seq->len = 0;
}

void tsimd_token_sequence_free(tsimd_token_sequence_t *seq) {
    tsimd_xfree(seq->ids);
    seq->ids = NULL;
    seq->count = 0;
    seq->cap = 0;
}
