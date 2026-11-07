#include "tokenizer.h"
#include "bpe.h"
#include "vocabulary.h"
#include "unicode.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MISTRAL_MODEL TSIMD_SRC_DIR "/models/reference/mistral7b_tokenizer.model"
#define QWEN_MODEL TSIMD_SRC_DIR "/models/reference/qwen2.5-0.5b_tokenizer.json"

static int g_fail;
static int checks;

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static uint32_t naive_replay(const tsimd_vocab_t *v,
                             const uint8_t *bytes, uint32_t len,
                             uint32_t *out, uint32_t cap) {
    uint32_t ids[4096];
    uint32_t n = 0;
    if (len > 4096) return 0;

    for (uint32_t i = 0; i < len; i++) ids[n++] = v->byte_ids[bytes[i]];
    for (;;) {
        uint32_t bestp = UINT32_MAX, besti = 0, bestid = 0;
        for (uint32_t i = 0; i + 1 < n; i++) {
            uint32_t id = 0;
            uint32_t p = tsimd_vocab_merge_prio_id(v, ids[i], ids[i + 1],
                                                   &id);
            if (p == UINT32_MAX) continue;
            if (p < bestp || (p == bestp && i < besti)) {
                bestp = p;
                besti = i;
                bestid = id;
            }
        }
        if (bestp == UINT32_MAX) break;
        ids[besti] = bestid;
        memmove(&ids[besti + 1], &ids[besti + 2],
                (size_t)(n - besti - 2) * sizeof(uint32_t));
        n--;
    }
    uint32_t nout = n < cap ? n : cap;
    memcpy(out, ids, (size_t)nout * sizeof(uint32_t));
    return nout;
}

static uint32_t naive_replay_spm(const tsimd_vocab_t *v,
                                 const uint8_t *bytes, uint32_t len,
                                 uint32_t *out, uint32_t cap) {
    uint32_t ids[4096];
    uint32_t n = 0;
    for (uint32_t i = 0; i < len;) {
        uint32_t cp;
        tsimd_char_class_t cls;
        size_t a = tsimd_utf8_next(bytes + i, len - i, &cp, &cls);
        if (a == 0) a = 1;
        uint32_t p = a == 1 ? v->byte_ids[bytes[i]]
                            : tsimd_vocab_find(v, bytes + i, (uint32_t)a);
        if (p != UINT32_MAX) {
            ids[n++] = p;
        } else {
            for (uint32_t b = 0; b < (uint32_t)a; b++)
                ids[n++] = v->byte_ids[bytes[i + b]];
        }
        if (n + 5 > 4096) return 0;
        i += (uint32_t)a;
    }
    for (;;) {
        uint32_t bestp = UINT32_MAX, besti = 0, bestid = 0;
        for (uint32_t i = 0; i + 1 < n; i++) {
            uint32_t id = 0;
            uint32_t p = tsimd_vocab_merge_prio_id(v, ids[i], ids[i + 1],
                                                   &id);
            if (p == UINT32_MAX) continue;
            if (p < bestp || (p == bestp && i < besti)) {
                bestp = p;
                besti = i;
                bestid = id;
            }
        }
        if (bestp == UINT32_MAX) break;
        ids[besti] = bestid;
        memmove(&ids[besti + 1], &ids[besti + 2],
                (size_t)(n - besti - 2) * sizeof(uint32_t));
        n--;
    }
    uint32_t nout = n < cap ? n : cap;
    memcpy(out, ids, (size_t)nout * sizeof(uint32_t));
    return nout;
}

#define NWORDS 24
static const char *WORDS[NWORDS] = {
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "tokenizer", "throughput", "sentence", "piece", "model", "fine",
    "tuning", "pipeline", "  ", " ", "don't", "e.g.", "U.S.A.",
    "european", "under", "world"
};

static size_t gen_text(uint8_t *dst, size_t cap) {
    size_t n = 0;
    while (n + 24 < cap) {
        const char *w = WORDS[rnd() % NWORDS];
        size_t wl = strlen(w);
        memcpy(dst + n, w, wl);
        n += wl;
        dst[n++] = (rnd() % 7 == 0) ? '\n' : ' ';
    }
    return n;
}

static void check_model(const char *path, tsimd_model_format_t fmt,
                        const char *label, int spm, size_t ntexts,
                        size_t tlen) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_SPM;
    if (tsimd_tokenizer_load_model(&t, fmt, path, NULL, &scheme, NULL, 0) !=
        0) {
        fprintf(stderr, "FAIL: cannot load %s\n", path);
        g_fail = 1;
        return;
    }
    tsimd_bpe_workspace_t ws;
    if (tsimd_bpe_workspace_init(&ws, t.vocab.max_token_len) != 0) {
        g_fail = 1;
        return;
    }
    uint8_t *raw = malloc(tlen + 64);
    uint8_t *norm = malloc(3 * (tlen + 64));
    uint32_t *ref = malloc((tlen + 64) * sizeof(uint32_t));
    size_t nheap = 0, nshort = 0, syms = 0;
    for (size_t it = 0; it < ntexts; it++) {

        size_t want = (it % 3 == 0) ? 40 + (rnd() % 200) : tlen;
        size_t n = gen_text(raw, want);

        size_t nn = 0;
        for (size_t i = 0; i < n; i++) {
            if (raw[i] == ' ') {
                norm[nn++] = 0xE2; norm[nn++] = 0x96; norm[nn++] = 0x81;
            } else {
                norm[nn++] = raw[i];
            }
        }
        const uint8_t *txt = spm ? norm : raw;
        size_t tl = spm ? nn : n;
        uint32_t nref = spm ? naive_replay_spm(&t.vocab, txt, (uint32_t)tl,
                                               ref, (uint32_t)tl + 64)
                            : naive_replay(&t.vocab, txt, (uint32_t)tl,
                                           ref, (uint32_t)tl + 64);
        ws.out_count = 0;
        int rc = spm ? tsimd_bpe_encode_spm(&t.vocab, &ws, txt,
                                            (uint32_t)tl)
                     : tsimd_bpe_encode(&t.vocab, &ws, txt, (uint32_t)tl);
        if (rc != 0) {
            fprintf(stderr, "FAIL %s: encode rc\n", label);
            g_fail = 1;
            break;
        }
        if (ws.out_count != nref ||
            memcmp(ws.out, ref, (size_t)nref * sizeof(uint32_t)) != 0) {
            size_t d = 0;
            while (d < nref && d < ws.out_count && ws.out[d] == ref[d]) d++;
            fprintf(stderr,
                    "FAIL %s: text %zu (%zu bytes) — %zu ids vs oracle %zu, "
                    "first diff at %zu\n", label, it, tl, ws.out_count, nref,
                    d);
            g_fail = 1;
            break;
        }
        syms += tl;
        if (tl > 400) nheap++;
        else nshort++;
        checks++;
    }
    printf("  %s: %zu texts checked (long %zu, short %zu), %zu bytes\n",
           label, checks, nheap, nshort, syms);
    free(raw);
    free(norm);
    free(ref);
    tsimd_bpe_workspace_free(&ws);
    tsimd_tokenizer_free(&t);
}

int main(void) {

    check_model(MISTRAL_MODEL, TSIMD_MODEL_SENTENCEPIECE, "mistral7b spm", 1,
                12, 700);

    check_model(QWEN_MODEL, TSIMD_MODEL_TOKENIZER_JSON, "qwen2.5 hf", 0, 12,
                700);
    if (g_fail) {
        fprintf(stderr, "test_bpe_heap_oracle: FAILED\n");
        return 1;
    }
    printf("test_bpe_heap_oracle: OK (heap path == naive rescan)\n");
    return 0;
}
