#include "tokenizer.h"
#include "bpe.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GEMMA_MODEL TSIMD_SRC_DIR "/models/reference/gemma4_tokenizer.json"
#define MISTRAL_MODEL TSIMD_SRC_DIR "/models/reference/mistral7b_tokenizer.model"

static uint64_t rng_state = 0x243F6A8885A308D3ull;
static uint64_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static size_t meta_norm(const uint8_t *in, size_t len, uint8_t *out,
                        size_t cap, int dummy_prefix) {
    size_t n = 0;
    if (dummy_prefix) {
        out[n++] = 0xE2;
        out[n++] = 0x96;
        out[n++] = 0x81;
    }
    for (size_t i = 0; i < len; i++) {
        if (in[i] == ' ') {
            out[n++] = 0xE2;
            out[n++] = 0x96;
            out[n++] = 0x81;
        } else {
            out[n++] = in[i];
        }
        if (n + 4 > cap) return (size_t)-1;
    }
    return n;
}

static int load_tok(tsimd_tokenizer_t *t, const char *path,
                    tsimd_model_format_t fmt) {
    memset(t, 0, sizeof *t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_SPM;
    if (tsimd_tokenizer_load_model(t, fmt, path, NULL, &scheme, NULL, 0) != 0)
        return -1;
    return 0;
}

static int check_one(const tsimd_tokenizer_t *t, const uint8_t *norm,
                     size_t nn, size_t target, const char *label) {
    tsimd_bpe_workspace_t wa, wb;
    if (tsimd_bpe_workspace_init(&wa, t->vocab.max_token_len) != 0) return -1;
    if (tsimd_bpe_workspace_init(&wb, t->vocab.max_token_len) != 0) return -1;
    int rc = 0;
    if (tsimd_bpe_encode_spm(&t->vocab, &wa, norm, (uint32_t)nn) != 0) rc = -1;
    if (rc == 0 &&
        tsimd_bpe_encode_spm_blocked(&t->vocab, &wb, norm, (uint32_t)nn,
                                     target) != 0)
        rc = -1;
    if (rc == 0) {
        if (wa.out_count != wb.out_count ||
            (wa.out_count &&
             memcmp(wa.out, wb.out, wa.out_count * sizeof(uint32_t)) != 0)) {
            size_t d = 0;
            while (d < wa.out_count && d < wb.out_count &&
                   wa.out[d] == wb.out[d])
                d++;
            fprintf(stderr,
                    "FAIL %s target %zu: %zu vs %zu ids, first diff at %zu\n",
                    label, target, wa.out_count, wb.out_count, d);
            rc = 1;
        }
    }
    tsimd_bpe_workspace_free(&wa);
    tsimd_bpe_workspace_free(&wb);
    return rc;
}

int main(void) {
    int fail = 0;
    tsimd_tokenizer_t tg, tm;
    if (load_tok(&tg, GEMMA_MODEL, TSIMD_MODEL_TOKENIZER_JSON) != 0) {
        fprintf(stderr, "FAIL: cannot load gemma4\n");
        return 1;
    }
    if (load_tok(&tm, MISTRAL_MODEL, TSIMD_MODEL_SENTENCEPIECE) != 0) {
        fprintf(stderr, "FAIL: cannot load mistral7b\n");
        return 1;
    }

    int gok = tsimd_vocab_spm_seam_ok(&tg.vocab);
    int mok = tsimd_vocab_spm_seam_ok(&tm.vocab);
    printf("coarse seam property: gemma4 %s, mistral7b %s\n",
           gok ? "OK" : "not global (the per-seam proof runs)",
           mok ? "OK" : "not global (the per-seam proof runs)");
    {
        const uint8_t sample[] = "hello world this is a sample text";
        uint8_t norm[128];
        size_t nn = meta_norm(sample, sizeof sample - 1, norm, sizeof norm, 0);
        size_t s1 = tsimd_spm_seam_find(&tg.vocab, norm, nn, 8, 4);
        size_t s2 = tsimd_spm_seam_find(&tm.vocab, norm, nn, 8, 4);
        printf("precise seam search: gemma4 %zu, mistral7b %zu\n", s1, s2);
        if (s1 == 0 || s2 == 0) {
            fprintf(stderr, "FAIL: the real models must expose proved seams\n");
            fail = 1;
        }
    }

    static const size_t targets[] = { 4096, 16384, 65536, 1u << 20 };
    const char *WORDS[] = { "the", "quick", "brown", "fox", "jumps", "over",
                            "lazy", "dog", "tokenizer", "throughput",
                            "sentence", "piece", "gemma", "model", "fine",
                            "tuning", "pipeline", "  ", " ", "\t",
                            "don't", "e.g.", "U.S.A.", "\xe2\x82\xac",
                            "\xc3\xa9t\xc3\xa9", "\xe4\xbd\xa0\xe5\xa5\xbd" };
    const size_t NW = sizeof WORDS / sizeof WORDS[0];
    uint8_t raw[1 << 16], norm[3 * (1 << 16)];
    for (int it = 0; it < 12 && !fail; it++) {
        size_t n = 0;

        while (n < 60000) {
            const char *w = WORDS[rnd() % NW];
            size_t wl = strlen(w);
            if (n + wl + 2 > sizeof raw) break;
            memcpy(raw + n, w, wl);
            n += wl;
            raw[n++] = (rnd() % 5 == 0) ? '\n' : ' ';
        }
        size_t nn = meta_norm(raw, n, norm, sizeof norm, 0);
        if (nn == (size_t)-1) { fail = 1; break; }
        for (size_t ti = 0; ti < sizeof targets / sizeof targets[0]; ti++) {
            if (check_one(&tg, norm, nn, targets[ti], "gemma") != 0) fail = 1;
            if (check_one(&tm, norm, nn, targets[ti], "mistral") != 0) fail = 1;
        }
    }

    tsimd_tokenizer_free(&tg);
    tsimd_tokenizer_free(&tm);
    if (fail) {
        fprintf(stderr, "test_bpe_seam: FAILED\n");
        return 1;
    }
    printf("test_bpe_seam: OK (blocked == whole on all cases)\n");
    return 0;
}
