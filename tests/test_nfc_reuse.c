#include "tokenizer.h"
#include "unicode.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_NFC \
    TSIMD_SRC_DIR "/models/reference/qwen3_tokenizer.json"

static int g_fail;
static int checks;

static uint32_t *encode_ids(tsimd_tokenizer_t *t, const uint8_t *data,
                            size_t len, size_t *nout) {
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(t, data, len, &seq) != 0) return NULL;
    uint32_t *ids = malloc(seq.count ? seq.count * sizeof(uint32_t) : 4);
    if (!ids) {
        tsimd_token_sequence_free(&seq);
        return NULL;
    }
    memcpy(ids, seq.ids, seq.count * sizeof(uint32_t));
    *nout = seq.count;
    tsimd_token_sequence_free(&seq);
    return ids;
}

static void chk_ids(const uint32_t *a, size_t na, const uint32_t *b, size_t nb,
                    const char *what) {
    checks++;
    if (na != nb || memcmp(a, b, na * sizeof(uint32_t)) != 0) {
        size_t i = 0;
        while (i < na && i < nb && a[i] == b[i]) i++;
        fprintf(stderr,
                "FAIL %s: %zu vs %zu ids, first diff at %zu (%u vs %u)\n",
                what, na, nb, i, i < na ? a[i] : 0u, i < nb ? b[i] : 0u);
        g_fail = 1;
    }
}

static size_t gen_mixed(uint8_t *dst, size_t want, unsigned seed) {
    static const char *words[] = {"the", "of", "and", "a", "to", "in"};
    size_t n = 0;
    unsigned r = seed | 1u;
    while (n + 16 < want) {
        r = r * 1103515245u + 12345u;
        int kind = (int)((r >> 16) % 10);
        if (kind < 5) {
            const char *w = words[(r >> 8) % 6];
            size_t l = strlen(w);
            memcpy(dst + n, w, l);
            n += l;
            dst[n++] = ' ';
        } else if (kind < 7) {
            memcpy(dst + n, "a\xcc\x81", 3);
            n += 3;
        } else if (kind < 9) {
            memcpy(dst + n, "\xe4\xb8\xad\xe6\x96\x87", 6);
            n += 6;
        } else {
            memcpy(dst + n, "\xef\xac\x81", 3);
            n += 3;
        }
    }
    return n;
}

int main(void) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON, MODEL_NFC,
                                   NULL, &sch, NULL, 0) != 0) {
        fprintf(stderr, "FAIL: cannot load %s\n", MODEL_NFC);
        return 1;
    }
    if (t.norm != TSIMD_NORM_NFC) {
        fprintf(stderr, "FAIL: model is not NFC\n");
        return 1;
    }

    static uint8_t small[4096];
    size_t nsmall = gen_mixed(small, sizeof(small), 7u);
    static uint8_t big[3u << 20];
    size_t nbig = gen_mixed(big, sizeof(big), 99u);
    static uint8_t ascii_only[8192];
    memset(ascii_only, 'x', sizeof(ascii_only));

    const uint8_t *inputs[4] = {small, big, ascii_only, small};
    const size_t lens[4] = {nsmall, nbig, sizeof(ascii_only), nsmall};
    const char *names[4] = {"small", "big(3MiB)", "ascii-only", "small again"};

    uint32_t *reuse[4];
    size_t nreuse[4];
    for (int i = 0; i < 4; i++) {
        reuse[i] = encode_ids(&t, inputs[i], lens[i], &nreuse[i]);
        if (!reuse[i]) {
            fprintf(stderr, "FAIL %s: encode failed\n", names[i]);
            g_fail = 1;
            tsimd_tokenizer_free(&t);
            return 1;
        }
        tsimd_tokenizer_t fresh;
        memset(&fresh, 0, sizeof fresh);
        tsimd_pretok_scheme_t schf = TSIMD_PRETOK_GPT2;
        if (tsimd_tokenizer_load_model(&fresh, TSIMD_MODEL_TOKENIZER_JSON,
                                       MODEL_NFC, NULL, &schf, NULL, 0) != 0) {
            fprintf(stderr, "FAIL: fresh load\n");
            g_fail = 1;
            tsimd_tokenizer_free(&t);
            return 1;
        }
        size_t nf = 0;
        uint32_t *f = encode_ids(&fresh, inputs[i], lens[i], &nf);
        tsimd_tokenizer_free(&fresh);
        if (!f) {
            fprintf(stderr, "FAIL %s: fresh encode failed\n", names[i]);
            g_fail = 1;
            tsimd_tokenizer_free(&t);
            return 1;
        }
        char what[64];
        snprintf(what, sizeof what, "%s: reuse == fresh", names[i]);
        chk_ids(reuse[i], nreuse[i], f, nf, what);
        free(f);
    }

    chk_ids(reuse[0], nreuse[0], reuse[3], nreuse[3],
            "small before == small after the big encode");
    {
        uint32_t *again = encode_ids(&t, small, nsmall, &nreuse[3]);
        chk_ids(reuse[0], nreuse[0], again, nreuse[3],
                "same encode twice");
        free(again);
    }

    {
        static uint8_t out1[4u << 20];
        static uint8_t out2[4u << 20];
        size_t a = tsimd_nfc_normalize(big, nbig, out1, sizeof(out1));
        if (a == (size_t)-1) {
            fprintf(stderr, "FAIL: normalize big returned -1\n");
            g_fail = 1;
        } else {
            size_t b = tsimd_nfc_normalize(small, nsmall, out2, sizeof(out2));
            size_t c = tsimd_nfc_normalize(big, nbig, out2, sizeof(out2));
            checks++;
            if (b == (size_t)-1 || c != a || memcmp(out1, out2, a) != 0) {
                fprintf(stderr,
                        "FAIL: normalize reuse differs (%zu vs %zu)\n", a, c);
                g_fail = 1;
            }
        }
    }

    for (int i = 0; i < 4; i++) free(reuse[i]);
    tsimd_tokenizer_free(&t);
    if (g_fail) {
        fprintf(stderr, "test_nfc_reuse: FAILED (%d checks)\n", checks);
        return 1;
    }
    printf("test_nfc_reuse: OK (%d checks)\n", checks);
    return 0;
}
