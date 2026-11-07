#include "tokenizer.h"
#include "bpe.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>

#define MODEL \
    TSIMD_SRC_DIR "/models/reference/deepseek_tokenizer.json"

static int g_fail;
static int checks;

static int load_tok(tsimd_tokenizer_t *t) {
    memset(t, 0, sizeof *t);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    return tsimd_tokenizer_load_model(t, TSIMD_MODEL_TOKENIZER_JSON, MODEL, NULL,
                                      &sch, NULL, 0);
}

static uint32_t *enc(tsimd_tokenizer_t *t, const uint8_t *d, size_t n,
                     size_t *nout) {
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(t, d, n, &seq) != 0) return NULL;
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
        fprintf(stderr, "FAIL %s: %zu vs %zu ids, first diff %zu\n", what, na,
                nb, i);
        g_fail = 1;
    }
}

static void fill_text(uint8_t *dst, size_t n, unsigned seed) {
    static const char *w[] = {"the", "of", "and", "a", "to", "in", "is",
                              "you", "that", "it", "x", "1", "zz"};
    size_t o = 0;
    unsigned r = seed | 1u;
    while (o + 8 < n) {
        r = r * 1103515245u + 12345u;
        const char *p = w[(r >> 8) % 13];
        size_t l = strlen(p);
        memcpy(dst + o, p, l);
        o += l;
        dst[o++] = (r & 0x10000) ? ' ' : '\n';
    }
}

int main(void) {
    setenv("TSIMD_PCACHE_UPGRADE_MB", "1", 1);
    tsimd_tokenizer_t t;
    if (load_tok(&t) != 0) {
        fprintf(stderr, "FAIL: load\n");
        return 1;
    }
    size_t base_entries = t.cache.mask + 1;

    static uint8_t small[64];
    memcpy(small, "the quick brown fox jumps over the lazy dog 1234 zz\n", 50);
    size_t nsmall = 50;
    size_t na = 0;
    uint32_t *a = enc(&t, small, nsmall, &na);
    if (!a) {
        fprintf(stderr, "FAIL: small encode\n");
        return 1;
    }

    static uint8_t big9[2u << 20];
    fill_text(big9, sizeof(big9), 11u);
    size_t nb = 0;
    uint32_t *b = enc(&t, big9, sizeof(big9), &nb);
    if (!b) {
        fprintf(stderr, "FAIL: 2 MiB encode\n");
        return 1;
    }
    checks++;
    if (t.cache.mask + 1 < (1u << 21)) {
        fprintf(stderr, "FAIL: no upgrade after 2 MiB (entries %zu, base %zu)\n",
                t.cache.mask + 1, base_entries);
        g_fail = 1;
    }

    {
        tsimd_tokenizer_t fresh;
        if (load_tok(&fresh) != 0) {
            fprintf(stderr, "FAIL: fresh load\n");
            return 1;
        }
        size_t nf = 0;
        uint32_t *f = enc(&fresh, big9, sizeof(big9), &nf);
        tsimd_tokenizer_free(&fresh);
        if (!f) {
            fprintf(stderr, "FAIL: fresh 2 MiB encode\n");
            return 1;
        }
        chk_ids(b, nb, f, nf, "2 MiB: upgraded == fresh");
        free(f);
    }

    {
        size_t na2 = 0;
        uint32_t *a2 = enc(&t, small, nsmall, &na2);
        if (!a2) {
            fprintf(stderr, "FAIL: small after upgrade\n");
            return 1;
        }
        chk_ids(a, na, a2, na2, "small before == small after the upgrade");
        free(a2);
    }

    static uint8_t big17[2u << 20];
    fill_text(big17, sizeof(big17), 77u);
    size_t nc = 0;
    uint32_t *c = enc(&t, big17, sizeof(big17), &nc);
    if (!c) {
        fprintf(stderr, "FAIL: second 2 MiB encode\n");
        return 1;
    }
    checks++;
    if (t.cache.mask + 1 < (1u << 22)) {
        fprintf(stderr, "FAIL: no upgrade past 2x the threshold (entries %zu)\n",
                t.cache.mask + 1);
        g_fail = 1;
    }
    {
        tsimd_tokenizer_t fresh;
        if (load_tok(&fresh) != 0) {
            fprintf(stderr, "FAIL: fresh load 2\n");
            return 1;
        }
        size_t nf = 0;
        uint32_t *f = enc(&fresh, big17, sizeof(big17), &nf);
        tsimd_tokenizer_free(&fresh);
        if (!f) {
            fprintf(stderr, "FAIL: fresh second 2 MiB encode\n");
            return 1;
        }
        chk_ids(c, nc, f, nf, "second 2 MiB: upgraded == fresh");
        free(f);
    }
    {
        size_t na3 = 0;
        uint32_t *a3 = enc(&t, small, nsmall, &na3);
        if (!a3) {
            fprintf(stderr, "FAIL: small after the second upgrade\n");
            return 1;
        }
        chk_ids(a, na, a3, na3, "small before == small after the second upgrade");
        free(a3);
    }

    free(a);
    free(b);
    free(c);
    tsimd_tokenizer_free(&t);
    if (g_fail) {
        fprintf(stderr, "test_pcache_upgrade: FAILED (%d checks)\n", checks);
        return 1;
    }
    printf("test_pcache_upgrade: OK (%d checks)\n", checks);
    return 0;
}
