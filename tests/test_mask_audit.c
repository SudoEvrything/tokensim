#include "pretokenizer.h"
#include "tokenizer.h"

size_t tsimd_match_kimi(const uint8_t *d, size_t n, size_t i);
int tsimd_pretokenize_kimi_masks_ex(const uint8_t *data, size_t len,
                                   tsimd_span_fn fn, void *ud);

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KIMI TSIMD_SRC_DIR "/models/reference/kimi_k2_tiktoken.model"
#define CL100K TSIMD_SRC_DIR "/models/cl100k_base.tiktoken"

static int compare_ids(const char *what, tsimd_tokenizer_t *t,
                       const uint8_t *text, size_t n) {
    tsimd_token_sequence_t seq, par;
    memset(&seq, 0, sizeof seq);
    memset(&par, 0, sizeof par);
    if (tsimd_tokenizer_encode(t, text, n, &seq) != 0) {
        fprintf(stderr, "%s: serial encode failed\n", what);
        return 1;
    }
    if (tsimd_tokenizer_encode_parallel(t, text, n, 6, &par) != 0) {
        fprintf(stderr, "%s: parallel encode failed\n", what);
        tsimd_token_sequence_free(&seq);
        return 1;
    }
    int bad = seq.count != par.count ||
              memcmp(seq.ids, par.ids, seq.count * 4) != 0;
    if (bad)
        fprintf(stderr,
                "%s: serial %zu ids vs parallel %zu ids (divergence)\n",
                what, seq.count, par.count);
    tsimd_token_sequence_free(&seq);
    tsimd_token_sequence_free(&par);
    return bad;
}

#define MAXSP 4096

typedef struct {
    size_t starts[MAXSP];
    size_t lens[MAXSP];
    size_t n;
} mspans_t;

static int mask_collect(void *ud, size_t off, size_t end) {
    mspans_t *m = ud;
    if (m->n < MAXSP) {
        m->starts[m->n] = off;
        m->lens[m->n] = end - off;
        m->n++;
    }
    return 0;
}

static int kimi_mask_span_check(void) {
    size_t cap = 400u * 1024u;
    unsigned char *buf = malloc(cap);
    if (!buf) return 1;
    size_t n = 0;
    for (int i = 0; i < 150000; i++) buf[n++] = 'y';
    buf[n++] = '1';
    buf[n++] = '2';
    buf[n++] = 0xD9;
    buf[n++] = 0xA3;
    for (int i = 0; i < 150000; i++) buf[n++] = 'y';
    mspans_t mask;
    memset(&mask, 0, sizeof mask);
    if (tsimd_pretokenize_kimi_masks_ex(buf, n, mask_collect, &mask) != 0) {
        fprintf(stderr, "kimi mask path failed\n");
        free(buf);
        return 1;
    }
    size_t i = 0;
    int bad = 0;
    while (i < n) {
        size_t e = tsimd_match_kimi(buf, n, i);
        if (bad == 0 && mask.n > 0 && i != mask.starts[0]) {
            fprintf(stderr,
                    "kimi mask span %zu: mask [%zu,%zu) vs scalar [%zu,%zu)"
                    " (the mixed-width digit run)\n",
                    0, mask.starts[0], mask.starts[0] + mask.lens[0], i, e);
            bad = 1;
        }
        if (mask.n > 0) {
            for (size_t k = 0; k + 1 < mask.n; k++) {
                mask.starts[k] = mask.starts[k + 1];
                mask.lens[k] = mask.lens[k + 1];
            }
            mask.n--;
        }
        i = e;
    }
    free(buf);
    if (bad) return 1;
    printf("kimi mask spans match the scalar on the mixed-width digit run\n");
    return 0;
}

int main(void) {
    setenv("TSIMD_KIMI_MASKS", "1", 1);
    int bad = kimi_mask_span_check();
    unsigned char *text = malloc(400u * 1024u);
    size_t n = 0;
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;

    {
        tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
        tsimd_pretok_scheme_t ki = TSIMD_PRETOK_KIMI;
        if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TIKTOKEN, KIMI, NULL,
                                       &ki, NULL, 0) != 0) {
            fprintf(stderr, "cannot load kimi\n");
            return 1;
        }
        n = 0;
        for (int i = 0; i < 150000; i++) text[n++] = 'y';
        text[n++] = '1';
        text[n++] = '2';
        text[n++] = 0xD9;
        text[n++] = 0xA3;
        for (int i = 0; i < 150000; i++) text[n++] = 'y';
        bad += compare_ids("kimi mixed-width digits", &t, text, n);
        tsimd_tokenizer_free(&t);
    }

    {
        tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
        tsimd_pretok_scheme_t cl = TSIMD_PRETOK_CL100K;
        if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TIKTOKEN, CL100K,
                                       NULL, &cl, NULL, 0) != 0) {
            fprintf(stderr, "cannot load cl100k\n");
            return 1;
        }
        n = 0;
        for (int i = 0; i < 150000; i++) text[n++] = 'a';
        text[n++] = 0xD9;
        text[n++] = 0xA3;
        for (int i = 0; i < 150000; i++) text[n++] = '2';
        text[n++] = 'x';
        bad += compare_ids("cl100k all-digit window", &t, text, n);
        tsimd_tokenizer_free(&t);
    }
    bad += kimi_mask_span_check();

    free(text);
    bad += kimi_mask_span_check();
    if (bad) {
        fprintf(stderr, "mask-audit divergences: %d\n", bad);
        return 1;
    }
    printf("test_mask_audit: the kimi digit and cl100k window paths match "
           "the serial encodes\n");
    return 0;
}
