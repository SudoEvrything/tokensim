#include "bpe.h"
#include "pretokenizer.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define G3 TSIMD_SRC_DIR "/models/reference/gemma3-4b_tokenizer.json"
#define XLMR TSIMD_SRC_DIR "/models/reference/xlmr_spiece.model"
#define FIXDIR_ARG (argc > 1 ? argv[1] : TSIMD_SRC_DIR "/tests/fixtures")

static int seqs_equal(const tsimd_token_sequence_t *a,
                      const tsimd_token_sequence_t *b) {
    return a->count == b->count &&
           (a->count == 0 ||
            memcmp(a->ids, b->ids, a->count * sizeof(uint32_t)) == 0);
}

static int fail_ids(const char *label, const tsimd_vocab_t *v,
                    const tsimd_token_sequence_t *got,
                    const uint32_t *want, size_t n) {
    fprintf(stderr, "FAIL: %s: %zu ids want %zu\n", label,
            (size_t)got->count, n);
    for (size_t i = 0; i < got->count && i < n; i++) {
        if (got->ids[i] != want[i]) {
            uint32_t gl, wl;
            const uint8_t *gb = tsimd_vocab_bytes(v, got->ids[i], &gl);
            const uint8_t *wb = tsimd_vocab_bytes(v, want[i], &wl);
            fprintf(stderr, "  id %zu: got %u '", i, got->ids[i]);
            for (uint32_t k = 0; k < gl; k++)
                fputc(gb[k] >= 0x20 && gb[k] < 0x7f ? gb[k] : '?', stderr);
            fprintf(stderr, "' want %u '", want[i]);
            for (uint32_t k = 0; k < wl; k++)
                fputc(wb[k] >= 0x20 && wb[k] < 0x7f ? wb[k] : '?', stderr);
            fprintf(stderr, "'\n");
            break;
        }
    }
    return 1;
}

static int case_run(tsimd_tokenizer_t *t, const char *label,
                    const uint8_t *text, size_t len, const uint32_t *want,
                    size_t n, int threads) {
    tsimd_token_sequence_t seq, par;
    memset(&seq, 0, sizeof seq);
    memset(&par, 0, sizeof par);
    if (tsimd_tokenizer_encode(t, text, len, &seq) != 0) {
        fprintf(stderr, "FAIL: %s: encode error\n", label);
        return 1;
    }
    if (tsimd_tokenizer_encode_whole_chain_par(t, text, len, threads, &par) !=
        0) {
        fprintf(stderr, "FAIL: %s: whole_chain error\n", label);
        tsimd_token_sequence_free(&seq);
        return 1;
    }
    int rc = 0;
    if (want && (seq.count != n || memcmp(seq.ids, want, n * sizeof(uint32_t))))
        rc |= fail_ids(label, &t->vocab, &seq, want, n);
    if (!seqs_equal(&seq, &par)) {
        fprintf(stderr, "FAIL: %s: whole_chain(%dt) != sequential\n", label,
                threads);
        rc |= fail_ids(label, &t->vocab, &par, seq.ids, seq.count);
    }
    tsimd_token_sequence_free(&seq);
    tsimd_token_sequence_free(&par);
    return rc;
}

static int fixture_run(tsimd_tokenizer_t *t, const char *fixdir, size_t *count_out) {
    char path[512];
    snprintf(path, sizeof path, "%s/gemma3_4b_chunk6_seam.bin", fixdir);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "FAIL: open %s\n", path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long flen = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = malloc((size_t)flen);
    if (!data || fread(data, 1, (size_t)flen, f) != (size_t)flen) {
        fprintf(stderr, "FAIL: read fixture\n");
        return 1;
    }
    fclose(f);
    tsimd_token_sequence_t s1, s6, p1, p6;
    memset(&s1, 0, sizeof s1);
    memset(&s6, 0, sizeof s6);
    memset(&p1, 0, sizeof p1);
    memset(&p6, 0, sizeof p6);
    int rc = 0;
    if (tsimd_tokenizer_encode(t, data, (size_t)flen, &s1) != 0 ||
        tsimd_tokenizer_encode_whole_chain_par(t, data, (size_t)flen, 6, &p6) !=
            0 ||
        tsimd_tokenizer_encode_parallel(t, data, (size_t)flen, 1, &p1) != 0 ||
        tsimd_tokenizer_encode_parallel(t, data, (size_t)flen, 6, &s6) != 0) {
        fprintf(stderr, "FAIL: fixture encode error\n");
        free(data);
        return 1;
    }
    if (!seqs_equal(&s1, &p6)) {
        fprintf(stderr,
                "FAIL: fixture whole_chain(6t) != sequential (%zu vs %zu)\n",
                (size_t)p6.count, (size_t)s1.count);
        rc = 1;
    }
    if (!seqs_equal(&s1, &s6)) {
        fprintf(stderr,
                "FAIL: fixture encode_parallel(6t) != encode_parallel(1t) "
                "(%zu vs %zu)\n",
                (size_t)s6.count, (size_t)p1.count);
        rc = 1;
    }
    if (!seqs_equal(&s1, &p1)) {
        fprintf(stderr, "FAIL: fixture encode_parallel(1t) != sequential\n");
        rc = 1;
    }
    printf("fixture: n=%zu (reference 229661)\n", (size_t)s1.count);
    if (s1.count != 229661) {
        fprintf(stderr, "FAIL: fixture count drifted from the recorded "
                        "sequential reference\n");
        rc = 1;
    }
    *count_out = s1.count;
    tsimd_token_sequence_free(&s1);
    tsimd_token_sequence_free(&s6);
    tsimd_token_sequence_free(&p1);
    tsimd_token_sequence_free(&p6);
    free(data);
    return rc;
}

int main(int argc, char **argv) {
    const char *fixdir = FIXDIR_ARG;
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON, G3, NULL,
                                   &sch, NULL, 0) != 0) {
        fprintf(stderr, "load gemma3-4b failed\n");
        return 2;
    }
    int rc = 0;
    rc |= case_run(&t, "ws-html-head", (const uint8_t *)" <html> <head>", 14,
                   (const uint32_t[]){236743, 217, 655, 2834, 236813}, 5, 6);
    rc |= case_run(&t, "html-adjacent", (const uint8_t *)"<html></html> x", 15,
                   NULL, 0, 6);
    rc |= case_run(&t, "pad-lead", (const uint8_t *)"<pad> lead", 10, NULL, 0,
                   6);
    rc |= case_run(&t, "eos-tail", (const uint8_t *)"tail <eos>", 10, NULL, 0,
                   6);
    rc |= case_run(&t, "no-added", (const uint8_t *)"plain text only", 15, NULL,
                   0, 6);
    rc |= case_run(&t, "midword-html", (const uint8_t *)"abc<html>def", 11, NULL,
                   0, 6);
    rc |= case_run(&t, "html-repeat", (const uint8_t *)"a <html> b </html> c ",
                    21, NULL, 0, 6);
    rc |= fixture_run(&t, fixdir, &(size_t){0});
    tsimd_tokenizer_free(&t);

    tsimd_tokenizer_t x;
    memset(&x, 0, sizeof x);
    tsimd_pretok_scheme_t sch2 = TSIMD_PRETOK_SPM;
    if (tsimd_tokenizer_load_model(&x, TSIMD_MODEL_SENTENCEPIECE, XLMR, NULL,
                                   &sch2, NULL, 0) != 0) {
        fprintf(stderr, "load xlmr failed\n");
        return 2;
    }
    printf("xlmr bare added_count=%zu\n", x.added_count);
    tsimd_tokenizer_free(&x);
    memset(&x, 0, sizeof x);
    const tsimd_added_token_t xextra[] = {
        {"<s>", 0, 3, 1},
        {"</s>", 2, 4, 1},
    };
    if (tsimd_tokenizer_load_model(&x, TSIMD_MODEL_SENTENCEPIECE, XLMR, NULL,
                                   &sch2, xextra, 2) != 0) {
        fprintf(stderr, "load xlmr+extra failed\n");
        return 2;
    }
    printf("xlmr extra added_count=%zu\n", x.added_count);
    if (x.added_count >= 2) {
        rc |= case_run(&x, "xlmr-mid-special",
                       (const uint8_t *)"hello<s>world</s>tail", 21, NULL, 0,
                       6);
        rc |= case_run(&x, "xlmr-lead-special", (const uint8_t *)"<s>lead", 7,
                       NULL, 0, 6);
        rc |= case_run(&x, "xlmr-adjacent", (const uint8_t *)"<s></s>", 7, NULL,
                       0, 6);
    }
    tsimd_tokenizer_free(&x);
    if (rc) return 1;
    printf("test_gemma_added_par: OK\n");
    return 0;
}
