#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer.h"

static tsimd_tokenizer_t tok;
static int ready = 0;

static void fail_dump(const uint8_t *d, size_t n, size_t reps) {
    fprintf(stderr, "PAR-MISMATCH size=%zu reps=%zu head:", n, reps);
    for (size_t i = 0; i < n && i < 64; i++) fprintf(stderr, " %02x", d[i]);
    fprintf(stderr, "\n");
}

static tsimd_pretok_scheme_t scheme_of(const char *s) {
    if (strcmp(s, "o200k") == 0) return TSIMD_PRETOK_O200K;
    if (strcmp(s, "cl100k") == 0) return TSIMD_PRETOK_CL100K;
    if (strcmp(s, "kimi") == 0) return TSIMD_PRETOK_KIMI;
    if (strcmp(s, "gem4") == 0) return TSIMD_PRETOK_GEM4;
    return TSIMD_PRETOK_GPT2;
}

int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc;
    (void)argv;
    const char *spec = getenv("TSIMD_FUZZ_SPEC");
    if (!spec) return 0;
    char path[512];
    strncpy(path, spec, sizeof path - 1);
    path[sizeof path - 1] = 0;
    char *semi = strchr(path, ';');
    if (!semi) return 0;
    *semi = 0;
    const char *kind = semi + 1;
    char *semi2 = strchr(semi + 1, ';');
    const char *scheme = NULL;
    if (semi2) {
        *semi2 = 0;
        scheme = semi2 + 1;
    }
    memset(&tok, 0, sizeof tok);
    tsimd_model_format_t fmt = strcmp(kind, "spm") == 0
                                   ? TSIMD_MODEL_SENTENCEPIECE
                                   : strcmp(kind, "tiktoken") == 0
                                         ? TSIMD_MODEL_TIKTOKEN
                                         : TSIMD_MODEL_TOKENIZER_JSON;
    tsimd_pretok_scheme_t sch = scheme ? scheme_of(scheme)
                                       : TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&tok, fmt, path, NULL, &sch, NULL,
                                   0) != 0) {
        fprintf(stderr, "PAR-FUZZ: model load failed: %s (%s)\n", path,
                kind);
        return 0;
    }
    ready = 1;
    return 0;
}

static int encode_cmp(const uint8_t *d, size_t n, unsigned th,
                      const uint32_t *ref, size_t ref_n, size_t reps,
                      const uint8_t *orig, size_t osize) {
    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    if (tsimd_tokenizer_encode_parallel(&tok, d, n, th, &seq) != 0) {
        fprintf(stderr, "PAR-ERROR th=%u\n", th);
        fail_dump(orig, osize, reps);
        return 1;
    }
    int bad = 0;
    if (seq.count != ref_n) {
        fprintf(stderr, "PAR-COUNT th=%u par=%zu seq=%zu\n", th, seq.count,
                ref_n);
        bad = 1;
    } else if (ref_n && memcmp(seq.ids, ref, ref_n * 4) != 0) {
        size_t i = 0;
        while (i < ref_n && seq.ids[i] == ref[i]) i++;
        fprintf(stderr, "PAR-ID th=%u at %zu: %u vs %u\n", th, i,
                i < seq.count ? seq.ids[i] : 0, i < ref_n ? ref[i] : 0);
        bad = 1;
    }
    if (bad) fail_dump(orig, osize, reps);
    tsimd_token_sequence_free(&seq);
    return bad;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (!ready || size < 8) return 0;
    size_t min_reps = ((1u << 18) + size - 1) / size;
    size_t max_reps = (8u << 20) / size;
    if (max_reps <= min_reps) return 0;
    size_t span = max_reps - min_reps + 1;
    size_t reps = min_reps + (((size_t)data[0] << 8 | data[1]) % span);
    size_t total = size * reps;

    uint8_t *d = malloc(total);
    if (!d) return 0;
    for (size_t i = 0; i < reps; i++) memcpy(d + i * size, data, size);

    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    if (tsimd_tokenizer_encode(&tok, d, total, &seq) != 0) {
        if (getenv("TSIMD_FUZZ_DBG"))
            fprintf(stderr, "PAR-SEQ-ERR total=%zu\n", total);
        free(d);
        return 0;
    }
    uint32_t *ref = NULL;
    size_t ref_n = 0;
    if (seq.count) {
        ref = malloc(seq.count * 4);
        if (!ref) {
            tsimd_token_sequence_free(&seq);
            free(d);
            return 0;
        }
        memcpy(ref, seq.ids, seq.count * 4);
        ref_n = seq.count;
    }
    tsimd_token_sequence_free(&seq);

    int bad = encode_cmp(d, total, 3, ref, ref_n, reps, data, size);
    bad |= encode_cmp(d, total, 6, ref, ref_n, reps, data, size);
    bad |= encode_cmp(d, total, 16, ref, ref_n, reps, data, size);

    free(ref);
    free(d);
    if (bad) abort();
    return 0;
}
