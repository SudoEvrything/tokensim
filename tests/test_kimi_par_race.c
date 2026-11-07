#include "bpe.h"
#include "pretokenizer.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KIMI TSIMD_SRC_DIR "/models/reference/kimi_k2_tiktoken.model"
#define OWT TSIMD_SRC_DIR "/testdata/owt.txt"

#define RUNS 16

int main(void) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_KIMI;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TIKTOKEN, KIMI, NULL,
                                   &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed\n");
        return 2;
    }
    FILE *f = fopen(OWT, "rb");
    if (!f) {
        fprintf(stderr, "open " OWT "\n");
        return 2;
    }
    size_t cap = 1u << 20;
    uint8_t *data = malloc(cap);
    if (!data || fread(data, 1, cap, f) != cap) {
        fprintf(stderr, "read corpus\n");
        return 2;
    }
    fclose(f);

    tsimd_token_sequence_t ref;
    memset(&ref, 0, sizeof ref);
    if (tsimd_tokenizer_encode(&t, data, cap, &ref) != 0) {
        fprintf(stderr, "sequential encode failed\n");
        return 2;
    }
    printf("reference n=%zu\n", (size_t)ref.count);
    int rc = 0;
    for (int r = 0; r < RUNS; r++) {
        tsimd_token_sequence_t seq;
        memset(&seq, 0, sizeof seq);
        if (tsimd_tokenizer_encode_parallel(&t, data, cap, 6, &seq) != 0) {
            fprintf(stderr, "FAIL: run %d: encode error\n", r);
            rc = 1;
            break;
        }
        if (seq.count != ref.count) {
            fprintf(stderr, "FAIL: run %d: count %zu want %zu\n", r,
                    (size_t)seq.count, (size_t)ref.count);
            rc = 1;
        } else if (memcmp(seq.ids, ref.ids, ref.count * sizeof(uint32_t))) {
            size_t k = 0;
            while (k < ref.count && seq.ids[k] == ref.ids[k]) k++;
            fprintf(stderr, "FAIL: run %d: first id diff at %zu\n", r, k);
            rc = 1;
        }
        tsimd_token_sequence_free(&seq);
        if (rc) break;
    }
    tsimd_token_sequence_free(&ref);
    free(data);
    tsimd_tokenizer_free(&t);
    if (rc) return 1;
    printf("test_kimi_par_race: OK (%d runs deterministic)\n", RUNS);
    return 0;
}
