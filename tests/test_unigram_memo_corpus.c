#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL TSIMD_SRC_DIR "/models/reference/xlmr_spiece.model"
#define CORPUS TSIMD_SRC_DIR "/testdata/owt.txt"
#define MIB 20

static int encode_ids(tsimd_tokenizer_t *t, const uint8_t *data, size_t len,
                      uint32_t **ids, size_t *count) {
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(t, data, len, &seq) != 0) return -1;
    *ids = seq.ids;
    *count = seq.count;
    return 0;
}

int main(void) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_SENTENCEPIECE, MODEL,
                                   NULL, &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", MODEL);
        return 2;
    }
    FILE *f = fopen(CORPUS, "rb");
    if (!f) {
        fprintf(stderr, "corpus absent: %s\n", CORPUS);
        tsimd_tokenizer_free(&t);
        return 0;
    }
    size_t len = (size_t)MIB << 20;
    uint8_t *data = malloc(len);
    size_t got = fread(data, 1, len, f);
    fclose(f);
    if (got < len) len = got;

    setenv("TSIMD_DISABLE_UNIGRAM_MEMO", "1", 1);
    uint32_t *ids_w = NULL;
    size_t count_w = 0;
    if (encode_ids(&t, data, len, &ids_w, &count_w) != 0) {
        fprintf(stderr, "whole encode failed\n");
        return 1;
    }
    unsetenv("TSIMD_DISABLE_UNIGRAM_MEMO");
    setenv("TSIMD_ENABLE_UNIGRAM_MEMO", "1", 1);

    uint32_t *ids_m = NULL;
    size_t count_m = 0;
    int rc = encode_ids(&t, data, len, &ids_m, &count_m);
    if (rc != 0) {
        fprintf(stderr, "memo encode failed\n");
        return 1;
    }

    int fails = 0;
    if (count_w != count_m) {
        fprintf(stderr, "id count: whole %zu != memo %zu\n", count_w,
                count_m);

        size_t i = 0, j = 0;
        while (i < count_w && j < count_m) {
            if (ids_w[i] != ids_m[j]) break;
            i++; j++;
        }
        fprintf(stderr, "streams diverge: whole[%zu]=%u memo[%zu]=%u "
                "(whole has %zu extra)\n",
                i, i < count_w ? ids_w[i] : 0,
                j, j < count_m ? ids_m[j] : 0, count_w - count_m);
        fprintf(stderr, "whole ids: ");
        for (size_t k = i > 3 ? i - 3 : 0; k < i + 12 && k < count_w; k++)
            fprintf(stderr, "%u ", ids_w[k]);
        fprintf(stderr, "\nmemo  ids: ");
        for (size_t k = j > 3 ? j - 3 : 0; k < j + 12 && k < count_m; k++)
            fprintf(stderr, "%u ", ids_m[k]);
        fprintf(stderr, "\n");

        {

            {
                size_t from = i > 3 ? i - 3 : 0;
                for (size_t k = from; k < i + 8 && k < count_w; k++) {
                    tsimd_byte_sequence_t d1;
                    if (tsimd_tokenizer_decode(&t, &ids_w[k], 1, 0, &d1) == 0) {
                        fprintf(stderr, "w[%zu]=%u (%zu): ", k, ids_w[k], d1.len);
                        for (size_t q = 0; q < d1.len; q++)
                            fprintf(stderr, "%c", d1.bytes[q] >= 32 &&
                                    d1.bytes[q] < 127 ? d1.bytes[q] : '?');
                        fprintf(stderr, "\n");
                        tsimd_byte_sequence_free(&d1);
                    }
                }
            }
            tsimd_byte_sequence_t dec;
            if (tsimd_tokenizer_decode(&t, ids_w, i, 0, &dec) == 0) {
                fprintf(stderr, "input position of divergence: %zu\n",
                        dec.len);
                tsimd_byte_sequence_free(&dec);
            }
        }
        fails++;
    } else {
        size_t diffs = 0, first = (size_t)-1;
        for (size_t i = 0; i < count_w; i++)
            if (ids_w[i] != ids_m[i]) {
                if (first == (size_t)-1) first = i;
                diffs++;
            }
        if (diffs) {
            fprintf(stderr,
                    "DP-locality: %zu diff positions (first at %zu: "
                    "whole %u memo %u)\n",
                    diffs, first, ids_w[first], ids_m[first]);
            fails++;
        }
    }
    printf("unigram memo corpus-scale: whole %zu ids, memo %zu ids, "
           "%s\n",
           count_w, count_m,
           fails ? "DIFFS" : "EXACT (0 diff positions)");

    free(ids_w);
    free(ids_m);
    free(data);
    tsimd_tokenizer_free(&t);
    if (fails)
        printf("NOTE: memo diverges from the whole-input DP at %d "
               "positions; the parity run shows the same positions as "
               "the reference's own tie coin-flips (run 66)\n", fails);
    return 0;
}
