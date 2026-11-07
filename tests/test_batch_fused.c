#include "tokenizer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check_model(const char *name, int fmt, const char *m1,
                       const char *m2, tsimd_pretok_scheme_t scheme) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t det = scheme;
    if (tsimd_tokenizer_load_model(&t, fmt, m1, m2, &det, NULL, 0) != 0) {
        fprintf(stderr, "load failed: %s\n", name);
        return 2;
    }

    size_t len = (size_t)1 << 20;
    uint8_t *data = malloc(len);
    if (!data) return 2;
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < len; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        data[i] = (uint8_t)(" The quick brown fox, it's 12:30 AM.\n"
                            "naive resume 42x times!\r\n\t "
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"[rng % 96]);
    }

    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(&t, data, len, &seq) != 0) {
        fprintf(stderr, "sequential encode failed\n");
        return 1;
    }

    tsimd_input_span_t docs[1] = {{data, len}};
    tsimd_token_batch_t batch;
    if (tsimd_tokenizer_encode_batch(&t, docs, 1, &batch, 1) != 0) {
        fprintf(stderr, "batch encode failed\n");
        return 1;
    }
    int fails = 0;
    if (batch.id_count != seq.count) {
        fprintf(stderr, "%s: id count batch %zu != seq %zu\n", name,
                batch.id_count, seq.count);
        fails++;
    } else if (seq.count &&
               memcmp(batch.ids, seq.ids, seq.count * sizeof(uint32_t)) != 0) {
        for (size_t i = 0; i < seq.count; i++)
            if (batch.ids[i] != seq.ids[i]) {
                fprintf(stderr, "%s: first id diff at %zu: batch %u != seq %u\n",
                        name, i, batch.ids[i], seq.ids[i]);
                break;
            }
        fails++;
    }
    printf("%s: batch-shaped 1t %zu ids vs sequential %zu: %s\n", name,
           batch.id_count, seq.count, fails ? "DIFFS" : "OK");
    free(data);
    tsimd_token_batch_free(&batch);
    tsimd_token_sequence_free(&seq);
    tsimd_tokenizer_free(&t);
    return fails;
}

int main(void) {
    int fails = 0;
    fails += check_model("cl100k", TSIMD_MODEL_TIKTOKEN,
                         TSIMD_SRC_DIR "/models/cl100k_base.tiktoken",
                         NULL, TSIMD_PRETOK_CL100K);
    fails += check_model("o200k", TSIMD_MODEL_TIKTOKEN,
                         TSIMD_SRC_DIR "/models/o200k_base.tiktoken",
                         NULL, TSIMD_PRETOK_O200K);
    fails += check_model("deepseek", TSIMD_MODEL_TOKENIZER_JSON,
                         TSIMD_SRC_DIR "/models/reference/deepseek_tokenizer.json",
                         NULL, TSIMD_PRETOK_GPT2);
    fails += check_model("nemotron", TSIMD_MODEL_TOKENIZER_JSON,
                         TSIMD_SRC_DIR "/models/reference/nemotron_tokenizer.json",
                         NULL, TSIMD_PRETOK_GPT2);
    return fails ? 1 : 0;
}
