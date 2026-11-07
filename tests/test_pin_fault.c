#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL TSIMD_SRC_DIR "/models/o200k_base.tiktoken"

int main(void) {
    setenv("TSIMD_PIN_FAULT", "1", 1);
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_O200K;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TIKTOKEN, MODEL, NULL,
                                   &sch, NULL, 0) != 0) {
        fprintf(stderr, "load failed\n");
        return 2;
    }
    size_t cap = 1u << 20;
    uint8_t *data = malloc(cap);
    if (!data) return 2;
    for (size_t i = 0; i < cap; i++)
        data[i] = (uint8_t)(0x20 + i % 95);
    tsimd_token_sequence_t ref;
    memset(&ref, 0, sizeof ref);
    if (tsimd_tokenizer_encode(&t, data, cap, &ref) != 0) {
        fprintf(stderr, "seq encode failed\n");
        return 2;
    }
    for (int th = 2; th <= 6; th += 2) {
        tsimd_token_sequence_t seq;
        memset(&seq, 0, sizeof seq);
        if (tsimd_tokenizer_encode_parallel(&t, data, cap, th, &seq) != 0) {
            fprintf(stderr, "FAIL: par encode th=%d\n", th);
            return 1;
        }
        if (seq.count != ref.count ||
            memcmp(seq.ids, ref.ids, ref.count * sizeof(uint32_t)) != 0) {
            fprintf(stderr, "FAIL: ids th=%d\n", th);
            return 1;
        }
        tsimd_token_sequence_free(&seq);
    }
    tsimd_token_sequence_free(&ref);
    free(data);
    tsimd_tokenizer_free(&t);
    printf("pin_fault: OK\n");
    return 0;
}
