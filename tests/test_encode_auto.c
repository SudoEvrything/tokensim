#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CL100K TSIMD_SRC_DIR "/models/cl100k_base.tiktoken"

int main(void) {
    tsimd_tokenizer_t t;
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_CL100K;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TIKTOKEN, CL100K, NULL,
                                   &sch, NULL, 0) != 0) {
        fprintf(stderr, "cannot load cl100k\n");
        return 1;
    }
    size_t len = 6u << 20;
    uint8_t *buf = malloc(len);
    if (!buf) return 1;
    for (size_t i = 0; i < len; i++)
        buf[i] = (uint8_t)("The quick brown fox. "[i % 21]);

    tsimd_token_sequence_t a, b, c;
    memset(&a, 0, sizeof a);
    memset(&b, 0, sizeof b);
    memset(&c, 0, sizeof c);
    if (tsimd_tokenizer_encode_auto(&t, buf, len, &a) != 0) {
        fprintf(stderr, "encode_auto failed\n");
        return 1;
    }
    if (tsimd_tokenizer_encode_auto(&t, buf, len, &b) != 0) {
        fprintf(stderr, "encode_auto repeat failed\n");
        return 1;
    }
    if (tsimd_tokenizer_encode_parallel(&t, buf, len, 6, &c) != 0) {
        fprintf(stderr, "encode_parallel failed\n");
        return 1;
    }
    if (a.count != b.count || a.count != c.count ||
        memcmp(a.ids, b.ids, a.count * 4) != 0 ||
        memcmp(a.ids, c.ids, c.count * 4) != 0) {
        fprintf(stderr, "encode_auto ids differ from encode_parallel\n");
        return 1;
    }
    tsimd_byte_sequence_t d1, d2;
    memset(&d1, 0, sizeof d1);
    memset(&d2, 0, sizeof d2);
    if (tsimd_tokenizer_decode(&t, a.ids, a.count, 0, &d1) != 0 ||
        d1.len != len || memcmp(d1.bytes, buf, len) != 0) {
        fprintf(stderr, "auto round trip broken\n");
        return 1;
    }
    tsimd_byte_sequence_free(&d1);
    tsimd_byte_sequence_free(&d2);
    tsimd_token_sequence_free(&a);
    tsimd_token_sequence_free(&b);
    tsimd_token_sequence_free(&c);
    free(buf);
    tsimd_tokenizer_free(&t);
    printf("test_encode_auto: the default entry matches the parallel "
           "encode and round trips\n");
    return 0;
}
