#include "tokenizer.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CL100K TSIMD_SRC_DIR "/models/cl100k_base.tiktoken"

#define ROUNDS 8

typedef struct {
    tsimd_tokenizer_t *t;
    const uint32_t *ids;
    size_t n;
    const uint8_t *want;
    size_t want_len;
    int fails;
} dec_arg_t;

static void *dec_thread(void *arg) {
    dec_arg_t *a = arg;
    for (int r = 0; r < ROUNDS; r++) {
        tsimd_byte_sequence_t out;
        memset(&out, 0, sizeof out);
        if (tsimd_tokenizer_decode(a->t, a->ids, a->n, 0, &out) != 0) {
            a->fails++;
            continue;
        }
        if (out.len != a->want_len ||
            memcmp(out.bytes, a->want, a->want_len) != 0)
            a->fails++;
        tsimd_byte_sequence_free(&out);
    }
    return NULL;
}

static size_t fill_pattern(uint8_t *buf, size_t len, unsigned seed) {
    const char *pat_a = "The quick brown fox jumps over the lazy dog. ";
    const char *pat_b = "Machine learning models are trained on corpora. ";
    const char *pat = seed & 1 ? pat_b : pat_a;
    size_t plen = strlen(pat);
    for (size_t i = 0; i < len; i++)
        buf[i] = (uint8_t)pat[(i + seed * 7) % plen];
    return len;
}

int main(void) {
    tsimd_tokenizer_t t1, t2;
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_CL100K;
    if (tsimd_tokenizer_load_model(&t1, TSIMD_MODEL_TIKTOKEN, CL100K, NULL,
                                   &sch, NULL, 0) != 0 ||
        tsimd_tokenizer_load_model(&t2, TSIMD_MODEL_TIKTOKEN, CL100K, NULL,
                                   &sch, NULL, 0) != 0) {
        fprintf(stderr, "cannot load cl100k\n");
        return 1;
    }
    size_t len = 6u << 20;
    uint8_t *text1 = malloc(len);
    uint8_t *text2 = malloc(len);
    fill_pattern(text1, len, 0);
    fill_pattern(text2, len, 1);
    tsimd_token_sequence_t s1, s2;
    if (tsimd_tokenizer_encode_parallel(&t1, text1, len, 4, &s1) != 0 ||
        tsimd_tokenizer_encode_parallel(&t2, text2, len, 4, &s2) != 0) {
        fprintf(stderr, "encode failed\n");
        return 1;
    }
    if (s1.count < ((size_t)1 << 20) || s2.count < ((size_t)1 << 20)) {
        fprintf(stderr, "inputs too small for the parallel path\n");
        return 1;
    }
    tsimd_byte_sequence_t w1, w2;
    if (tsimd_tokenizer_decode_into(&t1, s1.ids, s1.count, 0, NULL, 0,
                                    &w1.len) != 0 ||
        tsimd_tokenizer_decode_into(&t2, s2.ids, s2.count, 0, NULL, 0,
                                    &w2.len) != 0) {
        fprintf(stderr, "count failed\n");
        return 1;
    }
    w1.bytes = malloc(w1.len);
    w2.bytes = malloc(w2.len);
    size_t g1 = 0, g2 = 0;
    if (tsimd_tokenizer_decode_into(&t1, s1.ids, s1.count, 0, w1.bytes,
                                    w1.len, &g1) != 0 ||
        g1 != w1.len ||
        tsimd_tokenizer_decode_into(&t2, s2.ids, s2.count, 0, w2.bytes,
                                    w2.len, &g2) != 0 ||
        g2 != w2.len) {
        fprintf(stderr, "reference decode failed\n");
        return 1;
    }
    if (w1.len != len || memcmp(w1.bytes, text1, len) != 0 ||
        w2.len != len || memcmp(w2.bytes, text2, len) != 0) {
        fprintf(stderr, "serial round trip broken before the test\n");
        return 1;
    }
    pthread_t th1, th2;
    dec_arg_t a1 = {&t1, s1.ids, s1.count, w1.bytes, w1.len, 0};
    dec_arg_t a2 = {&t2, s2.ids, s2.count, w2.bytes, w2.len, 0};
    pthread_create(&th1, NULL, dec_thread, &a1);
    pthread_create(&th2, NULL, dec_thread, &a2);
    pthread_join(th1, NULL);
    pthread_join(th2, NULL);
    int fails = a1.fails + a2.fails;
    tsimd_token_sequence_free(&s1);
    tsimd_token_sequence_free(&s2);
    free(w1.bytes);
    free(w2.bytes);
    free(text1);
    free(text2);
    tsimd_tokenizer_free(&t1);
    tsimd_tokenizer_free(&t2);
    if (fails) {
        fprintf(stderr, "%d concurrent decode mismatches\n", fails);
        return 1;
    }
    printf("test_decode_concurrent: two tokenizers decode concurrently, "
           "byte-exact every round\n");
    return 0;
}
