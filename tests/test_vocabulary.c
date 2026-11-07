#include "tokenizer.h"
#include "vocabulary.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *EXTRAS[] = {"ab", "abc", "abcd", " x"};

static int load_test_vocab(tsimd_vocab_t *v) {
    size_t len;
    char *buf = tsimd_test_make_vocab(EXTRAS, 4, &len);
    if (!buf) return -1;
    int rc = tsimd_vocab_load_buffer((const uint8_t *)buf, len, v);
    free(buf);
    return rc;
}

static uint64_t test_rng_state = 0x853c49e6748fea9bull;
static uint64_t test_rnd(void) {
    test_rng_state ^= test_rng_state << 13;
    test_rng_state ^= test_rng_state >> 7;
    test_rng_state ^= test_rng_state << 17;
    return test_rng_state;
}

static int test_pair_rank_dense(void) {
    tsimd_vocab_t v;
    memset(&v, 0, sizeof(v));
    int rc = tsimd_vocab_load_file(
        TSIMD_SRC_DIR "/models/o200k_base.tiktoken", &v);
    TEST_ASSERT(rc == 0);
    TEST_ASSERT(v.mgr_dense != NULL);

    for (uint32_t a = 0; a < 256; a++)
        for (uint32_t b = 0; b < 256; b++) {
            uint32_t r1 = tsimd_vocab_merge_rank(&v, a, b);
            uint32_t r2 = tsimd_vocab_merge_rank2(&v, a, b);
            TEST_ASSERT_EQ(r1, r2);
        }

    for (uint32_t a = 0; a < 2048; a += 13)
        for (uint32_t b = 0; b < 2048; b += 17) {
            TEST_ASSERT_EQ(tsimd_vocab_merge_rank(&v, a, b),
                           tsimd_vocab_merge_rank2(&v, a, b));
        }

    for (int i = 0; i < 200000; i++) {
        uint32_t a = (uint32_t)(test_rnd() % v.vocab_size);
        uint32_t b = (uint32_t)(test_rnd() % v.vocab_size);
        TEST_ASSERT_EQ(tsimd_vocab_merge_rank(&v, a, b),
                       tsimd_vocab_merge_rank2(&v, a, b));
    }
    tsimd_vocab_free(&v);
    printf("  pair-rank-dense: OK\n");
    return 0;
}

static int test_pair_priority(void) {

    tsimd_vocab_t v;
    memset(&v, 0, sizeof(v));
    int rc = tsimd_vocab_load_file(
        TSIMD_SRC_DIR "/models/o200k_base.tiktoken", &v);
    TEST_ASSERT(rc == 0);
    TEST_ASSERT(v.rank_to_id == NULL);

    for (uint32_t a = 0; a < 256; a++)
        for (uint32_t b = 0; b < 256; b++) {
            uint32_t m = tsimd_vocab_merge_rank(&v, a, b);
            TEST_ASSERT_EQ(tsimd_vocab_merge_priority(&v, a, b), m);

            TEST_ASSERT_EQ(tsimd_vocab_merge_priority_fast(&v, a, b), m);
            TEST_ASSERT_EQ(tsimd_vocab_id_of_priority(&v, m), m);
        }

    for (uint32_t a = 0; a < 2048; a += 13)
        for (uint32_t b = 0; b < 2048; b += 17) {
            TEST_ASSERT_EQ(tsimd_vocab_merge_priority(&v, a, b),
                           tsimd_vocab_merge_rank(&v, a, b));
            TEST_ASSERT_EQ(tsimd_vocab_merge_priority_fast(&v, a, b),
                           tsimd_vocab_merge_priority(&v, a, b));
        }

    for (int i = 0; i < 200000; i++) {
        uint32_t a = (uint32_t)(test_rnd() % v.vocab_size);
        uint32_t b = (uint32_t)(test_rnd() % v.vocab_size);
        TEST_ASSERT_EQ(tsimd_vocab_merge_priority(&v, a, b),
                       tsimd_vocab_merge_rank(&v, a, b));
        TEST_ASSERT_EQ(tsimd_vocab_merge_priority_fast(&v, a, b),
                       tsimd_vocab_merge_priority(&v, a, b));
    }
    tsimd_vocab_free(&v);

    {
        tsimd_tokenizer_t t;
        tsimd_added_token_t eot = {"<|eot|>", 0, 7, 1};
        tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
        TEST_ASSERT(tsimd_tokenizer_load_model(
                        &t, TSIMD_MODEL_GPT2,
                        TSIMD_SRC_DIR "/tests/fixtures/"
                        "mini_encoder.json",
                        TSIMD_SRC_DIR "/tests/fixtures/"
                        "mini_vocab.bpe",
                        &scheme, &eot, 1) == 0);
        const tsimd_vocab_t *gv = &t.vocab;
        TEST_ASSERT(gv->rank_to_id != NULL);
        TEST_ASSERT(gv->ranks != NULL);
        for (uint32_t a = 0; a < gv->vocab_size; a++)
            for (uint32_t b = 0; b < gv->vocab_size; b++) {
                uint32_t m = tsimd_vocab_merge_rank(gv, a, b);
                uint32_t p = tsimd_vocab_merge_priority(gv, a, b);
                if (m == UINT32_MAX) TEST_ASSERT_EQ(p, UINT32_MAX);
                else {
                    TEST_ASSERT_EQ(p, tsimd_vocab_priority(gv, m));
                    TEST_ASSERT_EQ(tsimd_vocab_id_of_priority(gv, p), m);
                }
            }
        tsimd_tokenizer_free(&t);
    }

    TEST_ASSERT(load_test_vocab(&v) == 0);
    TEST_ASSERT_EQ(tsimd_vocab_merge_priority(&v, 'a', 'b'), 256);
    TEST_ASSERT_EQ(tsimd_vocab_id_of_priority(&v, 256), 256);
    TEST_ASSERT_EQ(tsimd_vocab_merge_priority(&v, 256, 'c'), 257);
    TEST_ASSERT_EQ(tsimd_vocab_merge_priority(&v, 'b', 'c'), UINT32_MAX);
    TEST_ASSERT_EQ(tsimd_vocab_id_of_priority(&v, UINT32_MAX), UINT32_MAX);
    tsimd_vocab_free(&v);
    printf("  pair-priority: OK\n");
    return 0;
}

static size_t fp_run_pass(tsimd_vocab_t *v, uint64_t seed, uint32_t *res,
                          size_t cap) {
    uint64_t rng = seed;
    uint8_t probe[16];
    size_t n = 0;
    for (uint32_t id = 0; id < v->vocab_size && n < cap; id++) {
        uint32_t len;
        const uint8_t *b = tsimd_vocab_bytes(v, id, &len);
        if (len == 0) continue;
        res[n++] = tsimd_vocab_find(v, b, len);
    }
    for (size_t i = 0; i < 4096 && n < cap; i++) {
        uint32_t len = 1 + tsimd_test_rand(&rng) % 8;
        for (uint32_t k = 0; k < len; k++)
            probe[k] = (uint8_t)tsimd_test_rand(&rng);
        res[n++] = tsimd_vocab_find(v, probe, len);
    }
    for (uint32_t id = 0; id < v->vocab_size && n < cap; id++) {
        uint32_t len;
        const uint8_t *b = tsimd_vocab_bytes(v, id, &len);
        if (len == 0) continue;
        memcpy(probe, b, len);
        probe[tsimd_test_rand(&rng) % len] ^= 0x5A;
        res[n++] = tsimd_vocab_find(v, probe, len);
    }
    return n;
}

static int test_vocab_fp_ab(void) {
    size_t cap = 8192;
    uint32_t *res_on = malloc(cap * sizeof(uint32_t));
    uint32_t *res_off = malloc(cap * sizeof(uint32_t));
    TEST_ASSERT(res_on && res_off);

    tsimd_vocab_t v;
    TEST_ASSERT(load_test_vocab(&v) == 0);

    for (uint32_t id = 0; id < v.vocab_size; id++) {
        uint32_t len;
        const uint8_t *b = tsimd_vocab_bytes(&v, id, &len);
        if (len == 0) continue;
        TEST_ASSERT_EQ(tsimd_vocab_find(&v, b, len), id);
    }
    size_t n = fp_run_pass(&v, 0x9E3779B97F4A7C15ull, res_on, cap);
    TEST_ASSERT(n > 4096);

    tsimd_vocab_free(&v);
    setenv("TSIMD_DISABLE_VOCAB_FP", "1", 1);
    TEST_ASSERT(load_test_vocab(&v) == 0);
    size_t n2 = fp_run_pass(&v, 0x9E3779B97F4A7C15ull, res_off, cap);
    unsetenv("TSIMD_DISABLE_VOCAB_FP");
    TEST_ASSERT_EQ(n2, n);
    for (size_t i = 0; i < n; i++)
        TEST_ASSERT_EQ(res_off[i], res_on[i]);

    free(res_on);
    free(res_off);
    tsimd_vocab_free(&v);
    printf("  vocab fingerprint A/B: OK (%zu probes)\n", n);
    return 0;
}

int main(void) {
    tsimd_vocab_t v;

    TEST_ASSERT(tsimd_vocab_load_buffer((const uint8_t *)"YQ==", 4, &v) == -1);
    TEST_ASSERT(test_pair_rank_dense() == 0);
    TEST_ASSERT(test_pair_priority() == 0);
    TEST_ASSERT(test_vocab_fp_ab() == 0);
    TEST_ASSERT(tsimd_vocab_load_buffer((const uint8_t *)"!!!\n", 4, &v) == -1);
    TEST_ASSERT(tsimd_vocab_load_buffer((const uint8_t *)"YQ==\n", 5, &v) == -1);
    {

        static const char *bad[] = {"abc"};
        size_t len;
        char *buf = tsimd_test_make_vocab(bad, 1, &len);
        TEST_ASSERT(buf != NULL);
        TEST_ASSERT(tsimd_vocab_load_buffer((const uint8_t *)buf, len, &v) == -1);
        free(buf);
    }

    TEST_ASSERT(load_test_vocab(&v) == 0);
    TEST_ASSERT_EQ(v.vocab_size, 256 + 4);
    TEST_ASSERT_EQ(v.max_token_len, 4);

    TEST_ASSERT_EQ(v.byte_ids['a'], 'a');
    TEST_ASSERT_EQ(v.byte_ids[' '], ' ');
    TEST_ASSERT_EQ(v.byte_ids[0xFF], 0xFF);

    TEST_ASSERT_EQ(tsimd_vocab_find(&v, (const uint8_t *)"abcd", 4), 258);
    TEST_ASSERT_EQ(tsimd_vocab_find(&v, (const uint8_t *)"ab", 2), 256);
    TEST_ASSERT_EQ(tsimd_vocab_find(&v, (const uint8_t *)" x", 2), 259);
    TEST_ASSERT_EQ(tsimd_vocab_find(&v, (const uint8_t *)"a", 1), 'a');
    TEST_ASSERT_EQ(tsimd_vocab_find(&v, (const uint8_t *)"zz", 2), UINT32_MAX);
    TEST_ASSERT_EQ(tsimd_vocab_find(&v, (const uint8_t *)"abcde", 5), UINT32_MAX);

    TEST_ASSERT_EQ(tsimd_vocab_merge_rank(&v, 'a', 'b'), 256);
    TEST_ASSERT_EQ(tsimd_vocab_merge_rank(&v, 256, 'c'), 257);
    TEST_ASSERT_EQ(tsimd_vocab_merge_rank(&v, 257, 'd'), 258);
    TEST_ASSERT_EQ(tsimd_vocab_merge_rank(&v, ' ', 'x'), 259);
    TEST_ASSERT_EQ(tsimd_vocab_merge_rank(&v, 'b', 'c'), UINT32_MAX);
    TEST_ASSERT_EQ(tsimd_vocab_merge_rank(&v, 0, 0), UINT32_MAX);

    uint32_t blen;
    const uint8_t *b = tsimd_vocab_bytes(&v, 258, &blen);
    TEST_ASSERT_EQ(blen, 4);
    TEST_ASSERT(memcmp(b, "abcd", 4) == 0);
    tsimd_vocab_bytes(&v, 97, &blen);
    TEST_ASSERT_EQ(blen, 1);

    {
        size_t len;
        char *buf = tsimd_test_make_vocab(EXTRAS, 1, &len);
        TEST_ASSERT(buf != NULL);
        char *crlf = malloc(len * 2);
        size_t o = 0;
        for (size_t i = 0; i < len; i++) {
            if (buf[i] == '\n') crlf[o++] = '\r';
            crlf[o++] = buf[i];
        }
        tsimd_vocab_t v2;
        TEST_ASSERT(tsimd_vocab_load_buffer((const uint8_t *)crlf, o, &v2) == 0);
        TEST_ASSERT_EQ(v2.vocab_size, 257);
        tsimd_vocab_free(&v2);
        free(crlf);
        free(buf);
    }

    tsimd_vocab_free(&v);
    printf("test_vocabulary: OK\n");
    return 0;
}
