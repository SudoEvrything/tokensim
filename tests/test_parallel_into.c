#include "tokenizer.h"
#include "util.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *EXTRAS[] = {
    "ab", "abc", "abcd", " x", "th", "the", " t", " the",
    "er", "her", "in", "ing",
};
#define VOCAB_SIZE (256 + 12)
#define SPECIAL_A VOCAB_SIZE

static int write_vocab_file(const char *path) {
    size_t len;
    char *buf = tsimd_test_make_vocab(EXTRAS, 12, &len);
    if (!buf) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) { free(buf); return -1; }
    size_t wrote = fwrite(buf, 1, len, f);
    fclose(f);
    free(buf);
    return wrote == len ? 0 : -1;
}

static uint8_t *gen_text(uint64_t *rng, size_t len) {
    uint8_t *buf = malloc(len);
    if (!buf) return NULL;
    static const char *WORDS[] = {
        "the", "quick", "Brown", "fox", "jumps", "don't", "I'm", "we'll",
        "x=1;", "42", "3.14", "na\u00efve", "\u4e2d\u6587", "foo_bar",
        "(test)", "hello,", "world!", "OK", "\u00f3w \u0142\u00f3d\u017a",
    };
    static const char *TAILS[] = {" ", " ", " ", "\n", "\r\n", "\t", "  ",
                                  " \n ", ",", "; ", " 123 ", "'s "};
    size_t i = 0;
    while (i < len) {
        if ((tsimd_test_rand(rng) & 7) == 0) {
            size_t run = 1 + tsimd_test_rand(rng) % 40;
            uint8_t c = (uint8_t)"0123abcdefgh   "[tsimd_test_rand(rng) % 15];
            while (run-- && i < len) buf[i++] = c;
            continue;
        }
        const char *w = WORDS[tsimd_test_rand(rng) % 19];
        size_t wl = strlen(w);
        const char *t = TAILS[tsimd_test_rand(rng) % 12];
        size_t tl = strlen(t);
        if (i + wl + tl > len) break;
        memcpy(buf + i, w, wl);
        i += wl;
        memcpy(buf + i, t, tl);
        i += tl;
    }
    while (i < len) buf[i++] = ' ';
    return buf;
}

static int load_tok(tsimd_tokenizer_t *t) {
    char vocab_path[] = "/tmp/tsimd_pinto_vocab_XXXXXX";
    int fd = mkstemp(vocab_path);
    if (fd < 0 || write_vocab_file(vocab_path) != 0) {
        if (fd >= 0) close(fd);
        return -1;
    }
    close(fd);
    tsimd_added_token_t specials[2] = {
        {"<|eot|>", SPECIAL_A, 7, 1},
        {"hello", 0, 5, 0},
    };
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    int rc = tsimd_tokenizer_load(t, vocab_path, scheme, specials, 2);
    unlink(vocab_path);
    return rc;
}

static int cmp_ids(const uint32_t *a, size_t na, const uint32_t *b, size_t nb,
                   const char *what, int threads) {
    if (na != nb) {
        fprintf(stderr, "FAIL %s th=%d: count %zu vs %zu\n", what, threads, na, nb);
        return 1;
    }
    for (size_t i = 0; i < na; i++) {
        if (a[i] != b[i]) {
            fprintf(stderr, "FAIL %s th=%d: id %zu %u vs %u\n", what, threads, i,
                    (unsigned)a[i], (unsigned)b[i]);
            return 1;
        }
    }
    return 0;
}

static int test_all_threads(void) {
    tsimd_tokenizer_t t;
    TEST_ASSERT_EQ(load_tok(&t), 0);
    uint64_t rng = 0x51AB1E17ULL;
    size_t len = 900000;
    uint8_t *data = gen_text(&rng, len);
    TEST_ASSERT(data != NULL);
    for (size_t k = 1000; k + 7 < len; k += 51113)
        memcpy(data + k, "<|eot|>", 7);
    for (size_t k = 700; k + 5 < len; k += 30011)
        memcpy(data + k, "hello", 5);

    static const int THREADS[] = {1, 2, 3, 4, 7, 8};
    for (size_t s = 0; s < sizeof(THREADS) / sizeof(THREADS[0]); s++) {
        int th = THREADS[s];
        tsimd_token_sequence_t seq;
        memset(&seq, 0, sizeof(seq));
        TEST_ASSERT_EQ(tsimd_tokenizer_encode_parallel(&t, data, len, th, &seq), 0);
        size_t bound = tsimd_tokenizer_encode_bound(&t, len);
        TEST_ASSERT(bound >= seq.count);
        uint32_t *buf = malloc(bound * sizeof(uint32_t));
        TEST_ASSERT(buf != NULL);
        memset(buf, 0xAB, bound * sizeof(uint32_t));
        size_t count = 12345;
        TEST_ASSERT_EQ(tsimd_tokenizer_encode_parallel_into(&t, data, len, th,
                                                            buf, bound, &count),
                       0);
        TEST_ASSERT_EQ(count, seq.count);
        if (cmp_ids(buf, count, seq.ids, seq.count, "parallel_into", th)) {
            free(buf);
            tsimd_token_sequence_free(&seq);
            free(data);
            tsimd_tokenizer_free(&t);
            return 1;
        }
        size_t count2 = 0;
        TEST_ASSERT_EQ(tsimd_tokenizer_encode_into_raw(&t, data, len, buf, bound,
                                                       &count2),
                       0);
        if (cmp_ids(buf, count2, seq.ids, seq.count, "encode_into_raw", th)) {
            free(buf);
            tsimd_token_sequence_free(&seq);
            free(data);
            tsimd_tokenizer_free(&t);
            return 1;
        }
        free(buf);
        tsimd_token_sequence_free(&seq);
    }
    free(data);
    tsimd_tokenizer_free(&t);
    printf("  threads 1..8 agree: OK\n");
    return 0;
}

static int test_capacity_and_null(void) {
    tsimd_tokenizer_t t;
    TEST_ASSERT_EQ(load_tok(&t), 0);
    uint64_t rng = 0xC0FFEE11ULL;
    size_t len = 600000;
    uint8_t *data = gen_text(&rng, len);
    TEST_ASSERT(data != NULL);
    size_t bound = tsimd_tokenizer_encode_bound(&t, len);
    uint32_t *buf = malloc(bound * sizeof(uint32_t));
    TEST_ASSERT(buf != NULL);
    size_t count = 0;
    TEST_ASSERT(tsimd_tokenizer_encode_parallel_into(&t, data, len, 4, buf,
                                                     bound - 1, &count) != 0);
    TEST_ASSERT_EQ(count, 0);
    TEST_ASSERT(tsimd_tokenizer_encode_parallel_into(&t, data, len, 4, NULL,
                                                     bound, &count) != 0);
    TEST_ASSERT(tsimd_tokenizer_encode_parallel_into(&t, data, len, 4, buf, 0,
                                                     &count) != 0);
    TEST_ASSERT_EQ(tsimd_tokenizer_encode_parallel_into(&t, data, len, 4, buf,
                                                        bound, &count),
                   0);
    TEST_ASSERT(count > 0 && count <= bound);
    free(buf);
    free(data);
    tsimd_tokenizer_free(&t);
    printf("  capacity + null guards: OK\n");
    return 0;
}

static int test_empty_and_tiny(void) {
    tsimd_tokenizer_t t;
    TEST_ASSERT_EQ(load_tok(&t), 0);
    static const char *tiny[] = {"", "a", "the quick", "<|eot|>", "hello"};
    for (size_t i = 0; i < sizeof(tiny) / sizeof(tiny[0]); i++) {
        size_t len = strlen(tiny[i]);
        tsimd_token_sequence_t seq;
        memset(&seq, 0, sizeof(seq));
        TEST_ASSERT_EQ(tsimd_tokenizer_encode_parallel(
                           &t, (const uint8_t *)tiny[i], len, 4, &seq),
                       0);
        size_t bound = tsimd_tokenizer_encode_bound(&t, len);
        uint32_t *buf = malloc(bound * sizeof(uint32_t));
        TEST_ASSERT(buf != NULL);
        size_t count = 999;
        int irc = tsimd_tokenizer_encode_parallel_into(
            &t, (const uint8_t *)tiny[i], len, 4, buf, bound, &count);
        if (irc != 0) {
            fprintf(stderr, "FAIL tiny[%zu]=\"%s\" len=%zu bound=%zu rc=%d\n", i,
                    tiny[i], len, bound, irc);
            free(buf);
            tsimd_token_sequence_free(&seq);
            tsimd_tokenizer_free(&t);
            return 1;
        }
        if (cmp_ids(buf, count, seq.ids, seq.count, "tiny", 4)) {
            free(buf);
            tsimd_token_sequence_free(&seq);
            tsimd_tokenizer_free(&t);
            return 1;
        }
        free(buf);
        tsimd_token_sequence_free(&seq);
    }
    tsimd_tokenizer_free(&t);
    printf("  empty and tiny inputs: OK\n");
    return 0;
}

int main(void) {
    if (test_all_threads()) return 1;
    if (test_capacity_and_null()) return 1;
    if (test_empty_and_tiny()) return 1;
    printf("parallel into: OK\n");
    return 0;
}
