#ifndef TSIMD_TEST_UTIL_H
#define TSIMD_TEST_UTIL_H

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>

#define TEST_ASSERT(cond)                                                    \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
            return 1;                                                        \
        }                                                                    \
    } while (0)

#define TEST_ASSERT_EQ(a, b)                                                 \
    do {                                                                     \
        long long _a = (long long)(a), _b = (long long)(b);                  \
        if (_a != _b) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s == %s (%lld != %lld)\n",         \
                    __FILE__, __LINE__, #a, #b, _a, _b);                     \
            return 1;                                                        \
        }                                                                    \
    } while (0)

char *tsimd_test_make_vocab(const char **extras, size_t n_extras,
                           size_t *out_len);

long tsimd_test_unescape(const char *s, size_t n, uint8_t *out, size_t cap,
                        size_t *out_len);

char *tsimd_test_read_file(const char *path, size_t *out_len);

uint64_t tsimd_test_rand(uint64_t *state);

#endif
