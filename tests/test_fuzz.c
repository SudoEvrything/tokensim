#include "tokenizer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *EXTRAS[] = {"ab", "abc", "abcd", " x"};

static int check_coverage(tsimd_pretok_scheme_t scheme, const uint8_t *d,
                          size_t n) {
    tsimd_span_array_t spans;
    tsimd_span_array_init(&spans);
    if (tsimd_pretokenize(scheme, d, n, &spans) != 0) return -1;
    size_t pos = 0;
    for (size_t i = 0; i < spans.count; i++) {
        if (spans.spans[i].start != pos ||
            spans.spans[i].end <= spans.spans[i].start) {
            tsimd_span_array_free(&spans);
            return -1;
        }
        pos = spans.spans[i].end;
    }
    int ok = (pos == n);
    tsimd_span_array_free(&spans);
    return ok ? 0 : -1;
}

static int check_roundtrip(tsimd_tokenizer_t *t, const uint8_t *d, size_t n) {
    tsimd_token_sequence_t seq;
    if (tsimd_tokenizer_encode(t, d, n, &seq) != 0) return -1;

    uint8_t *out = malloc(n ? n : 1);
    size_t o = 0;
    int fail = 0;
    for (size_t i = 0; i < seq.count && !fail; i++) {
        uint32_t id = seq.ids[i];
        if (id >= t->vocab.vocab_size) { fail = 1; break; }
        uint32_t len;
        const uint8_t *b = tsimd_vocab_bytes(&t->vocab, id, &len);
        if (o + len > n) { fail = 1; break; }
        memcpy(out + o, b, len);
        o += len;
    }
    if (!fail && (o != n || (n > 0 && memcmp(out, d, n) != 0))) fail = 1;
    free(out);
    tsimd_token_sequence_free(&seq);
    return fail ? -1 : 0;
}

static int check_split(tsimd_tokenizer_t *t, tsimd_pretok_scheme_t scheme,
                       const uint8_t *d, size_t n) {
    tsimd_span_array_t spans;
    tsimd_span_array_init(&spans);
    if (tsimd_pretokenize(scheme, d, n, &spans) != 0) return -1;
    tsimd_token_sequence_t whole, pre, suf;
    int fail = 0;
    if (tsimd_tokenizer_encode(t, d, n, &whole) != 0) fail = 1;
    if (!fail && spans.count >= 2) {

        size_t cut = spans.spans[spans.count / 2].start;
        if (tsimd_tokenizer_encode(t, d, cut, &pre) != 0 ||
            tsimd_tokenizer_encode(t, d + cut, n - cut, &suf) != 0) {
            fail = 1;
        } else if (pre.count + suf.count != whole.count ||
                   (whole.count > 0 &&
                    (memcmp(whole.ids, pre.ids, pre.count * sizeof(uint32_t)) != 0 ||
                     memcmp(whole.ids + pre.count, suf.ids,
                            suf.count * sizeof(uint32_t)) != 0))) {
            fprintf(stderr, "split mismatch at cut=%zu (len %zu)\n", cut, n);
            fail = 1;
        }
        tsimd_token_sequence_free(&pre);
        tsimd_token_sequence_free(&suf);
    }
    tsimd_token_sequence_free(&whole);
    tsimd_span_array_free(&spans);
    return fail ? -1 : 0;
}

static void gen(uint8_t *buf, size_t n, int mode, uint64_t *rng) {
    for (size_t i = 0; i < n; i++) {
        uint64_t r = tsimd_test_rand(rng);
        uint8_t b;
        switch (mode) {
            case 0:
                b = (uint8_t)(r >> 32);
                break;
            case 1:
                b = (uint8_t)" \n \t.,;:'\"abcdefgxyzABC0129?!-()"[(r >> 33) % 30];
                break;
            case 2:
                b = (uint8_t)" \t\n\r \n \t"[(r >> 33) % 9];
                break;
            case 3:
                b = (uint8_t)"a\xC3\xA9\xE4\xB8\xAD\xF0\x9F\xFF\x80\xC0\xED"[(r >> 33) % 13];
                break;
            default:
                b = (uint8_t)"'stlveS'TL dmRE"[(r >> 33) % 15];
                break;
        }
        buf[i] = b;
    }
}

int main(void) {
    char vocab_path_tmpl[] = "/tmp/tsimd_fuzz_vocab_XXXXXX";
    int vocab_fd = mkstemp(vocab_path_tmpl);
    if (vocab_fd < 0) return 1;
    close(vocab_fd);
    const char *vocab_path = vocab_path_tmpl;
    size_t vlen;
    char *vbuf = tsimd_test_make_vocab(EXTRAS, 4, &vlen);
    TEST_ASSERT(vbuf != NULL);
    FILE *f = fopen(vocab_path, "wb");
    TEST_ASSERT(f != NULL);
    TEST_ASSERT(fwrite(vbuf, 1, vlen, f) == vlen);
    fclose(f);
    free(vbuf);

    tsimd_tokenizer_t t;
    TEST_ASSERT(tsimd_tokenizer_load(&t, vocab_path, TSIMD_PRETOK_GPT2,
                                    NULL, 0) == 0);

    uint64_t rng = 0x123456789ABCDEF0ULL;
    long checks = 0;

    uint8_t buf[64];
    for (size_t n = 0; n <= 48; n++) {
        for (int mode = 0; mode < 5; mode++) {
            gen(buf, n, mode, &rng);
            TEST_ASSERT(check_coverage(TSIMD_PRETOK_GPT2, buf, n) == 0);
            TEST_ASSERT(check_coverage(TSIMD_PRETOK_CL100K, buf, n) == 0);
            TEST_ASSERT(check_roundtrip(&t, buf, n) == 0);
            TEST_ASSERT(check_split(&t, TSIMD_PRETOK_GPT2, buf, n) == 0);
            TEST_ASSERT(check_split(&t, TSIMD_PRETOK_CL100K, buf, n) == 0);
            checks += 5;
        }
    }

    for (int iter = 0; iter < 40; iter++) {
        size_t n = 1000 + (size_t)(tsimd_test_rand(&rng) % 60000);
        int mode = (int)(tsimd_test_rand(&rng) % 5);
        uint8_t *big = malloc(n);
        TEST_ASSERT(big != NULL);
        gen(big, n, mode, &rng);
        TEST_ASSERT(check_coverage(TSIMD_PRETOK_GPT2, big, n) == 0);
        TEST_ASSERT(check_coverage(TSIMD_PRETOK_CL100K, big, n) == 0);
        TEST_ASSERT(check_roundtrip(&t, big, n) == 0);
        TEST_ASSERT(check_split(&t, TSIMD_PRETOK_GPT2, big, n) == 0);
        TEST_ASSERT(check_split(&t, TSIMD_PRETOK_CL100K, big, n) == 0);
        free(big);
        checks += 5;
    }

    for (size_t n = 14; n <= 66; n++) {
        gen(buf, n > 64 ? 64 : n, 3, &rng);
        TEST_ASSERT(check_coverage(TSIMD_PRETOK_GPT2, buf, n > 64 ? 64 : n) == 0);
        TEST_ASSERT(check_coverage(TSIMD_PRETOK_CL100K, buf, n > 64 ? 64 : n) == 0);
        checks += 2;
    }

    tsimd_tokenizer_free(&t);
    remove(vocab_path);
    printf("test_fuzz: OK (%ld invariant checks)\n", checks);
    return 0;
}
