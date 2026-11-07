#include "tokenizer.h"
#include "simd.h"
#include "input.h"
#include "util.h"

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
#define SPECIAL_B (VOCAB_SIZE + 1)

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
        "x=1;", "42", "3.14", "naïve", "中文", "🚀", "foo_bar", "(test)",
        "hello,", "world!", "OK", "ów łódź",
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
        const char *w = WORDS[tsimd_test_rand(rng) % 20];
        size_t wl = strlen(w);
        const char *t = TAILS[tsimd_test_rand(rng) % 12];
        size_t tl = strlen(t);
        if (i + wl + tl > len) break;
        memcpy(buf + i, w, wl);
        i += wl;
        memcpy(buf + i, t, tl);
        i += tl;
    }
    while (i < len) buf[i++] = 'x';
    return buf;
}

static int spans_match_at(tsimd_pretok_scheme_t scheme, const uint8_t *data,
                          size_t len, size_t p) {
    tsimd_span_array_t whole, left, right;
    tsimd_span_array_init(&whole);
    tsimd_span_array_init(&left);
    tsimd_span_array_init(&right);
    int fail = 0;
    if (tsimd_pretokenize_scalar_ref(scheme, data, len, &whole) != 0 ||
        tsimd_pretokenize_scalar_ref(scheme, data, p, &left) != 0 ||
        tsimd_pretokenize_scalar_ref(scheme, data + p, len - p, &right) != 0)
        fail = 1;
    if (!fail && left.count + right.count != whole.count) {
        fprintf(stderr, "seam %zu: %zu+%zu spans != %zu\n", p, left.count,
                right.count, whole.count);
        fail = 1;
    }
    for (size_t i = 0; !fail && i < left.count; i++)
        if (left.spans[i].start != whole.spans[i].start ||
            left.spans[i].end != whole.spans[i].end)
            fail = 1;
    for (size_t i = 0; !fail && i < right.count; i++) {
        const tsimd_span_t *w = &whole.spans[left.count + i];
        if (right.spans[i].start + p != w->start ||
            right.spans[i].end + p != w->end) {
            fprintf(stderr, "seam %zu: right span %zu [%u,%u) != [%u,%u)\n",
                    p, i, right.spans[i].start + (uint32_t)p,
                    right.spans[i].end + (uint32_t)p, w->start, w->end);
            fail = 1;
        }
    }
    tsimd_span_array_free(&whole);
    tsimd_span_array_free(&left);
    tsimd_span_array_free(&right);
    if (fail) fprintf(stderr, "seam mismatch at p=%zu, len=%zu\n", p, len);
    return fail;
}

static int test_safe_split(void) {
    static const tsimd_pretok_scheme_t SCHEMES[] = {
        TSIMD_PRETOK_GPT2, TSIMD_PRETOK_CL100K,
    };
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    for (size_t case_i = 0; case_i < 12; case_i++) {
        size_t len = 2 + (size_t)(tsimd_test_rand(&rng) % 5000);
        uint8_t *data = gen_text(&rng, len);
        if (!data) return 1;
        for (size_t s = 0; s < 2; s++) {
            for (size_t h = 0; h < 24; h++) {
                size_t hint = tsimd_test_rand(&rng) % len;
                long p = tsimd_pretok_safe_split(SCHEMES[s], data, len, hint, 0);
                if (p < 0) continue;
                if (p <= 0 || (size_t)p >= len) {
                    fprintf(stderr, "bad split %ld len %zu\n", p, len);
                    free(data);
                    return 1;
                }
                if (spans_match_at(SCHEMES[s], data, len, (size_t)p)) {
                    free(data);
                    return 1;
                }
            }
        }
        free(data);
    }

    uint8_t *digits = malloc(4096);
    if (!digits) return 1;
    memset(digits, '7', 4096);
    for (size_t s = 0; s < 2; s++)
        TEST_ASSERT_EQ(tsimd_pretok_safe_split(SCHEMES[s], digits, 4096, 2048, 0), -1);
    free(digits);

    rng = 0x9E3779B97F4A7C15ull;
    for (size_t case_i = 0; case_i < 12; case_i++) {
        size_t len = 2 + (size_t)(tsimd_test_rand(&rng) % 5000);
        uint8_t *data = gen_text(&rng, len);
        if (!data) return 1;
        for (size_t s = 0; s < 2; s++) {
            for (size_t h = 0; h < 24; h++) {
                size_t from = 1 + tsimd_test_rand(&rng) % (len - 1);
                long p = tsimd_pretok_safe_split(SCHEMES[s], data, len,
                                                 from, from);
                if (p < 0) continue;
                if ((size_t)p <= from || (size_t)p >= len) {
                    fprintf(stderr,
                            "lo-contract: p=%ld from=%zu len=%zu "
                            "(p must be strictly after from)\n",
                            p, from, len);
                    free(data);
                    return 1;
                }
                if ((size_t)p > from + 65538) {
                    fprintf(stderr, "lo-contract: p=%ld beyond the window "
                                    "from=%zu\n", p, from);
                    free(data);
                    return 1;
                }
                if (spans_match_at(SCHEMES[s], data, len, (size_t)p)) {
                    free(data);
                    return 1;
                }
            }
        }
        free(data);
    }
    printf("safe-split property: OK\n");
    return 0;
}

static int seq_eq(tsimd_tokenizer_t *t, const uint8_t *data, size_t len,
                  int threads, const char *what) {
    tsimd_token_sequence_t a, b;
    if (tsimd_tokenizer_encode(t, data, len, &a) != 0) return 1;
    if (tsimd_tokenizer_encode_parallel(t, data, len, threads, &b) != 0) {
        fprintf(stderr, "%s: parallel encode failed\n", what);
        tsimd_token_sequence_free(&a);
        return 1;
    }
    int fail = 0;
    if (a.count != b.count) {
        fprintf(stderr, "%s (threads=%d): %zu ids != %zu\n", what, threads,
                a.count, b.count);
        fail = 1;
    } else if (a.count > 0 && memcmp(a.ids, b.ids, a.count * 4) != 0) {
        for (size_t i = 0; i < a.count; i++)
            if (a.ids[i] != b.ids[i]) {
                fprintf(stderr,
                        "%s (threads=%d): first diff at %zu: %u != %u\n",
                        what, threads, i, b.ids[i], a.ids[i]);
                break;
            }
        fail = 1;
    }
    tsimd_token_sequence_free(&a);
    tsimd_token_sequence_free(&b);
    return fail;
}

static int test_parallel_matches(void) {
    char vocab_path[] = "/tmp/tsimd_par_vocab_XXXXXX";
    int fd = mkstemp(vocab_path);
    if (fd < 0 || write_vocab_file(vocab_path) != 0) {
        fprintf(stderr, "cannot write temp vocab\n");
        if (fd >= 0) { close(fd); unlink(vocab_path); }
        return 1;
    }
    close(fd);

    tsimd_added_token_t specials[2] = {
        {"<|eot|>", SPECIAL_A, 7, 1},
        {"hello", SPECIAL_B, 5, 0},
    };
    tsimd_tokenizer_t t;
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    int rc = tsimd_tokenizer_load(&t, vocab_path, scheme, specials, 2);
    unlink(vocab_path);
    TEST_ASSERT_EQ(rc, 0);

    static const int THREAD_SETS[] = {1, 2, 3, 7, 16};
    uint64_t rng = 0xDEADBEEFCAFEF00Dull;

    const char *tiny[] = {"", "a", "ab", " the the the", "<|eot|>", "x",
                          "hello world hello"};
    for (size_t i = 0; i < sizeof(tiny) / sizeof(tiny[0]); i++)
        for (size_t s = 0; s < sizeof(THREAD_SETS) / sizeof(int); s++)
            if (seq_eq(&t, (const uint8_t *)tiny[i], strlen(tiny[i]),
                       THREAD_SETS[s], tiny[i]))
                return 1;

    for (size_t case_i = 0; case_i < 4; case_i++) {
        size_t len = 300000 + case_i * 137000;
        uint8_t *data = gen_text(&rng, len);
        if (!data) return 1;
        for (size_t s = 0; s < sizeof(THREAD_SETS) / sizeof(int); s++)
            if (seq_eq(&t, data, len, THREAD_SETS[s], "generated"))
                return 1;
        free(data);
    }

    {
        size_t len = 400000;
        uint8_t *data = malloc(len);
        if (!data) return 1;
        memset(data, '5', len);
        for (size_t s = 0; s < sizeof(THREAD_SETS) / sizeof(int); s++)
            if (seq_eq(&t, data, len, THREAD_SETS[s], "digits"))
                return 1;
        free(data);
    }

    {
        size_t len = 350000;
        uint8_t *data = gen_text(&rng, len);
        if (!data) return 1;
        for (size_t k = 4096; k + 7 < len; k += 8191)
            memcpy(data + k, "<|eot|>", 7);
        for (size_t k = 100; k + 5 < len; k += 9999)
            memcpy(data + k, "hello", 5);
        for (size_t s = 0; s < sizeof(THREAD_SETS) / sizeof(int); s++)
            if (seq_eq(&t, data, len, THREAD_SETS[s], "specials"))
                return 1;
        free(data);
    }

    tsimd_tokenizer_free(&t);
    printf("parallel == sequential: OK\n");
    return 0;
}

static int test_batch(void) {
    char vocab_path[] = "/tmp/tsimd_par_vocab2_XXXXXX";
    int fd = mkstemp(vocab_path);
    if (fd < 0) return 1;
    if (write_vocab_file(vocab_path) != 0) { close(fd); unlink(vocab_path); return 1; }
    close(fd);

    tsimd_added_token_t batch_specials[1] = {
        {"<|eot|>", VOCAB_SIZE, 7, 1},
    };
    tsimd_tokenizer_t t;
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    TEST_ASSERT_EQ(tsimd_tokenizer_load(&t, vocab_path, scheme,
                                        batch_specials, 1), 0);
    unlink(vocab_path);

    uint64_t rng = 0x1234567890ABCDEFull;
    enum { N_DOCS = 9 };
    uint8_t *docs_data[N_DOCS];
    size_t lens[N_DOCS];
    size_t total = 0;
    for (size_t d = 0; d < N_DOCS; d++) {

        lens[d] = (d == 3) ? 0 : 30000 + d * 10000;
        docs_data[d] = gen_text(&rng, lens[d]);
        if (!docs_data[d]) return 1;
        total += lens[d];
    }
    uint8_t *all = malloc(total);
    if (!all) return 1;
    size_t off = 0;
    for (size_t d = 0; d < N_DOCS; d++) {
        memcpy(all + off, docs_data[d], lens[d]);
        off += lens[d];
    }

    tsimd_input_span_t spans[N_DOCS];
    for (size_t d = 0; d < N_DOCS; d++) {
        spans[d].data = docs_data[d];
        spans[d].length = lens[d];
    }
    for (size_t k = 0; k + 7 < lens[1]; k += 8191)
        memcpy(docs_data[1] + k, "<|eot|>", 7);

    for (int thr = 1; thr <= 8; thr += 3) {
        tsimd_token_batch_t batch;
        if (tsimd_tokenizer_encode_batch(&t, spans, N_DOCS, &batch, thr) != 0) {
            fprintf(stderr, "batch encode failed\n");
            return 1;
        }
        if (batch.n_docs != N_DOCS) return 1;
        size_t seq_off = 0;
        for (size_t d = 0; d < N_DOCS; d++) {
            tsimd_token_sequence_t one;
            if (tsimd_tokenizer_encode(&t, docs_data[d], lens[d], &one) != 0)
                return 1;
            size_t b0 = batch.offsets[d], b1 = batch.offsets[d + 1];
            if (b1 - b0 != one.count ||
                (one.count > 0 &&
                 memcmp(batch.ids + b0, one.ids, one.count * 4) != 0)) {
                fprintf(stderr, "doc %zu (threads=%d): batch != sequential\n",
                        d, thr);
                return 1;
            }
            if (b0 != seq_off) {
                fprintf(stderr, "doc %zu: offsets not dense\n", d);
                return 1;
            }
            seq_off = b1;
            tsimd_token_sequence_free(&one);
        }
        if (batch.offsets[N_DOCS] != batch.id_count) return 1;
        tsimd_token_batch_free(&batch);
    }

    tsimd_token_sequence_t whole;
    TEST_ASSERT_EQ(tsimd_tokenizer_encode(&t, all, total, &whole), 0);
    tsimd_token_sequence_free(&whole);

    for (size_t d = 0; d < N_DOCS; d++) free(docs_data[d]);
    free(all);
    tsimd_tokenizer_free(&t);
    printf("batch == sequential per doc: OK\n");
    return 0;
}

static int test_mmap(void) {
    char path[] = "/tmp/tsimd_mmap_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return 1;
    const char *text = "the quick brown fox  the  quick\t\tfox\n\nthe end";
    size_t len = strlen(text);
    if (write(fd, text, len) != (ssize_t)len) { close(fd); unlink(path); return 1; }
    close(fd);

    tsimd_input_t in;
    if (tsimd_input_mmap_file(path, &in) != 0) {
        printf("(mmap unavailable, skipping)\n");
        unlink(path);
        return 0;
    }
    int fail = (in.length != len || memcmp(in.data, text, len) != 0);
    tsimd_input_release(&in);
    unlink(path);
    TEST_ASSERT_EQ(fail, 0);
    printf("mmap input view: OK\n");
    return 0;
}

static int test_roundtrip(void) {
    char vocab_path[] = "/tmp/tsimd_rt_vocab_XXXXXX";
    int fd = mkstemp(vocab_path);
    if (fd < 0) return 1;
    if (write_vocab_file(vocab_path) != 0) { close(fd); unlink(vocab_path); return 1; }
    close(fd);

    tsimd_added_token_t specials[2] = {
        {"<|eot|>", VOCAB_SIZE, 7, 1},
        {"hello", SPECIAL_B, 5, 0},
    };
    tsimd_tokenizer_t t;
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    TEST_ASSERT_EQ(tsimd_tokenizer_load(&t, vocab_path, scheme, specials, 2), 0);
    unlink(vocab_path);

    uint64_t rng = 0xABCDEF0123456789ull;
    for (int case_i = 0; case_i < 3; case_i++) {
        size_t len = 100000u + (size_t)case_i * 50000u;
        uint8_t *data = gen_text(&rng, len);
        if (!data) return 1;
        for (size_t k = 2048; k + 7 < len; k += 6553)
            memcpy(data + k, "<|eot|>", 7);

        tsimd_token_sequence_t seq;
        TEST_ASSERT_EQ(tsimd_tokenizer_encode(&t, data, len, &seq), 0);

        tsimd_byte_sequence_t dec;
        TEST_ASSERT_EQ(tsimd_tokenizer_decode(&t, seq.ids, seq.count, 0, &dec), 0);
        TEST_ASSERT_EQ(dec.len, len);
        if (memcmp(dec.bytes, data, len) != 0) {
            fprintf(stderr, "round-trip mismatch (specials emitted)\n");
            return 1;
        }
        tsimd_byte_sequence_free(&dec);

        TEST_ASSERT_EQ(tsimd_tokenizer_decode(&t, seq.ids, seq.count, 1, &dec), 0);
        if (dec.len >= len) {
            fprintf(stderr, "skip_special produced too much\n");
            return 1;
        }
        tsimd_byte_sequence_free(&dec);

        tsimd_token_sequence_free(&seq);
        free(data);
    }

    {
        uint32_t bad[1] = {999999u};
        tsimd_byte_sequence_t dec;
        TEST_ASSERT_EQ(tsimd_tokenizer_decode(&t, bad, 1, 0, &dec), -1);
        TEST_ASSERT_EQ(tsimd_tokenizer_decode(&t, bad, 1, 1, &dec), -1);
    }

    tsimd_tokenizer_free(&t);
    printf("decode round-trip: OK\n");
    return 0;
}

static int test_static_arms(void) {
    char vocab_path[] = "/tmp/tsimd_arm_vocab_XXXXXX";
    int fd = mkstemp(vocab_path);
    if (fd < 0 || write_vocab_file(vocab_path) != 0) {
        if (fd >= 0) { close(fd); unlink(vocab_path); }
        return 1;
    }
    close(fd);

    tsimd_added_token_t specials[2] = {
        {"<|eot|>", SPECIAL_A, 7, 1},
        {"hello", SPECIAL_B, 5, 0},
    };
    tsimd_tokenizer_t t;
    int rc = tsimd_tokenizer_load(&t, vocab_path, TSIMD_PRETOK_GPT2,
                                  specials, 2);
    unlink(vocab_path);
    TEST_ASSERT_EQ(rc, 0);

    uint64_t rng = 0x0DDBA11CAFEF00Dull;
    enum { N_DOCS = 7 };
    uint8_t *docs_data[N_DOCS];
    size_t lens[N_DOCS];
    for (size_t d = 0; d < N_DOCS; d++) {
        lens[d] = 40000 + (size_t)(tsimd_test_rand(&rng) % 220000);
        docs_data[d] = gen_text(&rng, lens[d]);
        if (!docs_data[d]) return 1;

        for (size_t k = d * 977 + 13; k + 7 < lens[d]; k += 51113)
            memcpy(docs_data[d] + k, "<|eot|>", 7);
        for (size_t k = d * 313 + 7; k + 5 < lens[d]; k += 30011)
            memcpy(docs_data[d] + k, "hello", 5);
    }
    tsimd_input_span_t spans[N_DOCS];
    for (size_t d = 0; d < N_DOCS; d++) {
        spans[d].data = docs_data[d];
        spans[d].length = lens[d];
    }
    size_t total = 0;
    for (size_t d = 0; d < N_DOCS; d++) total += lens[d];
    uint8_t *all = malloc(total);
    if (!all) return 1;
    size_t off = 0;
    for (size_t d = 0; d < N_DOCS; d++) {
        memcpy(all + off, docs_data[d], lens[d]);
        off += lens[d];
    }

    for (int arm = 0; arm <= 2; arm++) {
        char armv[2] = {(char)('0' + arm), 0};
        setenv("TSIMD_PAR_STATIC", armv, 1);
        for (int thr = 2; thr <= 6; thr += 2) {
            if (seq_eq(&t, all, total, thr, "arms single-buffer"))
                return 1;
            tsimd_token_batch_t batch;
            if (tsimd_tokenizer_encode_batch(&t, spans, N_DOCS, &batch,
                                             thr) != 0) {
                fprintf(stderr, "arm %d threads %d: batch failed\n", arm,
                        thr);
                return 1;
            }
            for (size_t d = 0; d < N_DOCS; d++) {
                tsimd_token_sequence_t one;
                if (tsimd_tokenizer_encode(&t, docs_data[d], lens[d],
                                           &one) != 0)
                    return 1;
                size_t b0 = batch.offsets[d], b1 = batch.offsets[d + 1];
                if (b1 - b0 != one.count ||
                    (one.count > 0 &&
                     memcmp(batch.ids + b0, one.ids, one.count * 4) != 0)) {
                    fprintf(stderr,
                            "arm %d threads %d: doc %zu != sequential\n",
                            arm, thr, d);
                    return 1;
                }
                tsimd_token_sequence_free(&one);
            }
            tsimd_token_batch_free(&batch);
        }
    }
    unsetenv("TSIMD_PAR_STATIC");

    for (size_t d = 0; d < N_DOCS; d++) free(docs_data[d]);
    free(all);
    tsimd_tokenizer_free(&t);
    printf("static partition arms: OK\n");
    return 0;
}

static int test_pool_cache_init(void) {

    setenv("TSIMD_ENABLE_WSEED", "1", 1);
    char vocab_path[] = "/tmp/tsimd_pinit_vocab_XXXXXX";
    int fd = mkstemp(vocab_path);
    if (fd < 0 || write_vocab_file(vocab_path) != 0) {
        if (fd >= 0) { close(fd); unlink(vocab_path); }
        return 1;
    }
    close(fd);

    tsimd_tokenizer_t t;
    TEST_ASSERT_EQ(tsimd_tokenizer_load(&t, vocab_path, TSIMD_PRETOK_GPT2,
                                        NULL, 0), 0);
    unlink(vocab_path);

    size_t h0 = 0, m0 = 0, i0 = 0;
    tsimd_tokenizer_cache_stats(&t, &h0, &m0, &i0);
    if (i0 == 0) {
        fprintf(stderr, "pool-init: the sequential cache was not seeded\n");
        return 1;
    }

    enum { LEN = 300000 };
    uint8_t *data = malloc(LEN);
    if (!data) return 1;
    static const char rep[] = "the the the the the ";
    for (size_t i = 0; i < LEN; i++) data[i] = (uint8_t)rep[i % 20];

    tsimd_token_sequence_t s;
    if (tsimd_tokenizer_encode_parallel(&t, data, LEN, 2, &s) != 0) {
        fprintf(stderr, "pool-init: parallel encode failed\n");
        return 1;
    }
    tsimd_token_sequence_free(&s);
    size_t h1 = 0, m1 = 0, i1 = 0;
    tsimd_tokenizer_cache_stats(&t, &h1, &m1, &i1);

    if (tsimd_tokenizer_encode_parallel(&t, data, LEN, 4, &s) != 0) {
        fprintf(stderr, "pool-init: grow encode failed\n");
        return 1;
    }
    tsimd_token_sequence_free(&s);
    size_t h2 = 0, m2 = 0, i2 = 0;
    tsimd_tokenizer_cache_stats(&t, &h2, &m2, &i2);
    size_t grew = i2 - i1;
    if (grew < 2 * i0) {
        fprintf(stderr,
                "pool-init: growing 2 -> 4 workers inserted %zu entries — "
                "below the %zu of two seeded worker caches (a worker "
                "cache was not seeded)\n", grew, 2 * i0);
        return 1;
    }
    if (grew > 2 * i0 + 32) {
        fprintf(stderr,
                "pool-init: growing 2 -> 4 workers inserted %zu — above "
                "the %zu of two seeds (the input is not "
                "vocabulary-only?)\n", grew, 2 * i0);
        return 1;
    }

    if (tsimd_tokenizer_encode_parallel(&t, data, LEN, 4, &s) != 0)
        return 1;
    tsimd_token_sequence_free(&s);
    size_t h3 = 0, m3 = 0, i3 = 0;
    tsimd_tokenizer_cache_stats(&t, &h3, &m3, &i3);
    if (i3 != i2) {
        fprintf(stderr, "pool-init: a warm steady-state encode inserted "
                        "%zu more entries\n", i3 - i2);
        return 1;
    }
    free(data);
    tsimd_tokenizer_free(&t);

    char vocab2[] = "/tmp/tsimd_pinit2_vocab_XXXXXX";
    fd = mkstemp(vocab2);
    if (fd < 0 || write_vocab_file(vocab2) != 0) {
        if (fd >= 0) { close(fd); unlink(vocab2); }
        return 1;
    }
    close(fd);
    tsimd_tokenizer_t t2;
    TEST_ASSERT_EQ(tsimd_tokenizer_load(&t2, vocab2, TSIMD_PRETOK_GPT2,
                                        NULL, 0), 0);
    unlink(vocab2);
    uint8_t *data2 = malloc(LEN);
    if (!data2) return 1;
    for (size_t i = 0; i < LEN; i++) data2[i] = (uint8_t)rep[i % 20];
    setenv("TSIMD_POOL_INIT_FAIL", "1", 1);
    tsimd_token_sequence_t s2;
    int rc2 = tsimd_tokenizer_encode_parallel(&t2, data2, LEN, 4, &s2);
    if (rc2 == 0) {
        fprintf(stderr, "pool-init: a failed worker cache build did not "
                        "fail the encode\n");
        return 1;
    }
    unsetenv("TSIMD_POOL_INIT_FAIL");

    tsimd_token_sequence_t a, b;
    if (tsimd_tokenizer_encode(&t2, data2, LEN, &a) != 0 ||
        tsimd_tokenizer_encode(&t2, data2, LEN, &b) != 0) {
        fprintf(stderr, "pool-init: the tokenizer broke after the "
                        "failure\n");
        return 1;
    }
    if (a.count != b.count ||
        (a.count && memcmp(a.ids, b.ids, a.count * 4) != 0)) {
        fprintf(stderr, "pool-init: ids differ after the failure\n");
        return 1;
    }
    tsimd_token_sequence_free(&a);
    tsimd_token_sequence_free(&b);
    free(data2);
    tsimd_tokenizer_free(&t2);
    printf("pool cache init (parallel + seed + failure): OK\n");
    return 0;
}

extern size_t tsimd_serve_avx2_pieces(void);

static int test_serve_avx2_arms(void) {
    char vocab_path[] = "/tmp/tsimd_avx2_vocab_XXXXXX";
    int fd = mkstemp(vocab_path);
    if (fd < 0 || write_vocab_file(vocab_path) != 0) {
        if (fd >= 0) { close(fd); unlink(vocab_path); }
        return 1;
    }
    close(fd);
    tsimd_added_token_t specials[2] = {
        {"<|eot|>", SPECIAL_A, 7, 1},
        {"hello", SPECIAL_B, 5, 0},
    };
    tsimd_tokenizer_t t;
    TEST_ASSERT_EQ(tsimd_tokenizer_load(&t, vocab_path, TSIMD_PRETOK_GPT2,
                                        specials, 2), 0);
    unlink(vocab_path);

    enum { LEN = 300000 };
    uint64_t rng = 0x5EED107ull;
    uint8_t *data = gen_text(&rng, LEN);
    if (!data) return 1;
    for (size_t k = 4096; k + 7 < LEN; k += 8191)
        memcpy(data + k, "<|eot|>", 7);
    for (size_t k = 100; k + 5 < LEN; k += 9999)
        memcpy(data + k, "hello", 5);

    memset(data + 1000, 'A', 200);

    for (int thr = 2; thr <= 4; thr += 2) {
        setenv("TSIMD_SERVE_AVX2", "0", 1);
        size_t c0 = tsimd_serve_avx2_pieces();
        tsimd_token_batch_t off;
        if (tsimd_tokenizer_encode_batch(&t, &(tsimd_input_span_t){
                                                 (const uint8_t *)data,
                                                 LEN},
                                         1, &off, thr) != 0) {
            fprintf(stderr, "avx2: batch encode (off) failed\n");
            return 1;
        }
        uint32_t *ids_off = malloc(off.id_count * 4 + 4);
        if (!ids_off) return 1;
        memcpy(ids_off, off.ids, off.id_count * 4);
        size_t n_off = off.id_count;
        if (tsimd_serve_avx2_pieces() != c0) {
            fprintf(stderr, "avx2: the arm ran while disabled\n");
            return 1;
        }
        tsimd_token_batch_free(&off);

        setenv("TSIMD_SERVE_AVX2", "1", 1);
        tsimd_token_batch_t on;
        if (tsimd_tokenizer_encode_batch(&t, &(tsimd_input_span_t){
                                                (const uint8_t *)data, LEN},
                                        1, &on, thr) != 0) {
            fprintf(stderr, "avx2: batch encode (on) failed\n");
            return 1;
        }
        size_t c1 = tsimd_serve_avx2_pieces();
        if (c1 <= c0) {
            fprintf(stderr, "avx2: the arm did not run (counter %zu -> "
                            "%zu)\n", c0, c1);
            return 1;
        }
        if (on.id_count != n_off ||
            memcmp(on.ids, ids_off, n_off * 4) != 0) {
            fprintf(stderr, "avx2: ids differ at %d threads (%zu vs %zu)\n",
                    thr, on.id_count, n_off);
            return 1;
        }
        free(ids_off);
        tsimd_token_batch_free(&on);
    }

    for (int arm = 0; arm <= 1; arm++) {
        char av[2] = {(char)('0' + arm), 0};
        setenv("TSIMD_SERVE_AVX2", av, 1);
        for (int thr = 1; thr <= 4; thr += 3) {
            if (seq_eq(&t, data, LEN, thr, "avx2 single-buffer")) return 1;
        }
    }
    unsetenv("TSIMD_SERVE_AVX2");
    free(data);
    tsimd_tokenizer_free(&t);
    printf("serve avx2 arms: OK\n");
    return 0;
}

int main(void) {
    setenv("TSIMD_PCACHE_STATS", "1", 1);
    if (test_safe_split()) return 1;
    if (test_roundtrip()) return 1;
    if (test_parallel_matches()) return 1;
    if (test_batch()) return 1;
    if (test_static_arms()) return 1;
    if (test_pool_cache_init()) return 1;
    if (test_serve_avx2_arms()) return 1;
    if (test_mmap()) return 1;
    printf("test_parallel: OK\n");
    return 0;
}
