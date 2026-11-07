#include "bpe.h"
#include "pretokenizer.h"
#include "util.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

static void gen_span(uint32_t s, uint8_t *out, uint32_t len) {
    for (uint32_t i = 0; i < len; i++)
        out[i] = (uint8_t)((s * 31u + i * 7u) % 250u) + 1u;
}

static void gen_ids(const uint8_t *span, uint32_t len, uint32_t *out,
                    uint32_t n) {
    for (uint32_t i = 0; i < n; i++)
        out[i] = 256u + ((span[0] * 7u + (len > 1 ? span[len - 1] : 0) * 13u +
                          i * 29u) % 100000u);
}

static uint32_t gen_n(const uint8_t *span, uint32_t len) {
    return 1u + ((span[0] + (len > 1 ? span[len - 1] : 0)) %
                 TSIMD_PCACHE_MAX_IDS);
}

static int test_fidelity(void) {
    tsimd_pretok_cache_t c;
    TEST_ASSERT(tsimd_pretok_cache_init(&c, 1 << 12) == 0);

    for (uint32_t len = 1; len <= TSIMD_PCACHE_MAX_SPAN; len++) {
        for (uint32_t n = 1; n <= TSIMD_PCACHE_MAX_IDS; n++) {
            uint8_t span[TSIMD_PCACHE_MAX_SPAN];
            uint32_t ids[TSIMD_PCACHE_MAX_IDS];
            gen_span(len * 100u + n, span, len);
            gen_ids(span, len, ids, n);
            tsimd_pcache_insert(&c, span, len, ids, n);
        }
    }

    for (uint32_t len = 1; len <= TSIMD_PCACHE_MAX_SPAN; len++) {
        for (uint32_t n = 1; n <= TSIMD_PCACHE_MAX_IDS; n++) {
            uint8_t span[TSIMD_PCACHE_MAX_SPAN];
            uint32_t want[TSIMD_PCACHE_MAX_IDS];
            gen_span(len * 100u + n, span, len);
            gen_ids(span, len, want, n);
            uint32_t got_n = 0;
            uint32_t got[TSIMD_PCACHE_MAX_IDS];
            uint32_t *r = tsimd_pcache_lookup_into(&c, span, len, got,
                                                   &got_n);
            TEST_ASSERT(r != NULL);
            TEST_ASSERT_EQ(got_n, n);
            TEST_ASSERT(memcmp(got, want, n * sizeof(uint32_t)) == 0);
        }
    }

    {
        uint8_t span[8];
        gen_span(999999u, span, 8);
        uint32_t got_n = 0, got[TSIMD_PCACHE_MAX_IDS];
        TEST_ASSERT(tsimd_pcache_lookup_into(&c, span, 8, got, &got_n) ==
                    NULL);
    }
    tsimd_pretok_cache_free(&c);
    printf("  fidelity: OK\n");
    return 0;
}

static int test_edges(void) {
    tsimd_pretok_cache_t c;
    TEST_ASSERT(tsimd_pretok_cache_init(&c, 1 << 12) == 0);

    for (uint32_t len = 6; len <= 10; len++) {
        uint8_t span[TSIMD_PCACHE_MAX_SPAN];
        uint32_t ids[4] = {256u + len, 300u, 400u, 500u};
        gen_span(7000u + len, span, len);
        tsimd_pcache_insert(&c, span, len, ids, 4);
        uint32_t got_n = 0, got[TSIMD_PCACHE_MAX_IDS];
        uint32_t *r = tsimd_pcache_lookup_into(&c, span, len, got, &got_n);
        TEST_ASSERT(r != NULL && got_n == 4);
        TEST_ASSERT(memcmp(got, ids, 4 * sizeof(uint32_t)) == 0);
    }

    {
        uint8_t a[12] = "abcdefghijk";
        uint8_t b[12] = "abcdefghxyz";
        uint32_t ia[2] = {1001u, 1002u};
        uint32_t ib[2] = {2001u, 2002u};
        tsimd_pcache_insert(&c, a, 11, ia, 2);
        tsimd_pcache_insert(&c, b, 11, ib, 2);
        uint32_t got_n = 0, got[TSIMD_PCACHE_MAX_IDS];
        TEST_ASSERT(tsimd_pcache_lookup_into(&c, a, 11, got, &got_n) != NULL);
        TEST_ASSERT(memcmp(got, ia, 2 * sizeof(uint32_t)) == 0);
        TEST_ASSERT(tsimd_pcache_lookup_into(&c, b, 11, got, &got_n) != NULL);
        TEST_ASSERT(memcmp(got, ib, 2 * sizeof(uint32_t)) == 0);
    }

    {
        uint8_t a[15] = "abcdefghijklmno";
        uint8_t b[15] = "abcdefghijklmn";
        b[14] = 'x';
        uint32_t ia[1] = {31u};
        uint32_t ib[1] = {32u};
        tsimd_pcache_insert(&c, a, 15, ia, 1);
        tsimd_pcache_insert(&c, b, 14, ib, 1);
        uint32_t got_n = 0, got[TSIMD_PCACHE_MAX_IDS];
        TEST_ASSERT(tsimd_pcache_lookup_into(&c, a, 15, got, &got_n) != NULL);
        TEST_ASSERT_EQ(got[0], 31u);
        TEST_ASSERT(tsimd_pcache_lookup_into(&c, b, 14, got, &got_n) != NULL);
        TEST_ASSERT_EQ(got[0], 32u);
    }

    {
        uint8_t span[5] = {'a', 0x00, 'b', 0x00, 'c'};
        uint32_t ids[2] = {77u, 78u};
        tsimd_pcache_insert(&c, span, 5, ids, 2);
        uint32_t got_n = 0, got[TSIMD_PCACHE_MAX_IDS];
        TEST_ASSERT(tsimd_pcache_lookup_into(&c, span, 5, got, &got_n) !=
                    NULL);
        TEST_ASSERT(memcmp(got, ids, 2 * sizeof(uint32_t)) == 0);
    }

    {
        uint8_t span[9] = {0, 0, 0, 0, 0, 0, 0, 0, 'z'};
        uint32_t ids[1] = {99u};
        tsimd_pcache_insert(&c, span, 9, ids, 1);
        uint32_t got_n = 0, got[TSIMD_PCACHE_MAX_IDS];
        TEST_ASSERT(tsimd_pcache_lookup_into(&c, span, 9, got, &got_n) ==
                    NULL);

        uint8_t probe[15] = "abcdefghijklmno";
        TEST_ASSERT(tsimd_pcache_lookup_into(&c, probe, 15, got, &got_n) !=
                    NULL);
        TEST_ASSERT_EQ(got[0], 31u);
    }
    tsimd_pretok_cache_free(&c);
    printf("  edges: OK\n");
    return 0;
}

#define MERGE_N 4000

static int test_merge(void) {
    tsimd_pretok_cache_t src, dst;
    TEST_ASSERT(tsimd_pretok_cache_init(&src, 1 << 13) == 0);
    TEST_ASSERT(tsimd_pretok_cache_init(&dst, 1 << 13) == 0);

    for (uint32_t s = 0; s < MERGE_N; s++) {
        uint32_t len = 1 + (s % TSIMD_PCACHE_MAX_SPAN);
        uint8_t span[TSIMD_PCACHE_MAX_SPAN];
        gen_span(s, span, len);
        uint32_t n = gen_n(span, len);
        uint32_t ids[TSIMD_PCACHE_MAX_IDS];
        gen_ids(span, len, ids, n);
        tsimd_pcache_insert(&src, span, len, ids, n);
    }
    tsimd_pcache_merge(&dst, &src);

    for (uint32_t s = 0; s < MERGE_N; s++) {
        uint32_t len = 1 + (s % TSIMD_PCACHE_MAX_SPAN);
        uint8_t span[TSIMD_PCACHE_MAX_SPAN];
        gen_span(s, span, len);
        uint32_t n = gen_n(span, len);
        uint32_t want[TSIMD_PCACHE_MAX_IDS];
        gen_ids(span, len, want, n);
        uint32_t got_n = 0, got[TSIMD_PCACHE_MAX_IDS];
        uint32_t *r = tsimd_pcache_lookup_into(&dst, span, len, got,
                                               &got_n);
        if (r == NULL) {
            fprintf(stderr, "merge lost span %u (len %u)\n", s, len);
            return 1;
        }
        if (got_n != n || memcmp(got, want, n * sizeof(uint32_t)) != 0) {
            fprintf(stderr, "merge corrupted span %u (len %u): got_n=%u\n",
                    s, len, got_n);
            return 1;
        }
    }

    long hits = 0;
    for (uint32_t s = MERGE_N; s < MERGE_N + 500; s++) {
        uint32_t len = 1 + (s % TSIMD_PCACHE_MAX_SPAN);
        uint8_t span[TSIMD_PCACHE_MAX_SPAN];
        gen_span(s + 500000u, span, len);
        uint32_t got_n = 0, got[TSIMD_PCACHE_MAX_IDS];
        if (tsimd_pcache_lookup_into(&dst, span, len, got, &got_n) != NULL) {
            hits++;
            uint32_t n = gen_n(span, len);
            uint32_t want[TSIMD_PCACHE_MAX_IDS];
            gen_ids(span, len, want, n);
            if (got_n != n || memcmp(got, want, n * sizeof(uint32_t)) != 0) {
                fprintf(stderr,
                        "merge served WRONG ids for a probe span (s=%u)\n",
                        s);
                return 1;
            }
        }
    }
    printf("  (probe spans: %ld legitimate content-hits)\n", hits);
    tsimd_pretok_cache_free(&src);
    tsimd_pretok_cache_free(&dst);
    printf("  merge (%d spans): OK\n", MERGE_N);
    return 0;
}

#define CT 8
#define SPANS_PER_THREAD 4000

typedef struct {
    tsimd_pretok_cache_t *c;
    uint32_t base;
} ins_ctx_t;

static void *inserter(void *arg) {
    ins_ctx_t *ctx = arg;
    for (uint32_t k = 0; k < SPANS_PER_THREAD; k++) {
        uint32_t s = ctx->base + k;
        uint32_t len = 1 + (s % TSIMD_PCACHE_MAX_SPAN);
        uint8_t span[TSIMD_PCACHE_MAX_SPAN];
        gen_span(s, span, len);
        uint32_t n = gen_n(span, len);
        uint32_t ids[TSIMD_PCACHE_MAX_IDS];
        gen_ids(span, len, ids, n);
        tsimd_pcache_insert(ctx->c, span, len, ids, n);
    }
    return NULL;
}

static int test_concurrent(void) {
    tsimd_pretok_cache_t c;
    TEST_ASSERT(tsimd_pretok_cache_init(&c, 1 << 16) == 0);
    c.shared = 1;

    pthread_t th[CT];
    ins_ctx_t ctx[CT];
    for (int t = 0; t < CT; t++) {
        ctx[t].c = &c;
        ctx[t].base = (uint32_t)t * SPANS_PER_THREAD;
        TEST_ASSERT(pthread_create(&th[t], NULL, inserter, &ctx[t]) == 0);
    }
    for (int t = 0; t < CT; t++)
        pthread_join(th[t], NULL);

    long missing = 0, wrong = 0;
    for (int t = 0; t < CT; t++) {
        for (uint32_t k = 0; k < SPANS_PER_THREAD; k++) {
            uint32_t s = ctx[t].base + k;
            uint32_t len = 1 + (s % TSIMD_PCACHE_MAX_SPAN);
            uint8_t span[TSIMD_PCACHE_MAX_SPAN];
            gen_span(s, span, len);
            uint32_t n = gen_n(span, len);
            uint32_t want[TSIMD_PCACHE_MAX_IDS];
            gen_ids(span, len, want, n);
            uint32_t got_n = 0, got[TSIMD_PCACHE_MAX_IDS];
            uint32_t *r = tsimd_pcache_lookup_into(&c, span, len, got,
                                                   &got_n);
            if (r == NULL) {

                missing++;
                continue;
            }
            if (got_n != n || memcmp(got, want, n * sizeof(uint32_t)) != 0) {
                wrong++;
                if (wrong <= 3)
                    fprintf(stderr,
                            "CORRUPT span %u: got_n=%u want_n=%u "
                            "got0=%u want0=%u\n",
                            s, got_n, n, got[0], want[0]);
            }
        }
    }
    printf("  concurrent (%d threads x %d spans): %ld missing, "
           "%ld WRONG\n", CT, SPANS_PER_THREAD, missing, wrong);
    TEST_ASSERT(wrong == 0);
    tsimd_pretok_cache_free(&c);
    return 0;
}

static int test_two_stage(void) {
    tsimd_pretok_cache_t frozen, priv;
    TEST_ASSERT(tsimd_pretok_cache_init(&frozen, 1 << 12) == 0);
    TEST_ASSERT(tsimd_pretok_cache_init(&priv, 1 << 12) == 0);

    uint8_t s100[8], s200[8], s300[8];
    gen_span(100u, s100, 8);
    gen_span(200u, s200, 8);
    gen_span(300u, s300, 8);
    uint32_t f100[2] = {11u, 12u};
    uint32_t p200[2] = {21u, 22u};
    uint32_t f300[2] = {31u, 32u};
    uint32_t p300[2] = {99u, 98u};
    tsimd_pcache_insert(&frozen, s100, 8, f100, 2);
    tsimd_pcache_insert(&frozen, s300, 8, f300, 2);
    tsimd_pcache_insert(&priv, s200, 8, p200, 2);
    tsimd_pcache_insert(&priv, s300, 8, p300, 2);

    for (int rep = 0; rep < 2; rep++) {
        uint32_t got_n = 0, got[TSIMD_PCACHE_MAX_IDS];
        uint32_t *r = tsimd_pcache_lookup_into(&frozen, s100, 8, got,
                                               &got_n);
        if (!r)
            r = tsimd_pcache_lookup_into(&priv, s100, 8, got, &got_n);
        TEST_ASSERT(r != NULL && got_n == 2);
        TEST_ASSERT(memcmp(got, f100, 2 * sizeof(uint32_t)) == 0);

        got_n = 0;
        r = tsimd_pcache_lookup_into(&frozen, s200, 8, got, &got_n);
        if (!r)
            r = tsimd_pcache_lookup_into(&priv, s200, 8, got, &got_n);
        TEST_ASSERT(r != NULL && got_n == 2);
        TEST_ASSERT(memcmp(got, p200, 2 * sizeof(uint32_t)) == 0);

        got_n = 0;
        r = tsimd_pcache_lookup_into(&frozen, s300, 8, got, &got_n);
        if (!r)
            r = tsimd_pcache_lookup_into(&priv, s300, 8, got, &got_n);
        TEST_ASSERT(r != NULL && got_n == 2);
        if (memcmp(got, f300, 2 * sizeof(uint32_t)) != 0) {
            fprintf(stderr, "two-stage served the PRIVATE ids for a "
                    "frozen span (got %u,%u want 31,32)\n",
                    got[0], got[1]);
            return 1;
        }
    }
    tsimd_pretok_cache_free(&frozen);
    tsimd_pretok_cache_free(&priv);
    printf("  two-stage order: OK\n");
    return 0;
}

static void bounded_key_expected(const uint8_t *bytes, uint32_t len,
                                 uint64_t *k0, uint64_t *k1) {
    uint8_t k[16];
    memset(k, 0, 16);
    memcpy(k, bytes, len);
    k[15] = (uint8_t)len;
    memcpy(k0, k, 8);
    memcpy(k1, k + 8, 8);
}

static int test_bounded_key(void) {
    uint8_t buf[96];
    for (uint32_t len = 1; len <= TSIMD_PCACHE_MAX_SPAN; len++) {
        for (int t = 0; t < 64; t++) {

            uint8_t span[TSIMD_PCACHE_MAX_SPAN];
            gen_span(1300u + len * 511u + (uint32_t)t * 31u, span, len);
            for (uint32_t i = 0; i < len && i < 8; i++)
                if (t % 8 == 0) span[i] = 0;
            for (int off = 0; off <= 20; off++) {
                memset(buf, 0xA5, sizeof(buf));
                memcpy(buf + off, span, len);
                uint64_t e0, e1, eh, ek0, ek1;

                tsimd_pcache_key_hash(buf + off, len, &e0, &e1, &eh);
                bounded_key_expected(buf + off, len, &ek0, &ek1);
                TEST_ASSERT_EQ(e0, ek0);
                TEST_ASSERT_EQ(e1, ek1);

                const uint8_t *end = buf + 64;
                tsimd_pcache_key_hash_bounded(buf + off, len, end,
                                              &e0, &e1, &eh);
                TEST_ASSERT_EQ(e0, ek0);
                TEST_ASSERT_EQ(e1, ek1);
                TEST_ASSERT(eh != 0);

                const uint8_t *end2 = buf + 96;
                uint64_t f0, f1, fh;
                tsimd_pcache_key_hash_bounded(buf + off, len, end2,
                                              &f0, &f1, &fh);
                TEST_ASSERT_EQ(f0, ek0);
                TEST_ASSERT_EQ(f1, ek1);
                TEST_ASSERT(fh != 0);
            }
        }
    }

    {
        uint8_t z[16] = {0, 0, 0, 0, 0, 0, 0, 0, 'x'};
        uint64_t a0, a1, ah, b0, b1, bh;
        tsimd_pcache_key_hash(z, 9, &a0, &a1, &ah);
        tsimd_pcache_key_hash_bounded(z, 9, z + 16, &b0, &b1, &bh);
        TEST_ASSERT_EQ(a0, 0ull);
        TEST_ASSERT_EQ(b0, 0ull);
    }

    {
        uint8_t g[32];
        memset(g, 0x5A, sizeof(g));
        memcpy(g, "abc", 3);
        uint64_t r0, r1, rh;
        uint64_t e0, e1, eh;
        tsimd_pcache_key_hash(g, 3, &r0, &r1, &rh);
        tsimd_pcache_key_hash_bounded(g, 3, g + 32, &e0, &e1, &eh);
        TEST_ASSERT_EQ(r0, e0);
        TEST_ASSERT_EQ(r1, e1);
        bounded_key_expected(g, 3, &e0, &e1);
        TEST_ASSERT_EQ(r0, e0);
        TEST_ASSERT_EQ(r1, e1);
    }
    printf("  bounded-key: OK\n");
    return 0;
}

static uint64_t long_hash(const uint8_t *b, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static int test_long_cache(void) {
    tsimd_pretok_cache_t c;
    TEST_ASSERT(tsimd_pretok_cache_init(&c, 1 << 12) == 0);

    for (uint32_t len = 16; len <= 40; len++) {
        uint8_t span[64];
        uint32_t ids[4];
        gen_span(33000u + len, span, len);
        gen_ids(span, len, ids, 4);
        tsimd_plong_insert(&c, span, len, ids, 4);

        uint32_t got[4], got_n = 0;
        TEST_ASSERT(tsimd_plong_probe(&c, span, len, got, &got_n) == 1);
        TEST_ASSERT_EQ(got_n, 4u);
        TEST_ASSERT(memcmp(got, ids, 4 * sizeof(uint32_t)) == 0);
    }

    {
        uint8_t a[32] = "aaaaaa...aaaX", b[32] = "aaaaaa...aaaY";
        memset(a, 'a', 20); memset(b, 'a', 20);
        a[19] = 'X'; b[19] = 'Y';
        uint32_t ia[2] = {111u, 222u};
        tsimd_plong_insert(&c, a, 20, ia, 2);
        uint32_t got[4], got_n = 0;
        TEST_ASSERT(tsimd_plong_probe(&c, a, 20, got, &got_n) == 1);
        TEST_ASSERT(memcmp(got, ia, 2 * sizeof(uint32_t)) == 0);

        TEST_ASSERT(tsimd_plong_probe(&c, b, 20, got, &got_n) == 0);
    }

    {
        uint8_t big[256];
        memset(big, 'z', sizeof(big));
        for (uint32_t i = 0; i < 20000; i++) {
            big[0] = (uint8_t)(i & 0xFF);
            big[1] = (uint8_t)(i >> 8);
            uint32_t ids[2] = {1000u + i, 2000u + i};
            tsimd_plong_insert(&c, big, sizeof(big), ids, 2);
        }
        uint32_t got[4], got_n = 0;

        uint32_t n_hit = 0;
        for (uint32_t i = 19950; i < 20000; i++) {
            big[0] = (uint8_t)(i & 0xFF);
            big[1] = (uint8_t)(i >> 8);
            if (tsimd_plong_probe(&c, big, sizeof(big), got, &got_n) == 1)
                n_hit++;
        }
        TEST_ASSERT(n_hit > 0);
    }

    {
        uint8_t span[24];
        gen_span(4711u, span, sizeof(span));
        enum { N8 = 8, NBIG = 1223 };
        uint32_t ids8[N8], idsbig[NBIG];
        for (uint32_t i = 0; i < N8; i++) ids8[i] = 0x100u + i;
        for (uint32_t i = 0; i < NBIG; i++) idsbig[i] = 0x20000u + i;
        tsimd_plong_insert(&c, span, sizeof(span), ids8, N8);
        uint32_t n = 0;
        const uint32_t *r = tsimd_plong_probe_ex(&c, span, sizeof(span), &n);
        TEST_ASSERT(r != NULL);
        TEST_ASSERT_EQ(n, (uint32_t)N8);
        TEST_ASSERT(memcmp(r, ids8, N8 * sizeof(uint32_t)) == 0);

        uint32_t got[4], got_n = 0;
        TEST_ASSERT(tsimd_plong_probe(&c, span, sizeof(span), got,
                                      &got_n) == 0);

        uint8_t big[64];
        gen_span(5150u, big, sizeof(big));
        tsimd_plong_insert(&c, big, sizeof(big), idsbig, NBIG);
        n = 0;
        r = tsimd_plong_probe_ex(&c, big, sizeof(big), &n);
        TEST_ASSERT(r != NULL);
        TEST_ASSERT_EQ(n, (uint32_t)NBIG);
        TEST_ASSERT(memcmp(r, idsbig, NBIG * sizeof(uint32_t)) == 0);

        for (uint32_t i = 0; i < 4000; i++) {
            big[0] = (uint8_t)(i & 0xFF);
            big[1] = (uint8_t)(i >> 8);
            uint32_t many[6];
            for (uint32_t k = 0; k < 6; k++) many[k] = 0x30000u + i * 8 + k;
            tsimd_plong_insert(&c, big, sizeof(big), many, 6);
        }
        big[0] = 0xFF; big[1] = 0x0F;
        uint32_t many[6];
        for (uint32_t k = 0; k < 6; k++) many[k] = 0x30000u + 4095u * 8 + k;
        n = 0;
        r = tsimd_plong_probe_ex(&c, big, sizeof(big), &n);
        TEST_ASSERT(r != NULL);
        TEST_ASSERT_EQ(n, 6u);
        TEST_ASSERT(memcmp(r, many, 6 * sizeof(uint32_t)) == 0);
    }
    tsimd_pretok_cache_free(&c);
    printf("  long: OK\n");
    return 0;
}

static int test_vocab_seed(void) {
    tsimd_vocab_t v;
    memset(&v, 0, sizeof(v));
    TEST_ASSERT(tsimd_vocab_load_file(
        TSIMD_SRC_DIR "/models/o200k_base.tiktoken", &v) == 0);
    tsimd_pcache_frozen_t f;
    TEST_ASSERT(tsimd_pcache_frozen_build(&f, &v) == 0);

    size_t checked = 0;
    for (uint32_t id = 0; id < v.vocab_size; id++) {
        uint32_t len = v.offsets[id + 1] - v.offsets[id];
        if (len < 1 || len > TSIMD_PCACHE_MAX_SPAN) continue;
        const uint8_t *b = v.bytes + v.offsets[id];

        uint32_t z = 0;
        while (z < len && z < 8 && b[z] == 0) z++;
        if (z == (len < 8 ? len : 8)) continue;
        uint8_t k[16];
        memset(k, 0, 16);
        memcpy(k, b, len);
        k[15] = (uint8_t)len;
        uint64_t k0, k1, hash;
        memcpy(&k0, k, 8);
        memcpy(&k1, k + 8, 8);
        {
            uint64_t x0, x1;
            tsimd_pcache_key_hash_bounded(k, len, k + 16, &x0, &x1, &hash);
        }
        uint64_t val_out = 0;
        if (tsimd_pcache_frozen_probe(&f, k0, k1, hash, &val_out) != 1 ||
            (uint32_t)((val_out >> 3) & 0x1FFFFFFF) != id) {
            fprintf(stderr, "frozen mismatch id=%u len=%u k0=%llx k1=%llx "
                            "hash=%llx idx=%llx\n", id, len,
                    (unsigned long long)k0, (unsigned long long)k1,
                    (unsigned long long)hash,
                    (unsigned long long)((size_t)hash & f.mask));

            for (size_t s = 0; s <= f.mask; s++)
                if (f.entries[s].k0 == k0 && f.entries[s].k1 == k1) {
                    fprintf(stderr, "  entry found at %zu (val=%llx)\n", s,
                            (unsigned long long)f.entries[s].val);
                    break;
                }
            return 1;
        }
        checked++;
    }
    printf("  frozen: %zu tokens served [id]\n", checked);
    TEST_ASSERT(checked > 100000);

    {
        uint8_t nz[9];
        memset(nz, 0, 8);
        nz[8] = 1;
        uint64_t k0, k1, hash;
        tsimd_pcache_key_hash_bounded(nz, 9, nz + 9, &k0, &k1, &hash);
        uint64_t val_out = 0;
        TEST_ASSERT(tsimd_pcache_frozen_probe(&f, k0, k1, hash, &val_out) == 0);
    }
    tsimd_pcache_frozen_free(&f);
    tsimd_vocab_free(&v);
    printf("  frozen: OK\n");
    return 0;
}

static int test_insert_at(void) {
    tsimd_pretok_cache_t c;
    TEST_ASSERT(tsimd_pretok_cache_init(&c, 1 << 12) == 0);

    uint8_t span[12] = "at slot abc";
    uint32_t ids[2] = {9100u, 9101u};
    uint64_t k0, k1, h;
    tsimd_pcache_key_hash_bounded(span, 11, span + 12, &k0, &k1, &h);
    uint64_t pval = 0, pext = 0;
    size_t slot = 0;

    TEST_ASSERT(tsimd_pcache_probe_walk(&c, k0, k1, h, &pval, &pext,
                                        &slot) == 0);
    tsimd_pcache_insert_at_keyed(&c, slot, k0, k1, h, ids, 2);

    uint32_t got[4], gn = 0;
    TEST_ASSERT(tsimd_pcache_probe_home(&c, k0, k1, h, &pval, &pext) == 1);
    (void)got;
    (void)gn;
    TEST_ASSERT_EQ((uint32_t)(pval & 7), 2u);
    TEST_ASSERT_EQ((uint32_t)((pval >> 3) & 0x1FFFFFFF), ids[0]);
    TEST_ASSERT_EQ((uint32_t)(pval >> 32), ids[1]);
    tsimd_pretok_cache_free(&c);
    printf("  insert-at: OK\n");
    return 0;
}

int main(void) {
    printf("cache fidelity:\n");
    if (getenv("TSIMD_ALL")) { TEST_ASSERT(test_fidelity() == 0); }
    printf("cache edges:\n");
    if (getenv("TSIMD_ALL")) { TEST_ASSERT(test_edges() == 0); }
    printf("cache merge:\n");
    if (getenv("TSIMD_ALL")) { TEST_ASSERT(test_merge() == 0); }
    printf("cache concurrency:\n");
    if (getenv("TSIMD_ALL")) { TEST_ASSERT(test_concurrent() == 0); }
    printf("cache two-stage:\n");
    if (getenv("TSIMD_ALL")) { TEST_ASSERT(test_two_stage() == 0); }
    printf("cache bounded-key:\n");
    if (getenv("TSIMD_ALL")) { TEST_ASSERT(test_bounded_key() == 0); }
    printf("cache long:\n");
    if (getenv("TSIMD_ALL")) { TEST_ASSERT(test_long_cache() == 0); }
    printf("cache frozen:\n");
    TEST_ASSERT(test_vocab_seed() == 0);
    printf("cache insert-at:\n");
    TEST_ASSERT(test_insert_at() == 0);
    printf("test_cache_stress: OK\n");
    return 0;
}
