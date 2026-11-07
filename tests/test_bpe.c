#include "bpe.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *EXTRAS[] = {"ab", "abc", "abcd", " x", "abab", "abababab"};

int main(void) {
    tsimd_vocab_t v;
    size_t vlen;
    char *vbuf = tsimd_test_make_vocab(EXTRAS, 6, &vlen);
    TEST_ASSERT(vbuf != NULL);
    TEST_ASSERT(tsimd_vocab_load_buffer((const uint8_t *)vbuf, vlen, &v) == 0);
    free(vbuf);

    tsimd_bpe_workspace_t ws;
    TEST_ASSERT(tsimd_bpe_workspace_init(&ws, 16) == 0);

    ws.out_count = 0;
    TEST_ASSERT(tsimd_bpe_encode(&v, &ws, (const uint8_t *)"abcd", 4) == 0);
    TEST_ASSERT_EQ(ws.out_count, 1);
    TEST_ASSERT_EQ(ws.out[0], 258);

    ws.out_count = 0;
    TEST_ASSERT(tsimd_bpe_encode(&v, &ws, (const uint8_t *)"abdc", 4) == 0);
    TEST_ASSERT_EQ(ws.out_count, 3);
    TEST_ASSERT_EQ(ws.out[0], 256);
    TEST_ASSERT_EQ(ws.out[1], 'd');
    TEST_ASSERT_EQ(ws.out[2], 'c');

    ws.out_count = 0;
    TEST_ASSERT(tsimd_bpe_encode(&v, &ws, (const uint8_t *)" xab", 4) == 0);
    TEST_ASSERT_EQ(ws.out_count, 2);
    TEST_ASSERT_EQ(ws.out[0], 259);
    TEST_ASSERT_EQ(ws.out[1], 256);

    ws.out_count = 0;
    TEST_ASSERT(tsimd_bpe_encode(&v, &ws, (const uint8_t *)"ab ab", 5) == 0);
    TEST_ASSERT_EQ(ws.out_count, 3);
    TEST_ASSERT_EQ(ws.out[0], 256);
    TEST_ASSERT_EQ(ws.out[1], ' ');
    TEST_ASSERT_EQ(ws.out[2], 256);

    ws.out_count = 0;
    TEST_ASSERT(tsimd_bpe_encode(&v, &ws, (const uint8_t *)"a", 1) == 0);
    TEST_ASSERT_EQ(ws.out_count, 1);
    TEST_ASSERT_EQ(ws.out[0], 'a');
    TEST_ASSERT(tsimd_bpe_encode(&v, &ws, (const uint8_t *)"", 0) == 0);

    const uint8_t junk[] = {0x00, 0xFF, 0xC3, 0x28, 0x7F};
    ws.out_count = 0;
    TEST_ASSERT(tsimd_bpe_encode(&v, &ws, junk, sizeof(junk)) == 0);
    TEST_ASSERT_EQ(ws.out_count, 5);
    for (int i = 0; i < 5; i++)
        TEST_ASSERT_EQ(ws.out[i], junk[i]);

    {
        uint8_t big[1000];
        for (size_t i = 0; i < sizeof(big); i++)
            big[i] = (i % 2) ? 'b' : 'a';
        ws.out_count = 0;
        TEST_ASSERT(tsimd_bpe_encode(&v, &ws, big, sizeof(big)) == 0);

        TEST_ASSERT_EQ(ws.out_count, 125);
        for (size_t i = 0; i < 125; i++)
            TEST_ASSERT_EQ(ws.out[i], 261);
    }

    {

        const size_t N = 2000;
        uint8_t *buf = malloc(N);
        TEST_ASSERT(buf != NULL);
        for (size_t i = 0; i < N; i++)
            buf[i] = (uint8_t)("ab"[(i * 7919) % 2]);
        for (size_t i = 0; i < N - 1; i++)
            if (buf[i] == 'a' && buf[i + 1] == 'b' && (i % 97) == 0)
                buf[i + 1] = 'c';

        {
            const size_t M = 600;
            uint8_t *rep = malloc(M);
            TEST_ASSERT(rep != NULL);
            for (size_t i = 0; i < M; i++) rep[i] = (i % 2) ? 'b' : 'a';

            uint32_t ids2[1024];
            TEST_ASSERT(M <= 1024);
            for (size_t i = 0; i < M; i++) ids2[i] = v.byte_ids[rep[i]];
            size_t n2 = M;
            for (;;) {
                uint32_t best_rank = UINT32_MAX, best_merged = UINT32_MAX;
                size_t best = (size_t)-1;
                for (size_t i = 0; i + 1 < n2; i++) {
                    uint32_t m = tsimd_vocab_merge_rank(&v, ids2[i], ids2[i + 1]);
                    if (m == UINT32_MAX) continue;
                    uint32_t r = tsimd_vocab_priority(&v, m);
                    if (r < best_rank) { best_rank = r; best_merged = m; best = i; }
                }
                if (best == (size_t)-1) break;
                ids2[best] = best_merged;
                for (size_t j = best + 1; j + 1 < n2; j++) ids2[j] = ids2[j + 1];
                n2--;
            }
            ws.out_count = 0;
            TEST_ASSERT(tsimd_bpe_encode(&v, &ws, rep, (uint32_t)M) == 0);
            TEST_ASSERT_EQ(ws.out_count, n2);
            for (size_t i = 0; i < n2; i++)
                if (ws.out[i] != ids2[i]) {
                    fprintf(stderr, "repetition mismatch at %zu\n", i);
                    return 1;
                }
            free(rep);
        }

        uint32_t ref[4096];
        size_t ref_n = 0;
        {
            uint32_t ids[4096];
            TEST_ASSERT(N <= 4096);
            for (size_t i = 0; i < N; i++) ids[i] = v.byte_ids[buf[i]];
            size_t n = N;
            for (;;) {
                uint32_t best_rank = UINT32_MAX, best_merged = UINT32_MAX;
                size_t best = (size_t)-1;
                for (size_t i = 0; i + 1 < n; i++) {
                    uint32_t m = tsimd_vocab_merge_rank(&v, ids[i], ids[i + 1]);
                    if (m == UINT32_MAX) continue;
                    uint32_t r = tsimd_vocab_priority(&v, m);
                    if (r < best_rank) {
                        best_rank = r;
                        best_merged = m;
                        best = i;
                    }
                }
                if (best == (size_t)-1) break;
                ids[best] = best_merged;
                for (size_t j = best + 1; j + 1 < n; j++) ids[j] = ids[j + 1];
                n--;
            }
            for (size_t i = 0; i < n; i++) ref[ref_n++] = ids[i];
        }

        ws.out_count = 0;
        TEST_ASSERT(tsimd_bpe_encode(&v, &ws, buf, (uint32_t)N) == 0);
        TEST_ASSERT_EQ(ws.out_count, ref_n);
        for (size_t i = 0; i < ref_n; i++)
            if (ws.out[i] != ref[i]) {
                fprintf(stderr, "heap path mismatch at %zu: %u != %u\n", i,
                        ws.out[i], ref[i]);
                return 1;
            }
        free(buf);
    }

    tsimd_bpe_workspace_free(&ws);
    tsimd_vocab_free(&v);

    printf("test_bpe: OK\n");
    return 0;
}
