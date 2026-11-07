#include "bpe.h"
#include "tokenizer.h"

#include <stdio.h>
#include <string.h>

#define KIMI TSIMD_SRC_DIR "/models/reference/kimi_k2_tiktoken.model"

static int expect_ids(tsimd_tokenizer_t *t, const char *label,
                      const uint8_t *text, size_t len, const uint32_t *want,
                      size_t n) {
    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    if (tsimd_tokenizer_encode(t, text, len, &seq) != 0) {
        fprintf(stderr, "FAIL: %s: encode error\n", label);
        return 1;
    }
    int rc = 0;
    if (seq.count != n) {
        fprintf(stderr, "FAIL: %s: %zu ids want %zu\n", label,
                (size_t)seq.count, n);
        rc = 1;
    }
    for (size_t i = 0; i < n && i < seq.count; i++) {
        if (seq.ids[i] != want[i]) {
            fprintf(stderr, "FAIL: %s: id %zu = %u want %u\n", label, i,
                    seq.ids[i], want[i]);
            rc = 1;
            break;
        }
    }
    tsimd_token_sequence_free(&seq);
    return rc;
}

int main(void) {
    tsimd_tokenizer_t t;
    memset(&t, 0, sizeof t);
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_KIMI;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TIKTOKEN, KIMI, NULL,
                                   &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed\n");
        return 2;
    }
    int rc = 0;
    {
        uint32_t want[] = {19180, 11, 2695, 0, 646, 5072, 16331, 69275, 220,
                           6694, 13};
        rc |= expect_ids(&t, "ascii", (const uint8_t *)"Hello, world! The "
                                                    "quick brown fox 123.",
                         38, want, 11);
    }
    {
        uint32_t want[] = {1435, 95, 1855, 25875, 33360};
        rc |= expect_ids(&t, "cjk", (const uint8_t *)"漢字テスト", 15, want,
                         5);
    }
    {
        uint32_t want[] = {64, 1435, 95, 65};
        rc |= expect_ids(&t, "a-han-b", (const uint8_t *)"a漢b", 5, want, 4);
    }
    {
        uint32_t want[] = {64072};
        rc |= expect_ids(&t, "contraction", (const uint8_t *)"it's", 4, want,
                         1);
    }
    {
        uint32_t want[] = {220, 8134, 14803, 198, 3514, 123516};
        rc |= expect_ids(&t, "ws", (const uint8_t *)"  leading "
                                                    "spaces\nnew\ttab",
                         24, want, 6);
    }
    {
        uint32_t want[] = {4492, 29530, 11379, 38089, 156631, 35, 220, 5512,
                           220, 6694, 12972, 22, 1288, 117657, 6, 89};
        rc |= expect_ids(&t, "mixed", (const uint8_t *)"UPPERlower MiXeD 42 "
                                                       "1234567 x'y'z",
                         33, want, 16);
    }
    {
        uint32_t want[] = {88709, 70591, 7915, 44187, 367};
        rc |= expect_ids(&t, "accents", (const uint8_t *)"don't café naïve",
                         18, want, 5);
    }
    {
        uint32_t want[] = {197, 197, 41414, 1288};
        rc |= expect_ids(&t, "tabs", (const uint8_t *)"\t\tindent x", 10,
                         want, 4);
    }
    {
        uint32_t want[] = {4546, 63011, 25875, 32045, 33360, 343, 4546,
                           63011, 4663, 25875, 32045, 33360, 292};
        rc |= expect_ids(&t, "jp", (const uint8_t *)"日本語テキスト、日本語"
                                                    "のテキスト。",
                         51, want, 13);
    }
    {
        uint32_t want[] = {198, 220, 1288};
        rc |= expect_ids(&t, "ws-nl-2sp", (const uint8_t *)"\n  x", 4, want,
                         3);
    }
    {
        uint32_t want[] = {23118, 220, 1288};
        rc |= expect_ids(&t, "ws-2sp-2nl", (const uint8_t *)"  \n\n  x", 7,
                         want, 3);
    }
    {
        uint32_t want[] = {64, 1420, 220, 291};
        rc |= expect_ids(&t, "ws-a-sp-nl", (const uint8_t *)"a \n  b", 6, want,
                         4);
    }
    {
        uint32_t want[] = {3898, 256};
        rc |= expect_ids(&t, "ws-3nl-tail", (const uint8_t *)"\n\n\n  ", 5,
                         want, 2);
    }
    {
        uint32_t want[] = {87, 198, 256, 364};
        rc |= expect_ids(&t, "ws-x-nl-3sp", (const uint8_t *)"x\n   y", 6,
                         want, 4);
    }
    {
        uint32_t want[] = {64, 42877, 65};
        rc |= expect_ids(&t, "ws-crlf-mid", (const uint8_t *)"a\n\t\nb", 5,
                         want, 3);
    }
    tsimd_tokenizer_free(&t);
    if (rc) return 1;
    printf("test_kimi: OK\n");
    return 0;
}
