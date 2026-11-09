#include "tokensimd_sys.h"

#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CL100K TSIMD_SRC_DIR "/models/cl100k_base.tiktoken"
#define T5 TSIMD_SRC_DIR "/models/reference/t5_spiece.model"
#define QWEN TSIMD_SRC_DIR "/models/reference/qwen2.5-0.5b_tokenizer.json"

static int fails = 0;

static void check(int cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        fails++;
    }
}

int main(void) {
    char err[256];

    tsimd_sys *h = tsimd_sys_open(CL100K, "tiktoken", "cl100k", err, sizeof err);
    check(h != NULL, "open cl100k");
    if (!h) {
        fprintf(stderr, "  %s\n", err);
        return 1;
    }
    check(tsimd_sys_vocab_size(h) == 100256, "cl100k vocab size");
    check(strcmp(tsimd_sys_version(), "1.0.3") == 0,
          "version string");

    const char *text = "Hello world, this is a test.";
    int d_threads0 = tsimd_default_threads();
    uint32_t *ids = NULL;
    size_t n_ids = 0;
    check(tsimd_sys_encode(h, (const uint8_t *)text, strlen(text), 1, &ids,
                           &n_ids) == 0,
          "encode");
    check(n_ids == 8, "cl100k id count");
    uint8_t *bytes = NULL;
    size_t n_bytes = 0;
    check(tsimd_sys_decode(h, ids, n_ids, 0, &bytes, &n_bytes) == 0, "decode");
    check(n_bytes == strlen(text) &&
              memcmp(bytes, text, n_bytes) == 0,
          "decode round trip");
    tsimd_sys_free(bytes);

    size_t n_own = 0;
    const uint32_t *own = NULL;
    check(tsimd_sys_encode_own(h, (const uint8_t *)text, strlen(text), 1) == 0,
          "encode own");
    own = tsimd_sys_ids(h, &n_own);
    check(own != NULL && n_own == n_ids &&
              memcmp(own, ids, n_ids * sizeof(uint32_t)) == 0,
          "encode own ids equal the sequential ids");
    check(tsimd_sys_encode_own(h, (const uint8_t *)text, strlen(text), 4) == 0,
          "encode own parallel");
    own = tsimd_sys_ids(h, &n_own);
    check(own != NULL && n_own == n_ids &&
              memcmp(own, ids, n_ids * sizeof(uint32_t)) == 0,
          "encode own parallel ids equal the sequential ids");
    check(tsimd_sys_encode_own(h, (const uint8_t *)text, strlen(text), 0) == 0,
          "encode own automatic");
    own = tsimd_sys_ids(h, &n_own);
    check(own != NULL && n_own == n_ids &&
              memcmp(own, ids, n_ids * sizeof(uint32_t)) == 0,
          "encode own automatic ids equal the sequential ids");
    check(tsimd_sys_encode_own(h, (const uint8_t *)"", 0, 1) == 0,
          "encode own empty");
    own = tsimd_sys_ids(h, &n_own);
    check(n_own == 0, "encode own empty id count");

    uint32_t *ids4 = NULL;
    size_t n4 = 0;    check(tsimd_sys_encode(h, (const uint8_t *)text, strlen(text), 4, &ids4,
                           &n4) == 0,
          "encode parallel");
    check(n4 == n_ids && memcmp(ids4, ids, n_ids * sizeof(uint32_t)) == 0,
          "parallel ids equal the sequential ids");
    tsimd_sys_free(ids4);

    size_t bound = 0;
    check(tsimd_sys_encode_bound(h, (const uint8_t *)text, strlen(text), 1,
                                 &bound) == 0 && bound >= n_ids,
          "encode bound");
    uint32_t *into = malloc((bound ? bound : 1) * sizeof(uint32_t));
    size_t n_into = 0;
    memset(into, 0, (bound ? bound : 1) * sizeof(uint32_t));
    check(tsimd_sys_encode_into(h, (const uint8_t *)text, strlen(text), 1,
                                into, bound, &n_into) == 0 && n_into == n_ids,
          "encode into");
    check(memcmp(into, ids, n_ids * sizeof(uint32_t)) == 0,
          "encode into ids equal the sequential ids");
    memset(into, 0, (bound ? bound : 1) * sizeof(uint32_t));
    check(tsimd_sys_encode_into(h, (const uint8_t *)text, strlen(text), 4,
                                into, bound, &n_into) == 0 && n_into == n_ids,
          "encode into parallel");
    check(memcmp(into, ids, n_ids * sizeof(uint32_t)) == 0,
          "encode into parallel ids equal the sequential ids");
    uint32_t *tight = malloc(n_ids * sizeof(uint32_t));
    memset(tight, 0, n_ids * sizeof(uint32_t));
    check(tsimd_sys_encode_into(h, (const uint8_t *)text, strlen(text), 1,
                                tight, n_ids, &n_into) == 0 && n_into == n_ids,
          "encode into with a tight buffer");
    check(memcmp(tight, ids, n_ids * sizeof(uint32_t)) == 0,
          "tight-buffer ids equal the sequential ids");
    memset(tight, 0, n_ids * sizeof(uint32_t));
    check(tsimd_sys_encode_into(h, (const uint8_t *)text, strlen(text), 4,
                                tight, n_ids, &n_into) == 0 && n_into == n_ids,
          "encode into parallel with a tight buffer");
    check(memcmp(tight, ids, n_ids * sizeof(uint32_t)) == 0,
          "tight-buffer parallel ids equal the sequential ids");
    free(tight);
    check(tsimd_sys_encode_into(h, (const uint8_t *)text, strlen(text), 1,
                                into, 1, &n_into) != 0,
          "encode into must fail on a short buffer");
    check(tsimd_sys_encode_into(h, (const uint8_t *)text, strlen(text), 4,
                                into, 1, &n_into) != 0,
          "encode into parallel must fail on a short buffer");
    check(tsimd_default_threads() >= 1, "default thread count");
    memset(into, 0, (bound ? bound : 1) * sizeof(uint32_t));
    check(tsimd_sys_encode_into(h, (const uint8_t *)text, strlen(text), 0,
                                into, bound, &n_into) == 0 && n_into == n_ids,
          "encode into with automatic threads");
    check(memcmp(into, ids, n_ids * sizeof(uint32_t)) == 0,
          "automatic-thread ids equal the sequential ids");
    uint32_t *ids0 = NULL;
    size_t n0 = 0;
    check(tsimd_sys_encode(h, (const uint8_t *)text, strlen(text), 0, &ids0,
                           &n0) == 0 &&
              n0 == n_ids && memcmp(ids0, ids, n_ids * sizeof(uint32_t)) == 0,
          "automatic-thread encode ids equal the sequential ids");
    tsimd_sys_free(ids0);
    {
        const char *unit = "The quick brown fox jumps over the lazy dog. ";
        size_t ul = strlen(unit);
        size_t bl = (1u << 19) + 7;
        uint8_t *big = malloc(bl);
        for (size_t i = 0; i < bl; i++) big[i] = (uint8_t)unit[i % ul];
        uint32_t *b1 = NULL, *b0 = NULL, *b6 = NULL;
        size_t n1b = 0, n0b = 0, n6b = 0;
        check(tsimd_sys_encode(h, big, bl, 1, &b1, &n1b) == 0,
              "big payload sequential");
        check(tsimd_sys_encode(h, big, bl, 0, &b0, &n0b) == 0,
              "big payload automatic");
        check(tsimd_sys_encode(h, big, bl, 6, &b6, &n6b) == 0,
              "big payload 6 threads");
        check(n1b == n0b && n0b == n6b && n1b > 0 &&
                  memcmp(b1, b0, n1b * sizeof(uint32_t)) == 0 &&
                  memcmp(b1, b6, n1b * sizeof(uint32_t)) == 0,
              "big auto and 6-thread ids equal the sequential ids");
        {
            int d2 = tsimd_default_threads();
            check(d_threads0 < 2 || d2 == d_threads0,
                  "default thread count is stable across a pinned call");
        }
        tsimd_sys_free(b1);
        tsimd_sys_free(b0);
        tsimd_sys_free(b6);
        {
            const char *longer =
                "A considerably longer sentence, repeated to force the "
                "reuse buffer to grow beyond its first size. ";
            size_t nl = strlen(longer);
            uint32_t *ref = NULL;
            size_t nr = 0;
            check(tsimd_sys_encode(h, (const uint8_t *)longer, nl, 1, &ref,
                                   &nr) == 0,
                  "reference encode for the reuse checks");
            check(tsimd_sys_encode_own(h, (const uint8_t *)longer, nl, 1) == 0,
                  "encode own growth");
            own = tsimd_sys_ids(h, &n_own);
            check(own != NULL && n_own == nr &&
                      memcmp(own, ref, nr * sizeof(uint32_t)) == 0,
                  "grown-buffer ids equal the reference");
            const uint32_t *p_grown = own;
            check(tsimd_sys_encode_own(h, big, bl, 1) == 0,
                  "encode own on the big payload");
            own = tsimd_sys_ids(h, &n_own);
            check(n_own == n1b, "big own id count");
            const uint32_t *p_big = own;
            check(tsimd_sys_encode_own(h, (const uint8_t *)text,
                                       strlen(text), 1) == 0,
                  "encode own shrink");
            own = tsimd_sys_ids(h, &n_own);
            check(own == p_big, "the reuse buffer survives the smaller call");
            check(n_own == n_ids &&
                      memcmp(own, ids, n_ids * sizeof(uint32_t)) == 0,
                  "shrunk-call ids equal the sequential ids");
            tsimd_sys_free(ref);
        }
        free(big);
    }
    free(into);
    tsimd_sys_free(ids);
    tsimd_sys_close(h);

    h = tsimd_sys_open(T5, "sentencepiece", "auto", err, sizeof err);
    check(h != NULL, "open t5");
    if (h) {
        check(tsimd_sys_vocab_size(h) == 32000, "t5 vocab size");
        tsimd_sys_close(h);
    }
    h = tsimd_sys_open(QWEN, "json", "auto", err, sizeof err);
    check(h != NULL, "open qwen2.5");
    if (h) tsimd_sys_close(h);

    h = tsimd_sys_open("/nonexistent/model.json", "json", "auto", err,
                       sizeof err);
    check(h == NULL && err[0] != 0, "a missing model must fail with a message");
    h = tsimd_sys_open(CL100K, "nonsense", "auto", err, sizeof err);
    check(h == NULL && err[0] != 0, "a bad format must fail with a message");
    h = tsimd_sys_open(CL100K, "tiktoken", "nonsense", err, sizeof err);
    check(h == NULL && err[0] != 0, "a bad scheme must fail with a message");
    h = tsimd_sys_open(NULL, "json", "auto", err, sizeof err);
    check(h == NULL, "a NULL path must fail");
    tsimd_sys_close(NULL);

    if (fails) {
        fprintf(stderr, "%d sys API checks FAILED\n", fails);
        return 1;
    }
    printf("test_sys: the opaque C ABI loads, encodes, decodes and reports "
           "errors\n");
    return 0;
}
