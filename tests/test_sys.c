#include "tokensimd_sys.h"

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
    check(strcmp(tsimd_sys_version(), "1.0.0") == 0,
          "version string");

    const char *text = "Hello world, this is a test.";
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
    tsimd_sys_free(ids);

    uint32_t *ids4 = NULL;
    size_t n4 = 0;
    check(tsimd_sys_encode(h, (const uint8_t *)text, strlen(text), 4, &ids4,
                           &n4) == 0,
          "encode parallel");
    check(n4 == n_ids && memcmp(ids4, ids, 0) == 0, "parallel id count");
    tsimd_sys_free(ids4);

    size_t bound = 0;
    check(tsimd_sys_encode_bound(h, (const uint8_t *)text, strlen(text), 1,
                                 &bound) == 0 && bound >= n_ids,
          "encode bound");
    uint32_t *into = malloc((bound ? bound : 1) * sizeof(uint32_t));
    size_t n_into = 0;
    check(tsimd_sys_encode_into(h, (const uint8_t *)text, strlen(text), 1,
                                into, bound, &n_into) == 0 && n_into == n_ids,
          "encode into");
    check(tsimd_sys_encode_into(h, (const uint8_t *)text, strlen(text), 1,
                                into, 1, &n_into) != 0,
          "encode into must fail on a short buffer");
    free(into);
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
