#include "bpe.h"
#include "tokenizer.h"

#include <stdio.h>
#include <string.h>

#define GLM4 TSIMD_SRC_DIR "/models/reference/glm4_tokenizer.json"

static int expect_ids(tsimd_tokenizer_t *t, const char *text, size_t len,
                      const uint32_t *want, size_t n) {
    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    if (tsimd_tokenizer_encode(t, (const uint8_t *)text, len, &seq) != 0) {
        fprintf(stderr, "FAIL: encode error for %s\n", text);
        return 1;
    }
    int rc = 0;
    if (seq.count != n) {
        fprintf(stderr, "FAIL: %s: %zu ids want %zu\n", text,
                (size_t)seq.count, n);
        rc = 1;
    }
    for (size_t i = 0; i < n && i < seq.count; i++) {
        if (seq.ids[i] != want[i]) {
            fprintf(stderr,
                    "FAIL: %s: id %zu = %u want %u\n", text, i, seq.ids[i],
                    want[i]);
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
    tsimd_pretok_scheme_t scheme = TSIMD_PRETOK_GPT2;
    if (tsimd_tokenizer_load_model(&t, TSIMD_MODEL_TOKENIZER_JSON, GLM4,
                                   NULL, &scheme, NULL, 0) != 0) {
        fprintf(stderr, "load failed\n");
        return 2;
    }
    int rc = 0;
    {
        uint32_t want[] = {147145};
        rc |= expect_ids(&t, " enkele", 7, want, 1);
    }
    {
        uint32_t want[] = {268, 440, 273};
        rc |= expect_ids(&t, "enkele", 6, want, 3);
    }
    {
        uint32_t want[] = {144283};
        rc |= expect_ids(&t, " enkel", 6, want, 1);
    }
    {
        uint32_t want[] = {273};
        rc |= expect_ids(&t, "le", 2, want, 1);
    }
    {
        uint32_t want[] = {5242};
        rc |= expect_ids(&t, " van", 4, want, 1);
    }
    tsimd_tokenizer_free(&t);
    if (rc) return 1;
    printf("test_ignore_merges: OK\n");
    return 0;
}
